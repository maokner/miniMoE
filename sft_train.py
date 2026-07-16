"""
Supervised fine-tuning (SFT) for miniMoE.

Loads a pretrained base checkpoint and fine-tunes it on the instruction data
prepared by sft_data.py, so the model learns the User/Assistant chat format,
answers instead of merely continuing text, and stops cleanly on EOS.

Key differences from pretraining (train.py):
  - Loss is computed only on assistant tokens. The packed shards carry a mask;
    non-target positions become -100 labels, which F.cross_entropy ignores.
  - A much lower LR and a short cosine schedule over a few epochs, to adapt the
    model without washing out what it learned in pretraining.
  - The load-balancing aux loss stays on so experts don't collapse during SFT.

Multi-GPU via DDP, mirroring train.py. Each rank owns a disjoint, reshuffled
slice of the packed blocks, so one pass over the data is split across GPUs.
The effective batch is BATCH_SIZE * GRAD_ACCUM_STEPS * world_size.

Single GPU:
    BASE_CHECKPOINT=minimoe_step_0019073.pt python sft_train.py

Multi-GPU (e.g. 2 GPUs):
    BASE_CHECKPOINT=minimoe_step_0019073.pt \
      torchrun --standalone --nproc_per_node=2 sft_train.py

Config via env vars (see the block in main()).
"""
import csv
import glob
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Model, ModelConfig

IGNORE_INDEX = -100


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


class SFTBlockLoader:
    """Streams fixed-length blocks from the packed SFT shards.

    Each shard is a (tokens, mask) pair written by sft_data.py. We build a flat
    index of every full block across all shards. Under DDP, each epoch we shuffle
    that index with an epoch-derived seed that is identical on every rank, then
    hand rank r a disjoint contiguous slice of the shuffled order. Because the
    slice size is floor(total / world_size), every rank yields exactly the same
    number of batches - which DDP requires, or ranks would deadlock waiting on a
    gradient all-reduce that never comes. Shards are memory-mapped.
    """

    def __init__(self, split, batch_size, block_size, data_dir,
                 rank=0, world_size=1, seed=1234):
        self.batch_size = batch_size
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

        tok_files = sorted(glob.glob(os.path.join(data_dir, f"sft_{split}_*_tokens.npy")))
        if not tok_files:
            raise FileNotFoundError(
                f"No SFT {split} shards in {data_dir}. Run sft_data.py first."
            )

        self.token_shards = []
        self.mask_shards = []
        self.index = []  # (shard_id, start_offset)
        for shard_id, tok_path in enumerate(tok_files):
            mask_path = tok_path.replace("_tokens.npy", "_mask.npy")
            tokens = np.load(tok_path, mmap_mode="r")
            mask = np.load(mask_path, mmap_mode="r")
            self.token_shards.append(tokens)
            self.mask_shards.append(mask)
            # +1 token is needed to form the shifted target, so the last full
            # block starts no later than len - (block_size + 1).
            n_blocks = (len(tokens) - 1) // block_size
            for b in range(n_blocks):
                self.index.append((shard_id, b * block_size))

        # Blocks each rank consumes per epoch (floor, so all ranks match).
        self.blocks_per_rank = len(self.index) // world_size
        self.batches_per_epoch = self.blocks_per_rank // batch_size
        if self.batches_per_epoch == 0:
            raise ValueError(
                f"SFT {split} data has fewer than one batch per rank "
                f"({len(self.index)} blocks, world_size {world_size}, "
                f"batch_size {batch_size})."
            )

    def epoch(self, epoch_idx=0):
        """Yield (x, labels) tensors for this rank's slice of one shuffled pass.

        The permutation is seeded identically on all ranks so the disjoint
        partition is consistent; it changes every epoch so ranks see fresh data.
        """
        rng = np.random.default_rng(self.seed + epoch_idx)
        order = rng.permutation(len(self.index))
        start = self.rank * self.blocks_per_rank
        my_blocks = order[start:start + self.blocks_per_rank]

        bs, bl = self.batch_size, self.block_size
        for b0 in range(0, self.batches_per_epoch * bs, bs):
            picks = my_blocks[b0:b0 + bs]
            x = np.empty((bs, bl), dtype=np.int64)
            labels = np.empty((bs, bl), dtype=np.int64)
            for row, pick in enumerate(picks):
                shard_id, off = self.index[pick]
                toks = np.asarray(
                    self.token_shards[shard_id][off:off + bl + 1], dtype=np.int64
                )
                msk = np.asarray(
                    self.mask_shards[shard_id][off:off + bl + 1], dtype=np.int64
                )
                x[row] = toks[:-1]
                y = toks[1:].copy()
                y[msk[1:] == 0] = IGNORE_INDEX
                labels[row] = y
            yield torch.from_numpy(x), torch.from_numpy(labels)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def autocast_context(device):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def sync_device(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


CSV_FIELDS = ["phase", "epoch", "step", "train_loss", "val_loss", "lr", "grad_norm", "dt_ms"]


def write_csv_row(path, row):
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    """Estimate val loss on the master rank over the full val set.

    DDP forward does not run collectives (only backward does), so it is safe to
    call this on the master alone; the other ranks simply block at the next
    all-reduce until the master rejoins.
    """
    model.eval()
    losses = []
    for i, (x, labels) in enumerate(loader.epoch(0)):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device):
            _, loss = model(x, labels)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def save_checkpoint(path, model, optimizer, model_config, step, tokens_seen):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": vars(model_config),
            "step": step,
            "tokens_seen": tokens_seen,
        },
        path,
    )


def main():
    torch.manual_seed(67)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(67)
    torch.set_float32_matmul_precision("high")

    # ---- DDP / device setup (mirrors train.py) ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = get_device()

    base_checkpoint = os.environ.get("BASE_CHECKPOINT", "minimoe_step_0019073.pt")
    data_dir = os.environ.get("SFT_DATA_DIR", "sft_data")
    block_size = env_int("BLOCK_SIZE", 1024)
    batch_size = env_int("BATCH_SIZE", 16)
    grad_accum_steps = env_int("GRAD_ACCUM_STEPS", 4)
    epochs = env_int("EPOCHS", 3)
    max_lr = env_float("MAX_LR", 2e-5)
    min_lr = env_float("MIN_LR", max_lr / 10)
    warmup_steps = env_int("WARMUP_STEPS", 50)
    weight_decay = env_float("WEIGHT_DECAY", 0.1)
    eval_interval = env_int("EVAL_INTERVAL", 200)
    eval_batches = env_int("EVAL_BATCHES", 50)
    log_interval = env_int("LOG_INTERVAL", 20)
    use_compile = env_int("TORCH_COMPILE", 1 if device.startswith("cuda") else 0) != 0
    out_dir = os.environ.get("SFT_CHECKPOINT_DIR", "checkpoints")
    out_name = os.environ.get("SFT_CHECKPOINT_NAME", "minimoe_sft.pt")
    log_file = os.environ.get("SFT_LOG_FILE", "sft_log.csv")

    if not os.path.exists(base_checkpoint):
        raise SystemExit(f"Base checkpoint not found: {base_checkpoint}")

    if master_process:
        print(f"Using device: {device}, world_size: {ddp_world_size}")

    train_loader = SFTBlockLoader(
        "train", batch_size, block_size, data_dir,
        rank=ddp_rank, world_size=ddp_world_size,
    )
    # Validation runs only on the master over the full val set (world_size 1).
    val_loader = None
    if master_process:
        try:
            val_loader = SFTBlockLoader("val", batch_size, block_size, data_dir)
        except (FileNotFoundError, ValueError) as err:
            # Missing or too-small val set: skip validation rather than crash the
            # master (which would hang the other ranks until the NCCL timeout).
            print(f"Skipping validation: {err}")

    steps_per_epoch = train_loader.batches_per_epoch // grad_accum_steps
    max_steps = steps_per_epoch * epochs
    tokens_per_step = batch_size * block_size * grad_accum_steps * ddp_world_size
    if steps_per_epoch == 0:
        raise SystemExit(
            "grad_accum_steps is larger than the batches available per epoch; "
            "lower GRAD_ACCUM_STEPS or BATCH_SIZE."
        )
    if master_process:
        print(f"base checkpoint: {base_checkpoint}")
        print(f"train blocks: {len(train_loader.index):,} "
              f"({train_loader.blocks_per_rank:,}/rank), "
              f"batch {batch_size} x accum {grad_accum_steps} x {ddp_world_size} GPU")
        print(f"optimizer steps/epoch: {steps_per_epoch}, epochs: {epochs}, "
              f"total steps: {max_steps}")
        print(f"lr: max {max_lr:.2e} -> min {min_lr:.2e}, warmup {warmup_steps}")
        print(f"effective tokens/optimizer step: {tokens_per_step:,}")

    checkpoint = torch.load(base_checkpoint, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    base_model = Model(model_config)
    base_model.load_state_dict(checkpoint["model"])
    base_model.to(device)
    if master_process:
        print(f"loaded {sum(p.numel() for p in base_model.parameters()):,} parameters")

    model = torch.compile(base_model) if use_compile else base_model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
    optimizer = base_model.configure_optimizers(weight_decay, max_lr, device)

    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        loss_accum = torch.zeros((), device=device)
        micro = 0
        optimizer.zero_grad()
        for x, labels in train_loader.epoch(epoch):
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Skip the gradient all-reduce until the last micro-step of the
            # accumulation window (DDP performs the reduce on the final backward).
            is_last_micro = (micro == grad_accum_steps - 1)
            sync_context = (
                model.no_sync() if (ddp and not is_last_micro) else nullcontext()
            )
            with sync_context:
                with autocast_context(device):
                    _, loss = model(x, labels)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

            micro += 1
            if micro < grad_accum_steps:
                continue

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            micro = 0

            # Average the per-step loss across ranks for logging. This is a
            # collective, so every rank must call it every step - guarding it
            # behind master_process or a log interval would deadlock the group.
            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            step_is_log = step % log_interval == 0 or step == 1
            if master_process and step_is_log:
                sync_device(device)
                dt = time.time() - t0
                t0 = time.time()
                step_loss = loss_accum.item()
                print(
                    f"epoch {epoch} step {step}/{max_steps} "
                    f"loss {step_loss:.4f} lr {lr:.2e} "
                    f"grad_norm {norm.item():.3f} dt {1000 * dt / log_interval:.1f} ms/step",
                    flush=True,
                )
                write_csv_row(log_file, {
                    "phase": "train", "epoch": epoch, "step": step,
                    "train_loss": f"{step_loss:.6f}", "val_loss": "",
                    "lr": f"{lr:.3e}", "grad_norm": f"{norm.item():.4f}",
                    "dt_ms": f"{1000 * dt / log_interval:.2f}",
                })
            loss_accum = torch.zeros((), device=device)

            # Validation runs on the master only (val_loader is None elsewhere).
            # evaluate() uses no collectives, so the other ranks simply wait at
            # the next backward all-reduce until the master rejoins.
            if val_loader is not None and eval_interval > 0 and step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, eval_batches)
                print(f"  [eval] step {step} val_loss {val_loss:.4f}", flush=True)
                write_csv_row(log_file, {
                    "phase": "eval", "epoch": epoch, "step": step,
                    "train_loss": "", "val_loss": f"{val_loss:.6f}",
                    "lr": f"{lr:.3e}", "grad_norm": "", "dt_ms": "",
                })
                t0 = time.time()

        # End-of-epoch checkpoint on the master; other ranks wait at the barrier.
        if master_process:
            epoch_path = os.path.join(out_dir, f"minimoe_sft_epoch{epoch + 1}.pt")
            save_checkpoint(epoch_path, base_model, optimizer, model_config,
                            step, step * tokens_per_step)
            print(f"saved {epoch_path}", flush=True)
        if ddp:
            dist.barrier()

    if master_process:
        final_path = os.path.join(out_dir, out_name)
        save_checkpoint(final_path, base_model, optimizer, model_config,
                        step, step * tokens_per_step)
        print(f"saved final SFT checkpoint: {final_path}", flush=True)

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

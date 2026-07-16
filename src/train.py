import csv
import glob
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from hellaswag import (
    completion_losses,
    download as download_hellaswag,
    iterate_examples,
    render_example,
)
from model import Model, ModelConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, init_process_group


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def has_split_shards(split, data_dir):
    pattern = os.path.join(data_dir, f"edufineweb_{split}_*.npy")
    return bool(glob.glob(pattern))


class FineWebTokenLoader:
    def __init__(self, split, batch_size, block_size, rank=0, world_size=1, data_dir="edu_fineweb10B"):
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.batch_tokens = batch_size * block_size
        self.data_dir = data_dir

        pattern = os.path.join(data_dir, f"edufineweb_{split}_*.npy")
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(
                f"No FineWeb {split} shards found at {pattern}. Run src/fineweb.py first."
            )

        self.current_shard = 0
        self.tokens = None
        self.current_position = 0
        self._load_shard(0)

    def _load_shard(self, shard_index):
        self.current_shard = shard_index
        self.tokens = np.load(self.files[self.current_shard], mmap_mode="r")
        self.current_position = self.batch_tokens * self.rank

        required = self.current_position + self.batch_tokens + 1
        if len(self.tokens) < required:
            raise ValueError(
                f"Shard {self.files[self.current_shard]} has {len(self.tokens):,} tokens, "
                f"but rank {self.rank} needs at least {required:,}."
            )

    def next_batch(self):
        required = self.current_position + self.batch_tokens + 1
        if required > len(self.tokens):
            next_shard = (self.current_shard + 1) % len(self.files)
            self._load_shard(next_shard)

        buf_np = np.asarray(
            self.tokens[self.current_position : self.current_position + self.batch_tokens + 1],
            dtype=np.int64,
        )
        buf = torch.from_numpy(buf_np)
        x = buf[:-1].view(self.batch_size, self.block_size)
        y = buf[1:].view(self.batch_size, self.block_size)

        self.current_position += self.batch_tokens * self.world_size
        return x, y

    def skip_batches(self, num_batches):
        # Advance the loader as if num_batches had been consumed, without
        # materializing any data. Used when resuming from a checkpoint.
        for _ in range(num_batches):
            required = self.current_position + self.batch_tokens + 1
            if required > len(self.tokens):
                self._load_shard((self.current_shard + 1) % len(self.files))
            self.current_position += self.batch_tokens * self.world_size


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate(model, loader, device, eval_steps, autocast_device, ddp=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = loader.next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context(autocast_device):
                _, loss = model(x, y)
            total_loss += loss.item()
    model.train()

    avg_loss = torch.tensor(total_loss / eval_steps, device=device)
    if ddp:
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    return avg_loss.item()


def evaluate_hellaswag(
    model,
    device,
    max_examples,
    autocast_device,
    max_seq_length,
    split="val",
    rank=0,
    world_size=1,
    ddp=False,
):
    model.eval()
    correct = 0
    correct_norm = 0
    total = 0
    skipped = 0

    with torch.no_grad():
        for example_index, example in enumerate(iterate_examples(split)):
            if max_examples > 0 and example_index >= max_examples:
                break
            if example_index % world_size != rank:
                continue

            _, tokens, mask, label = render_example(example)
            if tokens.size(1) > max_seq_length:
                skipped += 1
                continue
            if tokens.size(1) < max_seq_length:
                pad_len = max_seq_length - tokens.size(1)
                tokens = torch.nn.functional.pad(tokens, (0, pad_len))
                mask = torch.nn.functional.pad(mask, (0, pad_len))

            tokens = tokens.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with autocast_context(autocast_device):
                logits, _ = model(tokens)

            sum_loss, avg_loss = completion_losses(logits, tokens, mask)
            pred = sum_loss.argmin().item()
            pred_norm = avg_loss.argmin().item()

            total += 1
            correct += int(pred == label)
            correct_norm += int(pred_norm == label)

    stats = torch.tensor(
        [correct, correct_norm, total, skipped],
        device=device,
        dtype=torch.float64,
    )
    if ddp:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    model.train()
    total = int(stats[2].item())
    if total == 0:
        return {
            "acc": None,
            "acc_norm": None,
            "correct": int(stats[0].item()),
            "correct_norm": int(stats[1].item()),
            "total": 0,
            "skipped": int(stats[3].item()),
        }

    return {
        "acc": stats[0].item() / total,
        "acc_norm": stats[1].item() / total,
        "correct": int(stats[0].item()),
        "correct_norm": int(stats[1].item()),
        "total": total,
        "skipped": int(stats[3].item()),
    }


CSV_FIELDS = [
    "phase",
    "step",
    "tokens_seen",
    "train_loss",
    "val_loss",
    "test_loss",
    "hellaswag_acc",
    "hellaswag_acc_norm",
    "hellaswag_total",
    "hellaswag_dt_ms",
    "lr",
    "grad_norm",
    "dt_ms",
    "eval_dt_ms",
    "tokens_per_sec",
]


def write_csv_row(path, row):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, model_config, step, tokens_seen):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": vars(model_config),
        "step": step,
        "tokens_seen": tokens_seen,
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)


def sync_device(device):
    if device == "cuda" or device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def autocast_context(autocast_device):
    if autocast_device == "cuda":
        return torch.autocast(device_type=autocast_device, dtype=torch.bfloat16)
    return nullcontext()


def main():
    torch.manual_seed(67)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(67)

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
        print(f"Using device: {device}")

    data_dir = os.environ.get("DATA_DIR", "edu_fineweb10B")
    total_batch_size = env_int("TOTAL_BATCH_SIZE", 2**19)
    block_size = env_int("BLOCK_SIZE", 1024)
    batch_size = env_int("BATCH_SIZE", 32)
    max_steps = env_int("MAX_STEPS", 19073)
    eval_steps = env_int("EVAL_STEPS", 10)
    eval_interval = env_int("EVAL_INTERVAL", 100)
    log_interval = env_int("LOG_INTERVAL", 100)
    hellaswag_interval = env_int("HELLASWAG_INTERVAL", 100)
    hellaswag_examples = env_int("HELLASWAG_EXAMPLES", 32)
    checkpoint_interval = env_int("CHECKPOINT_INTERVAL", 5000)
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints")
    test_steps = env_int("TEST_STEPS", 10)
    test_at_end = env_int("TEST_AT_END", 1) != 0
    log_file = os.environ.get("LOG_FILE", "train_log.csv")
    warmup_steps = env_int("WARMUP_STEPS", 715)
    max_lr = env_float("MAX_LR", 6e-4)
    min_lr = env_float("MIN_LR", max_lr / 10)
    weight_decay = env_float("WEIGHT_DECAY", 0.1)
    use_compile = env_int("TORCH_COMPILE", 1) != 0
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", "")

    assert total_batch_size % (batch_size * block_size * ddp_world_size) == 0, (
        "total_batch_size must be divisible by batch_size * block_size * world_size"
    )
    grad_accum_steps = total_batch_size // (batch_size * block_size * ddp_world_size)

    if master_process:
        print(f"data dir: {data_dir}")
        print(f"total desired batch size: {total_batch_size}")
        print(f"gradient accumulation steps: {grad_accum_steps}")
        print(f"max steps: {max_steps}")
        print(f"planned training tokens: {max_steps * total_batch_size:,}")
        print(f"warmup tokens: {warmup_steps * total_batch_size:,}")
        print(f"lr: max {max_lr:.2e}, min {min_lr:.2e}, warmup {warmup_steps} steps")
        print(f"weight decay: {weight_decay}")
        print(f"log file: {log_file}")
        if hellaswag_interval > 0 and hellaswag_examples != 0:
            print(
                f"HellaSwag eval: every {hellaswag_interval} steps, "
                f"{hellaswag_examples if hellaswag_examples > 0 else 'all'} examples"
            )
        if checkpoint_interval > 0:
            print(
                f"checkpoints: every {checkpoint_interval} steps, "
                f"directory: {checkpoint_dir}"
            )

    train_loader = FineWebTokenLoader(
        "train",
        batch_size,
        block_size,
        ddp_rank,
        ddp_world_size,
        data_dir,
    )
    val_loader = None
    if eval_interval > 0:
        val_loader = FineWebTokenLoader(
            "val",
            batch_size,
            block_size,
            ddp_rank,
            ddp_world_size,
            data_dir,
        )

    test_loader = None
    if test_at_end:
        if has_split_shards("test", data_dir):
            test_loader = FineWebTokenLoader(
                "test",
                batch_size,
                block_size,
                ddp_rank,
                ddp_world_size,
                data_dir,
            )
        elif master_process:
            print("No FineWeb test shards found; final test evaluation will be skipped.")

    torch.set_float32_matmul_precision("high")
    model_config = ModelConfig(max_seq_length=block_size)
    base_model = Model(model_config)
    base_model.to(device)
    if master_process:
        total_params = sum(p.numel() for p in base_model.parameters())
        print(f"model parameters: {total_params:,}")
    model = torch.compile(base_model) if use_compile else base_model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

    optimizer = base_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=max_lr,
        device=device,
    )

    start_step = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        base_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        train_loader.skip_batches(start_step * grad_accum_steps)
        if master_process:
            print(
                f"Resumed from {resume_checkpoint} at step {start_step} "
                f"({checkpoint['tokens_seen']:,} tokens seen)"
            )

    if hellaswag_interval > 0 and hellaswag_examples != 0:
        # Download once from the master process; letting every rank race on the
        # same file can corrupt it.
        if master_process:
            download_hellaswag("val")
        if ddp:
            dist.barrier()

    autocast_device = "cuda" if device.startswith("cuda") else device

    for step in range(start_step, max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sync_context = (
                model.no_sync()
                if ddp and micro_step < grad_accum_steps - 1
                else nullcontext()
            )
            with sync_context:
                with autocast_context(autocast_device):
                    _, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        sync_device(device)

        t1 = time.time()
        tokens_processed = grad_accum_steps * batch_size * block_size * ddp_world_size
        tokens_per_sec = tokens_processed / (t1 - t0)

        step_number = step + 1
        tokens_seen = step_number * total_batch_size
        should_log = (
            step_number == 1
            or step_number % log_interval == 0
            or step_number == max_steps
        )
        should_eval = (
            val_loader is not None
            and (step_number % eval_interval == 0 or step_number == max_steps)
        )
        should_hellaswag = (
            hellaswag_interval > 0
            and hellaswag_examples != 0
            and (step_number % hellaswag_interval == 0 or step_number == max_steps)
        )
        should_checkpoint = (
            checkpoint_interval > 0
            and (step_number % checkpoint_interval == 0 or step_number == max_steps)
        )

        val_loss = None
        eval_dt_ms = ""
        hellaswag_stats = None
        hellaswag_dt_ms = ""
        if should_eval:
            eval_t0 = time.time()
            val_loss = evaluate(
                model,
                val_loader,
                device,
                eval_steps,
                autocast_device,
                ddp,
            )
            eval_dt_ms = f"{1000 * (time.time() - eval_t0):.4f}"

        if should_hellaswag:
            hellaswag_t0 = time.time()
            hellaswag_stats = evaluate_hellaswag(
                model,
                device,
                hellaswag_examples,
                autocast_device,
                block_size,
                split="val",
                rank=ddp_rank,
                world_size=ddp_world_size,
                ddp=ddp,
            )
            hellaswag_dt_ms = f"{1000 * (time.time() - hellaswag_t0):.4f}"

        if master_process and (should_log or val_loss is not None or hellaswag_stats is not None):
            message = (
                f"Step {step_number}, Loss: {loss_accum.item():.4f}, "
                f"norm: {norm.item():.4f}, lr: {lr:.2e}, "
                f"dt: {1000 * (t1 - t0):.2f} ms, tok/sec: {tokens_per_sec:.2f}"
            )
            if val_loss is not None:
                message += f", val_loss: {val_loss:.4f}"
                message += f", eval_dt: {eval_dt_ms} ms"
            if hellaswag_stats is not None and hellaswag_stats["acc_norm"] is not None:
                message += (
                    f", hellaswag_acc: {hellaswag_stats['acc']:.4f}, "
                    f"hellaswag_acc_norm: {hellaswag_stats['acc_norm']:.4f}, "
                    f"hellaswag_n: {hellaswag_stats['total']}, "
                    f"hellaswag_dt: {hellaswag_dt_ms} ms"
                )
            print(message, flush=True)

            write_csv_row(
                log_file,
                {
                    "phase": "train",
                    "step": step_number,
                    "tokens_seen": tokens_seen,
                    "train_loss": f"{loss_accum.item():.6f}",
                    "val_loss": "" if val_loss is None else f"{val_loss:.6f}",
                    "test_loss": "",
                    "hellaswag_acc": (
                        ""
                        if hellaswag_stats is None or hellaswag_stats["acc"] is None
                        else f"{hellaswag_stats['acc']:.6f}"
                    ),
                    "hellaswag_acc_norm": (
                        ""
                        if hellaswag_stats is None or hellaswag_stats["acc_norm"] is None
                        else f"{hellaswag_stats['acc_norm']:.6f}"
                    ),
                    "hellaswag_total": (
                        "" if hellaswag_stats is None else hellaswag_stats["total"]
                    ),
                    "hellaswag_dt_ms": hellaswag_dt_ms,
                    "lr": f"{lr:.8e}",
                    "grad_norm": f"{norm.item():.6f}",
                    "dt_ms": f"{1000 * (t1 - t0):.4f}",
                    "eval_dt_ms": eval_dt_ms,
                    "tokens_per_sec": f"{tokens_per_sec:.4f}",
                },
            )

        if should_checkpoint:
            if master_process:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"minimoe_step_{step_number:07d}.pt",
                )
                save_checkpoint(
                    checkpoint_path,
                    base_model,
                    optimizer,
                    model_config,
                    step_number,
                    tokens_seen,
                )
                print(f"Saved checkpoint: {checkpoint_path}", flush=True)
            if ddp:
                dist.barrier()

    if test_loader is not None:
        test_t0 = time.time()
        test_loss = evaluate(
            model,
            test_loader,
            device,
            test_steps,
            autocast_device,
            ddp,
        )
        test_dt_ms = 1000 * (time.time() - test_t0)
        if master_process:
            print(f"Final test loss: {test_loss:.4f}, test_dt: {test_dt_ms:.4f} ms", flush=True)
            write_csv_row(
                log_file,
                {
                    "phase": "test",
                    "step": max_steps,
                    "tokens_seen": max_steps * total_batch_size,
                    "train_loss": "",
                    "val_loss": "",
                    "test_loss": f"{test_loss:.6f}",
                    "hellaswag_acc": "",
                    "hellaswag_acc_norm": "",
                    "hellaswag_total": "",
                    "hellaswag_dt_ms": "",
                    "lr": "",
                    "grad_norm": "",
                    "dt_ms": "",
                    "eval_dt_ms": f"{test_dt_ms:.4f}",
                    "tokens_per_sec": "",
                },
            )

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

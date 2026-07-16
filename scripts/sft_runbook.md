# miniMoE SFT runbook

Supervised fine-tuning of the pretrained miniMoE base checkpoint into a small
instruction-following chat model.

This trains on rented NVIDIA GPUs via DDP (`torchrun`). It scales cleanly across
GPUs; two configs that both finish 2 epochs comfortably inside a few hours:

- **2x A100 80GB** - `BATCH_SIZE=64` per GPU, simplest and most headroom.
- **4x A6000 48GB** - `BATCH_SIZE=32` per GPU, similar throughput, usually cheaper.

A single GPU also works (drop `torchrun`), just slower. It does **not** run on the
8GB M1 the base model was sampled on - SFT needs params + gradients + AdamW state +
activations, several times the memory inference alone already saturates locally.

## What SFT does here

The base model is a FineWeb-Edu text continuer with no notion of a conversation.
SFT teaches it three things: the `User:`/`Assistant:` chat format, to answer a
prompt instead of continuing it, and to stop cleanly by emitting EOS. It does
**not** make the model smart - at 280M params (110M active) the ceiling is short,
simple instruction-following, not a real assistant.

## 0. Prerequisites on the GPU box

```bash
pip install -r requirements.txt
# Fetch the base checkpoint (3.4 GB) from the HF model repo (note /resolve/):
wget -O minimoe_step_0019073.pt \
  https://huggingface.co/mokner123/miniMoE/resolve/main/minimoe_step_0019073.pt
ls -lh minimoe_step_0019073.pt   # must read ~3.4G, not a few KB
```

## 1. Prepare the SFT data

Downloads `HuggingFaceTB/smol-smoltalk` (the small-model SmolTalk variant),
renders it with the canonical chat template, masks everything but assistant
tokens, and packs 1024-token shards into `sft_data/`.

```bash
python src/sft_data.py
```

Swap the dataset with env vars if you want a different behavior mix:

```bash
DATASET=HuggingFaceTB/smoltalk DATASET_CONFIG=all python src/sft_data.py   # larger, broader
```

Sanity check the printed summary: the "supervised %" is the fraction of tokens
that carry loss (assistant content). For smol-smoltalk this is typically ~40-55%.

## 2. Run SFT (DDP)

**2x A100 80GB:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EPOCHS=2 BATCH_SIZE=64 GRAD_ACCUM_STEPS=1 \
BASE_CHECKPOINT=minimoe_step_0019073.pt \
  torchrun --standalone --nproc_per_node=2 src/sft_train.py
```

**4x A6000 48GB:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EPOCHS=2 BATCH_SIZE=32 GRAD_ACCUM_STEPS=1 \
BASE_CHECKPOINT=minimoe_step_0019073.pt \
  torchrun --standalone --nproc_per_node=4 src/sft_train.py
```

`--nproc_per_node` must equal your GPU count. The effective batch is
`BATCH_SIZE * GRAD_ACCUM_STEPS * nproc_per_node`; each GPU owns a disjoint,
reshuffled slice of the data, so more GPUs means fewer optimizer steps per epoch
and proportionally less wall time.

Other defaults (override with env vars): `MAX_LR=2e-5` cosine decay,
`WARMUP_STEPS=50`, bf16 autocast + `torch.compile` on CUDA. Loss on assistant
tokens only. Single GPU: drop `torchrun` and run `python src/sft_train.py`.

Checkpoints land in `checkpoints/`: one per epoch (`minimoe_sft_epoch{N}.pt`)
plus a final `minimoe_sft.pt`, written by the master rank only. They use the same
format as the base checkpoint, so `src/sample.py` loads them directly.

If a GPU OOMs, lower `BATCH_SIZE` (e.g. A6000 -> 24) and/or set `TORCH_COMPILE=0`.

Watch `sft_log.csv` (or stdout): training loss should fall steadily and the
periodic `val_loss` should track it. If val loss turns up while train loss keeps
falling, you are overfitting - prefer the earlier epoch checkpoint or drop to 2
epochs / lower LR.

## 3. Try it

Pull a checkpoint back down (or sample on the box) and chat with the SFT template:

```bash
python src/sample.py -c checkpoints/minimoe_sft.pt --sft
```

```
User: What is the capital of France?
Assistant: ...
```

`--sft` uses the exact training format and stops on EOS. Compare against the base
model's `--chat` (which just wraps the continuer) to see what SFT bought you.

## Tuning notes

- **Underfit / ignores the format:** more epochs, or `MAX_LR=3e-5`.
- **Repetitive or degenerate replies:** lower LR (`1e-5`), fewer epochs; sample
  with a lower temperature (`--temperature 0.6`).
- **OOM on a small card:** drop `BATCH_SIZE` to 8 and raise `GRAD_ACCUM_STEPS` to
  8 to keep the same effective batch, or set `TORCH_COMPILE=0`.
- **Different behavior target:** change the dataset in step 1; nothing else moves.

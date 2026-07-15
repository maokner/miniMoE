# miniMoE SFT runbook

Supervised fine-tuning of the pretrained miniMoE base checkpoint into a small
instruction-following chat model.

This runs on a **single rented GPU** (one A100 40GB, or a 24GB card like a 4090/L4
is plenty for a 280M model). It does **not** run on the 8GB M1 the base model was
sampled on - SFT needs params + gradients + AdamW state + activations, several
times the memory that inference alone already saturates locally.

Expect the whole run to finish in well under an hour and cost a couple of dollars.

## What SFT does here

The base model is a FineWeb-Edu text continuer with no notion of a conversation.
SFT teaches it three things: the `User:`/`Assistant:` chat format, to answer a
prompt instead of continuing it, and to stop cleanly by emitting EOS. It does
**not** make the model smart - at 280M params (110M active) the ceiling is short,
simple instruction-following, not a real assistant.

## 0. Prerequisites on the GPU box

```bash
pip install -r requirements.txt
# Copy the base checkpoint up to the box (3.4 GB):
#   scp minimoe_step_0019073.pt user@box:/workspace/miniMoE/
```

## 1. Prepare the SFT data

Downloads `HuggingFaceTB/smol-smoltalk` (the small-model SmolTalk variant),
renders it with the canonical chat template, masks everything but assistant
tokens, and packs 1024-token shards into `sft_data/`.

```bash
python sft_data.py
```

Swap the dataset with env vars if you want a different behavior mix:

```bash
DATASET=HuggingFaceTB/smoltalk DATASET_CONFIG=all python sft_data.py   # larger, broader
```

Sanity check the printed summary: the "supervised %" is the fraction of tokens
that carry loss (assistant content). For smol-smoltalk this is typically ~40-55%.

## 2. Run SFT

```bash
BASE_CHECKPOINT=minimoe_step_0019073.pt python sft_train.py
```

Defaults (override with env vars): 3 epochs, `MAX_LR=2e-5` with cosine decay,
`WARMUP_STEPS=50`, `BATCH_SIZE=16`, `GRAD_ACCUM_STEPS=4`, bf16 autocast +
`torch.compile` on CUDA. Loss on assistant tokens only.

Checkpoints land in `checkpoints/`: one per epoch
(`minimoe_sft_epoch{N}.pt`) plus a final `minimoe_sft.pt`. They use the same
format as the base checkpoint, so `sample.py` loads them directly.

Watch `sft_log.csv` (or stdout): training loss should fall steadily and the
periodic `val_loss` should track it. If val loss turns up while train loss keeps
falling, you are overfitting - prefer the earlier epoch checkpoint or drop to 2
epochs / lower LR.

## 3. Try it

Pull a checkpoint back down (or sample on the box) and chat with the SFT template:

```bash
python sample.py -c checkpoints/minimoe_sft.pt --sft
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

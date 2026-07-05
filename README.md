# miniMoE

A small GPT-style mixture-of-experts transformer trained from scratch on FineWeb-Edu.

The model is 6 transformer blocks with 768-dim embeddings, where each feed-forward layer is an 8-expert top-2 MoE.
That gives 280.4M total parameters, of which about 110.4M are active per token (roughly GPT-2 small compute).
Training uses the GPT-2 tokenizer, bf16 autocast, `torch.compile`, DDP, and a Switch-style load-balancing auxiliary loss (coefficient 0.01) with the router computed in fp32.

## Files

- `model.py` - model, MoE layer, and optimizer setup
- `train.py` - training loop (single GPU or DDP via torchrun)
- `fineweb.py` - downloads and tokenizes FineWeb-Edu sample-10BT into `.npy` shards
- `hellaswag.py` - HellaSwag download and eval helpers (also runs standalone against HF GPT-2 baselines)

## Running on a cluster

### 1. Setup

```bash
git clone <this repo> && cd miniMoE
pip install -r requirements.txt
```

Install a CUDA build of PyTorch if the cluster image does not already have one.

### 2. Prepare the data

```bash
python fineweb.py
```

This downloads FineWeb-Edu sample-10BT and writes ~100 shards (~20 GB) to `edu_fineweb10B/`.
The first 50M tokens become the val split and the next 50M the test split; the rest is train.
Run it once on shared storage before launching training.
HellaSwag val data downloads automatically at the start of training.

### 3. Train

8x GPU node (the intended setup):

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

Single GPU:

```bash
python train.py
```

Defaults: 19,073 steps x 524,288 tokens per step = 10.0B tokens, i.e. about one epoch of the data.
With `BATCH_SIZE=16` on 8 GPUs that is 4 gradient-accumulation micro-steps per optimizer step.
If you hit CUDA OOM, halve `BATCH_SIZE` (grad accum adjusts automatically, results are identical).
If you have headroom (e.g. 80GB cards), try raising `BATCH_SIZE` to 32 or 64 for throughput.

Recommended launch command for a multi-hour run (survives SSH disconnects):

```bash
nohup torchrun --standalone --nproc_per_node=8 train.py > train_stdout.log 2>&1 &
tail -f train_stdout.log
```

### 4. Resume after a crash

Checkpoints (model + optimizer state, ~3.4 GB each) are written to `checkpoints/` every 5,000 steps and at the end.
To resume:

```bash
RESUME_CHECKPOINT=checkpoints/minimoe_step_0005000.pt \
  torchrun --standalone --nproc_per_node=8 train.py
```

The data loader fast-forwards to the exact batch where the checkpoint was saved.

## Configuration

Everything is an environment variable.

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `edu_fineweb10B` | Directory containing the FineWeb shards |
| `TOTAL_BATCH_SIZE` | `524288` | Tokens per optimizer step (across all GPUs and grad accum) |
| `BATCH_SIZE` | `16` | Micro-batch sequences per GPU |
| `BLOCK_SIZE` | `1024` | Sequence length |
| `MAX_STEPS` | `19073` | Optimizer steps (19073 x 524288 = 10B tokens) |
| `MAX_LR` | `6e-4` | Peak learning rate |
| `MIN_LR` | `MAX_LR/10` | Final learning rate after cosine decay |
| `WARMUP_STEPS` | `715` | Linear LR warmup steps (375M tokens) |
| `WEIGHT_DECAY` | `0.1` | AdamW weight decay on >=2D tensors |
| `EVAL_INTERVAL` | `100` | Steps between val-loss evals (0 disables) |
| `EVAL_STEPS` | `10` | Val batches per eval |
| `HELLASWAG_INTERVAL` | `100` | Steps between HellaSwag evals (0 disables) |
| `HELLASWAG_EXAMPLES` | `32` | Examples per HellaSwag eval (-1 for all 10,042) |
| `CHECKPOINT_INTERVAL` | `5000` | Steps between checkpoints (0 disables) |
| `CHECKPOINT_DIR` | `checkpoints` | Checkpoint directory |
| `RESUME_CHECKPOINT` | (empty) | Path to a checkpoint to resume from |
| `LOG_INTERVAL` | `100` | Steps between console/CSV log rows |
| `LOG_FILE` | `train_log.csv` | Metrics CSV path |
| `TORCH_COMPILE` | `1` | Set 0 to disable `torch.compile` (useful for debugging) |
| `TEST_AT_END` | `1` | Run final test-split eval |

## Monitoring

Metrics stream to `train_log.csv` (train/val/test loss, HellaSwag acc, LR, grad norm, tokens/sec).
Watch for: val loss tracking train loss, grad norm staying mostly around or below 1.0 after warmup, and HellaSwag acc_norm drifting above the 25% random baseline as training progresses.
For reference, GPT-2 124M reaches about 29.5% HellaSwag acc_norm under this eval style.
If loss spikes and does not recover, resume from the last checkpoint with a lower `MAX_LR` (e.g. `4e-4`).

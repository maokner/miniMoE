# miniMoE GPU evaluation runbook

This runbook evaluates only the final 19,073-step base checkpoint and the final two-epoch SFT checkpoint.
It does not train models, recreate baselines, or invoke an external inference service.

## Prepare a clean GPU instance

Use a Linux host with an NVIDIA GPU, a working driver, Python 3, Git, and internet access.

```bash
git clone https://github.com/mokner123/miniMoE.git
cd miniMoE
bash scripts/prepare_gpu.sh
```

The preparation script creates a repo-local `.venv` (with access to any preinstalled CUDA PyTorch), installs the repository and evaluation dependencies into it, downloads both public checkpoints into `checkpoints/`, and verifies their pinned SHA-256 hashes.
It stops before evaluation if Linux, `nvidia-smi`, CUDA-enabled PyTorch, either download, or either hash is unavailable.
Rerunning it is safe: verified checkpoints are skipped, interrupted downloads resume, and a corrupt download is deleted and refetched.
`run_gpu_experiments.sh` uses the same `.venv` automatically; set `PYTHON=/path/to/python` to override.

## Smoke test

Run the complete single-GPU smoke path:

```bash
bash scripts/run_gpu_experiments.sh smoke
```

The smoke test loads both checkpoints, performs explicit forward passes, scores the first 32 HellaSwag validation examples for each model, captures routing telemetry, and greedily generates one response from each model.
The combined record is written to `results/smoke.json`.

Run the same smoke workflow on two GPUs:

```bash
NPROC_PER_NODE=2 bash scripts/run_gpu_experiments.sh smoke
```

The final merged predictions are sorted by original dataset index and should match the single-GPU predictions.
Minor floating-point score differences can occur across GPU architectures, but the expected predictions are deterministic for a fixed environment.

## Full experiment bundle

Run every portfolio experiment on one GPU:

```bash
bash scripts/run_gpu_experiments.sh full
```

Run it on multiple local GPUs by setting the process count:

```bash
NPROC_PER_NODE=4 bash scripts/run_gpu_experiments.sh full
```

The workflow runs, in order:

1. All 10,042 HellaSwag validation examples on the base checkpoint.
2. All 10,042 HellaSwag validation examples on the SFT checkpoint.
3. Prompt-suite and first-1,000-example routing analysis for the base checkpoint.
4. The same routing analysis for the SFT checkpoint.
5. Twelve base-checkpoint routing interventions on the first 1,000 HellaSwag examples.
6. The committed deterministic prompt suite on both checkpoints.
7. All final figures.

CUDA bf16 autocast, batch size 8 per GPU, seed 67, greedy prompt decoding, and disabled compilation are the defaults.
Pass command-specific flags directly to `src/experiments.py` when a different diagnostic setting is needed.

## Resume an interrupted run

Re-run the same command with the same output path and world size.
Each rank keeps completed example records under `results/.parts/`, skips their original indices on restart, and merges rank files without duplicates.
A record torn by a mid-write interruption is detected and re-evaluated on the next run.
Part files are keyed by world size, so changing the GPU count starts that command's shards fresh instead of resuming; keep the count fixed to reuse completed work.

For example:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 src/experiments.py hellaswag \
  --checkpoint checkpoints/minimoe_step_0019073.pt \
  --output results/hellaswag_base.json
```

If the full shell workflow is restarted, already completed evaluation commands quickly validate and merge their parts before proceeding.
Temporary rank parts and atomic-write state are ignored by Git.

## Expected outputs

```text
results/
  hellaswag_base.json
  hellaswag_sft.json
  routing_base.json
  routing_sft.json
  ablations.json
  prompt_outputs.json
  figures/
    hellaswag_base_vs_sft.png
    expert_utilization.png
    router_entropy_load_balance.png
    token_routing.png
    routing_ablations.png
```

Completed JSON files and figures are intentionally trackable.
Every top-level experiment result records its schema version, exact command, Git commit, environment, package versions, GPU data, checkpoint hash and metadata, dataset URL and hash when applicable, settings, metrics, and detailed observations.

Routing comparisons are descriptive.
Differences between base and SFT routes do not by themselves demonstrate semantic expert specialization.

## Copy results to a local machine

From the local machine, copy the complete trackable result directory with `rsync`:

```bash
rsync -av user@gpu-host:/path/to/miniMoE/results/ ./results/
```

Exclude resumable state if it is no longer needed:

```bash
rsync -av --exclude='.parts/' user@gpu-host:/path/to/miniMoE/results/ ./results/
```

Review the JSON and figures before committing them.
Do not update README benchmark claims until the real GPU artifacts have been produced and checked.

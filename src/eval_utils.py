"""Common experiment metadata, distributed sharding, and result utilities."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import shlex
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist


SCHEMA_VERSION = "minimoe-eval-v1"


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: str
    initialized_here: bool = False

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def seed_everything(seed: int, rank: int = 0) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def distributed_context(device_request: str = "auto") -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False

    if device_request == "auto":
        if torch.cuda.is_available():
            device = f"cuda:{local_rank}"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device_request == "cuda":
        device = f"cuda:{local_rank}"
    else:
        device = device_request

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.startswith("cuda") else "gloo"
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        initialized_here = True
    return DistributedContext(rank, world_size, local_rank, device, initialized_here)


def close_distributed(context: DistributedContext) -> None:
    if context.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def autocast_context(device: str, enabled: bool = True):
    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    names = ["torch", "tiktoken", "numpy", "requests", "matplotlib", "pytest", "ruff"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def environment_metadata(context: DistributedContext) -> dict[str, Any]:
    gpu = None
    if context.device.startswith("cuda"):
        index = torch.device(context.device).index or 0
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu,
        "world_size": context.world_size,
    }


def result_metadata(
    command: str,
    context: DistributedContext,
    settings: dict[str, Any],
    checkpoints: list[dict[str, Any]] | None = None,
    dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "argv": shlex.join(sys.argv),
        "git_commit": git_commit(),
        "environment": environment_metadata(context),
        "checkpoints": checkpoints or [],
        "dataset": dataset,
        "settings": settings,
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as handle:
        lines = handle.readlines()
    if lines and not lines[-1].endswith("\n"):
        # A writer killed mid-append leaves a torn final line (possibly parseable
        # JSON with only the newline missing). Truncate it so the owning rank
        # redoes that example and the next append cannot concatenate onto it.
        path.write_text("".join(lines[:-1]))
        lines = lines[:-1]
    records = []
    for line_number, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSONL in {path} at line {line_number}"
            ) from error
    return records


def completed_indices(path: str | Path, key: str = "index") -> set[int]:
    return {int(record[key]) for record in read_jsonl(path)}


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def merge_rank_parts(
    part_paths: Iterable[str | Path],
    key: str = "index",
) -> list[dict[str, Any]]:
    """Merge parts deterministically and reject inconsistent duplicates."""
    by_index: dict[int, dict[str, Any]] = {}
    for path in sorted(map(Path, part_paths), key=lambda item: str(item)):
        for record in read_jsonl(path):
            index = int(record[key])
            if index in by_index and record != by_index[index]:
                raise ValueError(f"Conflicting duplicate {key}={index} in {path}")
            by_index[index] = record
    return [by_index[index] for index in sorted(by_index)]


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def part_path(
    output: str | Path, rank: int, world_size: int, tag: str | None = None
) -> Path:
    # World size is part of the name: shards from a run with a different GPU
    # count cover different index sets, so mixing them at merge time would
    # produce conflicting duplicates instead of a clean resume.
    output = Path(output)
    suffix = f".{tag}" if tag else ""
    return (
        output.parent
        / ".parts"
        / f"{output.stem}{suffix}.ws{world_size:02d}.rank{rank:04d}.jsonl"
    )

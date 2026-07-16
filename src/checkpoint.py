"""Shared loading and fingerprinting for miniMoE ``.pt`` checkpoints."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from model import Model, ModelConfig


@lru_cache(maxsize=None)
def _sha256_file(path: str, size: int, mtime_ns: int, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _sha256_file(str(resolved), stat.st_size, stat.st_mtime_ns, chunk_size)


def read_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_metadata(
    path: str | Path, checkpoint: dict[str, Any] | None = None
) -> dict[str, Any]:
    checkpoint = checkpoint if checkpoint is not None else read_checkpoint(path)
    return {
        "path": str(Path(path)),
        "filename": Path(path).name,
        "sha256": sha256_file(path),
        "step": checkpoint.get("step"),
        "tokens_seen": checkpoint.get("tokens_seen"),
        "model_config": dict(checkpoint["model_config"]),
    }


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[Model, ModelConfig, dict[str, Any]]:
    checkpoint = read_checkpoint(path, map_location="cpu")
    config = ModelConfig(**checkpoint["model_config"])
    model = Model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    metadata = checkpoint_metadata(path, checkpoint)
    metadata["model_config"] = asdict(config)
    return model, config, metadata

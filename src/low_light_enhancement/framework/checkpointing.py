from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def build_checkpoint_path(experiment_name: str, run_index: int) -> Path:
    return Path("checkpoints") / experiment_name / f"{run_index}.pt"


def save_checkpoint(checkpoint_path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    *,  # next options are keyword-only
    map_location: torch.device
) -> dict[str, Any]:
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=map_location
        )
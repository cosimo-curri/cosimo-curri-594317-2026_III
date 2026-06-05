from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.low_light_enhancement.framework.augmentation import (
    apply_train_augmentation,
    is_augmentation_enabled
)
from src.low_light_enhancement.framework.io import (
    read_csv_rows,
    read_rgb_image
)
from src.low_light_enhancement.framework.transforms import image_to_tensor


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        *,  # next options are keyword-only
        require_target: bool = False,
        augmentation_config: dict[str, Any] | None = None
    ) -> None:
        rows, _ = read_csv_rows(manifest_path)

        self.manifest_path = manifest_path
        self.split = split

        self.rows = [
            row
            for row in rows
            if row["split"] == split
        ]

        self.require_target = require_target
        self.augmentation_config = augmentation_config

        if split != "train" and is_augmentation_enabled(augmentation_config):
            raise ValueError(
                "Data augmentation can only be enabled for the training split. "
                f"Got split={split!r}."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]

        input_image = read_rgb_image(Path(row["input_path"]))
        target_image: Image.Image | None = None
        target_path = row.get("target_path", "")

        if target_path:
            target_image = read_rgb_image(Path(target_path))
        elif self.require_target:
            raise ValueError(
                f"Missing target_path in {self.manifest_path} at row {index}."
            )

        input_image, target_image = apply_train_augmentation(
            input_image,
            target_image,
            self.augmentation_config
        )

        target_tensor: Tensor | None = None
        input_tensor = image_to_tensor(input_image)

        if target_image is not None:
            target_tensor = image_to_tensor(target_image)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "input_path": row["input_path"],
            "target_path": target_path,
            "metadata": {
                "dataset": row.get("dataset", ""),
                "category": row.get("category", ""),
                "illumination_level": row.get("illumination_level", "")
            }
        }

    def dataset_name(self) -> str:
        if not self.rows:
            return self.manifest_path.stem

        return self.rows[0].get("dataset", self.manifest_path.stem)


def summarize_manifest(manifest_path: Path) -> dict[str, Any]:
    rows, fieldnames = read_csv_rows(manifest_path)

    split_counts: Counter[str] = Counter()
    target_path_nonempty_counts: Counter[str] = Counter()
    datasets: set[str] = set()

    for row in rows:
        split = row.get("split", "")
        dataset = row.get("dataset", "")
        target_path = row.get("target_path", "")

        split_counts[split] += 1

        if target_path:
            target_path_nonempty_counts[split] += 1

        if dataset:
            datasets.add(dataset)

    return {
        "manifest_path": manifest_path.as_posix(),
        "num_rows": len(rows),
        "columns": fieldnames,
        "datasets": sorted(datasets),
        "split_counts": dict(sorted(split_counts.items())),
        "target_path_nonempty_counts": dict(
            sorted(target_path_nonempty_counts.items())
        )
    }


def collate_manifest_samples(
    samples: list[dict[str, Any]]
) -> dict[str, Any]:
    inputs = torch.stack([
        sample["input"]
        for sample in samples
    ])

    targets = [
        sample["target"]
        for sample in samples
    ]

    has_target = [
        target is not None
        for target in targets
    ]

    if all(has_target):
        target_batch = torch.stack([
            target
            for target in targets
            if target is not None
        ])
    else:
        target_batch = None

    metadata: dict[str, list[str]] = {}

    for key in samples[0]["metadata"]:
        metadata[key] = [
            sample["metadata"].get(key, "")
            for sample in samples
        ]

    return {
        "input": inputs,
        "target": target_batch,
        "has_target": has_target,
        "input_path": [sample["input_path"] for sample in samples],
        "target_path": [sample["target_path"] for sample in samples],
        "metadata": metadata
    }


def build_dataloader(
    manifest_path: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    require_target: bool = False,
    pin_memory: bool | None = None,
    augmentation_config: dict[str, Any] | None = None
) -> DataLoader[dict[str, Any]]:
    dataset = ManifestDataset(
        manifest_path=manifest_path,
        split=split,
        require_target=require_target,
        augmentation_config=augmentation_config
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_manifest_samples,
        worker_init_fn=seed_worker
    )


def seed_worker(worker_id: int) -> None:
    del worker_id

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
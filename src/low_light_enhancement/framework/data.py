from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.low_light_enhancement.framework.io import read_csv_rows
from src.low_light_enhancement.framework.transforms import image_to_tensor


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        *,  # next options are keyword-only
        require_target: bool = False
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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]

        input_tensor = read_image_tensor(Path(row["input_path"]))
        target_tensor: Tensor | None = None
        target_path = row.get("target_path", "")

        if target_path:
            target_tensor = read_image_tensor(Path(target_path))
        elif self.require_target:
            raise ValueError(
                f"Missing target_path in {self.manifest_path} at row {index}."
            )

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


def read_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def read_image_tensor(image_path: Path) -> Tensor:
    return image_to_tensor(read_rgb_image(image_path))


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
    require_target: bool = False
) -> DataLoader[dict[str, Any]]:
    dataset = ManifestDataset(
        manifest_path=manifest_path,
        split=split,
        require_target=require_target
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_manifest_samples
    )
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

# Imports used only for type annotations -> (not at runtime)
if TYPE_CHECKING:
    from PIL import Image
    from torch import Tensor


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(json_path: Path) -> Any:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()


def read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    return rows, fieldnames


def write_csv_rows(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rgb_image(image_path: Path) -> Image.Image:
    from PIL import Image, ImageOps

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def read_image_tensor(image_path: Path) -> Tensor:
    from src.low_light_enhancement.framework.transforms import image_to_tensor
    return image_to_tensor(read_rgb_image(image_path))
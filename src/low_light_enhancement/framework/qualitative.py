from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from src.low_light_enhancement.framework.checkpointing import load_checkpoint
from src.low_light_enhancement.framework.io import (
    read_csv_rows,
    read_image_tensor,
    read_json
)
from src.low_light_enhancement.framework.registry import build_model_wrapper
from src.low_light_enhancement.framework.torch_utils import get_device
from src.low_light_enhancement.framework.transforms import tensor_to_image


class QualitativeExporter:
    def __init__(
        self,
        checkpoint_path: Path,
        output_dir: Path
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = get_device()

        self.checkpoint = load_checkpoint(
            checkpoint_path,
            map_location=self.device
        )

        self.config = self.checkpoint["resolved_config"]
        self.wrapper = build_model_wrapper(self.config["model"]["name"])

        self.model = self.wrapper.build_model(
            self.config["model"]
        ).to(self.device)

        self.model.load_state_dict(self.checkpoint["best_model_state_dict"])
        self.model.eval()

        self.manifest_rows = build_manifest_row_index(self.config)

    def export(self, candidates_path: Path) -> None:
        candidates = load_candidates(candidates_path)

        for index, candidate in enumerate(candidates, start=1):
            self.export_candidate(index, candidate)

    def export_candidate(
        self,
        index: int,
        candidate: dict[str, Any]
    ) -> None:
        input_path = candidate["input_path"]
        failure_type = candidate.get("failure_type", "selected")
        row = self.manifest_rows[input_path]

        input_tensor = read_image_tensor(Path(input_path))

        with torch.no_grad():
            batch = {
                "input": input_tensor.unsqueeze(0).to(self.device)
            }

            output = self.wrapper.forward(self.model, batch)
            prediction = self.wrapper.get_prediction(output)[0]

        output_path = self.output_dir / failure_type
        output_path.mkdir(parents=True, exist_ok=True)

        prefix = build_output_prefix(index, row)

        panel_images = [
            tensor_to_image(input_tensor),
            tensor_to_image(prediction)
        ]

        panel_labels = ["Input", "Output"]

        target_path = row.get("target_path")

        if target_path:
            target_tensor = read_image_tensor(Path(target_path))
            panel_images.append(tensor_to_image(target_tensor))
            panel_labels.append("Target")

        panel = make_panel(
            images=panel_images,
            labels=panel_labels
        )

        panel.save(output_path / f"{prefix}_panel.png")


def load_candidates(candidates_path: Path) -> list[dict[str, Any]]:
    return read_json(candidates_path)


def build_manifest_row_index(
    config: dict[str, Any]
) -> dict[str, dict[str, str]]:
    row_index: dict[str, dict[str, str]] = {}
    manifest_paths = [Path(config["data"]["train_manifest"])]

    for test_config in config["data"].get("test_manifests", []):
        manifest_paths.append(Path(test_config["manifest"]))

    for manifest_path in manifest_paths:
        rows, _ = read_csv_rows(manifest_path)

        for row in rows:
            row_index[row["input_path"]] = row

    return row_index


def build_output_prefix(index: int, row: dict[str, str]) -> str:
    dataset = row.get("dataset", "dataset")
    return f"{index:04d}_{dataset}"


def make_panel(
    images: list[Image.Image],
    labels: list[str] | None = None
) -> Image.Image:
    widths = [image.width for image in images]
    heights = [image.height for image in images]

    label_height = 32 if labels is not None else 0

    panel = Image.new(
        "RGB",
        (sum(widths), max(heights) + label_height),
        "white"
    )

    current_x = 0

    if labels is not None:
        draw = ImageDraw.Draw(panel)
        font = ImageFont.load_default()

    for index, image in enumerate(images):
        if labels is not None:
            draw_centered_label(
                draw=draw,
                label=labels[index],
                x=current_x,
                width=image.width,
                height=label_height,
                font=font
            )

        panel.paste(image, (current_x, label_height))
        current_x += image.width

    return panel


def draw_centered_label(
    *,  # next options are keyword-only
    draw: ImageDraw.ImageDraw,
    label: str,
    x: int,
    width: int,
    height: int,
    font: ImageFont.ImageFont
) -> None:
    bbox = draw.textbbox((0, 0), label, font=font)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = x + (width - text_width) // 2
    text_y = (height - text_height) // 2

    draw.text((text_x, text_y), label, fill="black", font=font)
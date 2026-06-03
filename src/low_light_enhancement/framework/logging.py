from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_PRINT_ORDER = (
    "loss",
    "train_loss",
    "ssim",
    "psnr",
    "mae",
    "niqe",
    "brisque",
    "mean_luminance",
    "dark_pixel_ratio",
    "bright_clipping_ratio",
    "luminance_std",
    "rgb_channel_imbalance",
    "laplacian_sharpness"
)


def format_metrics(metrics: dict[str, float]) -> str:
    parts = []

    for metric_name in METRIC_PRINT_ORDER:
        if metric_name in metrics:
            parts.append(f"{metric_name}={metrics[metric_name]:.6f}")

    for metric_name, value in metrics.items():
        if metric_name not in METRIC_PRINT_ORDER:
            parts.append(f"{metric_name}={value:.6f}")

    return " | ".join(parts)


class JsonlLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.log_path.open("w", encoding="utf-8")

    def close(self) -> None:
        self.file.close()

    def log(self, event: str, payload: dict[str, Any] | None = None) -> None:
        if payload is None:
            payload = {}

        row = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        row.update(payload)

        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()


class TextLogger:
    def info(self, message: str) -> None:
        print(message, flush=True)
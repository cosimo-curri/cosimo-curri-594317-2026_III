from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pyiqa
import torch
from torch import Tensor

from src.low_light_enhancement.framework.image_ops import (
    compute_laplacian,
    compute_luminance,
    compute_ssim,
)


class MetricComputer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.niqe_metric: Any | None = None
        self.brisque_metric: Any | None = None

    def compute_samples(
        self,
        prediction: Tensor,
        target: Tensor | None,
        *,  # next options are keyword-only
        include_no_reference: bool,
        include_diagnostics: bool
    ) -> list[dict[str, float]]:
        prediction = prediction.detach().float().clamp(0.0, 1.0)
        target = None if target is None else target.detach().float().clamp(0.0, 1.0)

        sample_metrics = [
            {}
            for _ in range(prediction.shape[0])
        ]

        if include_diagnostics:
            sample_metrics = [
                compute_diagnostic_metrics(prediction[index:index + 1])
                for index in range(prediction.shape[0])
            ]

        if include_no_reference:
            niqe_values = self.compute_no_reference_metric("niqe", prediction)
            brisque_values = self.compute_no_reference_metric("brisque", prediction)

            for index, metrics in enumerate(sample_metrics):
                metrics["niqe"] = niqe_values[index]
                metrics["brisque"] = brisque_values[index]

        if target is not None:
            ssim_values = compute_ssim_per_sample(prediction, target)
            psnr_values = compute_psnr_per_sample(prediction, target)
            mae_values = compute_mae_per_sample(prediction, target)

            for index, metrics in enumerate(sample_metrics):
                metrics["ssim"] = ssim_values[index]
                metrics["psnr"] = psnr_values[index]
                metrics["mae"] = mae_values[index]

        return sample_metrics

    def compute_no_reference_metric(
        self,
        metric_name: str,
        prediction: Tensor
    ) -> list[float]:
        metric = self.get_pyiqa_metric(metric_name)

        with torch.no_grad():
            values = metric(prediction.to(self.device))
            values = values.detach().float().cpu().reshape(-1)

        if values.numel() != prediction.shape[0]:
            raise ValueError(
                f"{metric_name} returned {values.numel()} values for "
                f"{prediction.shape[0]} samples."
            )

        return [float(value) for value in values.tolist()]

    def get_pyiqa_metric(self, metric_name: str) -> Any:
        if metric_name == "niqe":
            if self.niqe_metric is None:
                self.niqe_metric = pyiqa.create_metric(
                    "niqe",
                    device=self.device,
                    as_loss=False
                )

            return self.niqe_metric

        if metric_name == "brisque":
            if self.brisque_metric is None:
                self.brisque_metric = pyiqa.create_metric(
                    "brisque",
                    device=self.device,
                    as_loss=False
                )

            return self.brisque_metric

        raise ValueError(f"Unsupported no-reference metric: {metric_name!r}.")


def compute_diagnostic_metrics(prediction: Tensor) -> dict[str, float]:
    luminance = compute_luminance(prediction)
    channel_means = prediction.mean(dim=(0, 2, 3))
    laplacian = compute_laplacian(luminance)
    bright_clipping = prediction.amax(dim=1, keepdim=True) > 0.98

    return {
        "mean_luminance": float(luminance.mean().item()),
        "dark_pixel_ratio": float((luminance < 0.10).float().mean().item()),
        "bright_clipping_ratio": float(
            bright_clipping.float().mean().item()
        ),
        "luminance_std": float(luminance.std(unbiased=False).item()),
        "rgb_channel_imbalance": float(
            (channel_means.max() - channel_means.min()).item()
        ),
        "laplacian_sharpness": float(laplacian.var(unbiased=False).item())
    }


def compute_mae_per_sample(prediction: Tensor, target: Tensor) -> list[float]:
    mae = (prediction - target).abs().mean(dim=(1, 2, 3))
    return [float(value) for value in mae.cpu().tolist()]


def compute_psnr_per_sample(prediction: Tensor, target: Tensor) -> list[float]:
    mse = (prediction - target).pow(2).mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1.0e-10))

    return [float(value) for value in psnr.cpu().tolist()]


def compute_ssim_per_sample(
    prediction: Tensor,
    target: Tensor,
    window_size: int = 11
) -> list[float]:
    values = []

    for index in range(prediction.shape[0]):
        values.append(
            float(
                compute_ssim(
                    prediction[index:index + 1],
                    target[index:index + 1],
                    window_size=window_size
                ).item()
            )
        )

    return values


def aggregate_metrics(
    sample_metrics: Iterable[dict[str, float]]
) -> tuple[dict[str, float], dict[str, float]]:
    values_by_metric: dict[str, list[float]] = {}

    for metrics in sample_metrics:
        for metric_name, value in metrics.items():
            values_by_metric.setdefault(metric_name, []).append(value)

    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for metric_name, values in values_by_metric.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        means[metric_name] = float(tensor.mean().item())
        stds[metric_name] = float(tensor.std(unbiased=False).item())

    return means, stds
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.low_light_enhancement.framework.config_validation import (
    require_non_negative_int,
    require_positive_float,
    require_positive_int
)
from src.low_light_enhancement.framework.image_ops import compute_luminance
from src.low_light_enhancement.framework.losses import build_loss
from src.low_light_enhancement.models.unet import build_unet


class IlluminationGuidedUNetWrapper:
    def __init__(self) -> None:
        self.illumination_method = "luminance"
        self.illumination_smoothing_kernel_size = 0
        self.illumination_smoothing_sigma = 1.5

    def build_model(self, model_config: dict[str, Any]) -> nn.Module:
        config = dict(model_config)
        config.pop("name", None)

        self.illumination_method = config.pop(
            "illumination_method",
            "luminance"
        )

        self.illumination_smoothing_kernel_size = require_non_negative_int(
            config.pop("illumination_smoothing_kernel_size", 0),
            "model.illumination_smoothing_kernel_size"
        )

        self.illumination_smoothing_sigma = float(
            config.pop("illumination_smoothing_sigma", 1.5)
        )

        self.validate_illumination_config()

        # RGB + illumination map
        config["in_channels"] = 4

        return build_unet(config)

    def build_loss(self, loss_config: dict[str, Any]) -> nn.Module:
        return build_loss(loss_config)

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, Any]
    ) -> Tensor:
        input_rgb = batch["input"]
        illumination_map = self.compute_illumination_map(input_rgb)
        guided_input = torch.cat([input_rgb, illumination_map], dim=1)

        return model(guided_input)

    def get_prediction(self, output: Tensor) -> Tensor:
        return output

    def compute_illumination_map(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "IlluminationGuidedUNetWrapper expects input tensors with "
                f"shape [B, 3, H, W]. Got {tuple(image.shape)}."  # 4th here
            )

        if self.illumination_method == "luminance":
            illumination = compute_luminance(image)
        elif self.illumination_method == "max_rgb":
            illumination = image.amax(dim=1, keepdim=True)
        else:
            raise RuntimeError(
                "Invalid illumination_method state: "
                f"{self.illumination_method!r}."
            )

        if self.illumination_smoothing_kernel_size > 1:
            illumination = gaussian_blur_2d(
                illumination,
                kernel_size=self.illumination_smoothing_kernel_size,
                sigma=self.illumination_smoothing_sigma
            )

        return illumination.clamp(0.0, 1.0)

    def validate_illumination_config(self) -> None:
        if self.illumination_method not in {"luminance", "max_rgb"}:
            raise ValueError(
                "model.illumination_method must be one of "
                "{\"luminance\", \"max_rgb\"}. "
                f"Got {self.illumination_method!r}."
            )

        if self.illumination_smoothing_kernel_size == 0:
            return

        require_positive_int(
            self.illumination_smoothing_kernel_size,
            "model.illumination_smoothing_kernel_size"
        )

        if self.illumination_smoothing_kernel_size % 2 == 0:
            raise ValueError(
                "model.illumination_smoothing_kernel_size must be odd when "
                "Gaussian smoothing is enabled. "
                f"Got {self.illumination_smoothing_kernel_size}."
            )

        self.illumination_smoothing_sigma = require_positive_float(
            self.illumination_smoothing_sigma,
            "model.illumination_smoothing_sigma"
        )


def gaussian_blur_2d(
    image: Tensor,
    *,  # next options are keyword-only
    kernel_size: int,
    sigma: float
) -> Tensor:
    coords = torch.arange(
        kernel_size,
        device=image.device,
        dtype=image.dtype
    ) - (kernel_size - 1) / 2.0

    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    padding = kernel_size // 2

    # Reflection padding avoids artificial dark borders in the illumination map
    padded = F.pad(
        image,
        (padding, padding, padding, padding),
        mode="reflect"
    )

    return F.conv2d(padded, kernel_2d)
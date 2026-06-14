from __future__ import annotations

from typing import Any

from torch import Tensor, nn
import torch.nn.functional as F

from src.low_light_enhancement.framework.config_validation import (
    require_non_negative_float,
    require_positive_float,
    require_positive_int
)
from src.low_light_enhancement.framework.image_ops import (
    compute_ssim,
    gaussian_blur_2d
)
from src.low_light_enhancement.framework.photometric import (
    build_photometric_perturbation_config
)


_WEIGHT_SUM_TOLERANCE = 1e-6


class L1SSIMLoss(nn.Module):
    def __init__(
        self,
        *,  # next options are keyword-only
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        window_size: int = 11
    ) -> None:
        super().__init__()

        l1_weight = require_non_negative_float(
            l1_weight,
            "loss.l1_weight"
        )

        ssim_weight = require_non_negative_float(
            ssim_weight,
            "loss.ssim_weight"
        )

        window_size = require_positive_int(
            window_size,
            "loss.window_size"
        )

        total_weight = l1_weight + ssim_weight

        if abs(total_weight - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                "For L1SSIMLoss, l1_weight + ssim_weight must be 1. "
                f"Got l1_weight={l1_weight}, ssim_weight={ssim_weight}, "
                f"sum={total_weight}."
            )

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.window_size = window_size

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        l1_loss = F.l1_loss(prediction, target)

        # SSIM is also used as a part of training objective
        ssim_loss = 1.0 - compute_ssim(
            prediction=prediction,
            target=target,
            window_size=self.window_size
        )

        return self.l1_weight * l1_loss + self.ssim_weight * ssim_loss


class FACECLLoss(nn.Module):
    def __init__(
        self,
        *,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        low_frequency_weight: float = 0.1,
        high_frequency_weight: float = 0.1,
        consistency_weight: float = 0.05,
        frequency_kernel_size: int = 15,
        frequency_sigma: float = 3.0,
        window_size: int = 11,
        consistency_config: dict[str, Any] | None = None
    ) -> None:
        super().__init__()

        self.reconstruction_loss = L1SSIMLoss(
            l1_weight=l1_weight,
            ssim_weight=ssim_weight,
            window_size=window_size
        )

        self.low_frequency_weight = require_non_negative_float(
            low_frequency_weight,
            "loss.low_frequency_weight"
        )

        self.high_frequency_weight = require_non_negative_float(
            high_frequency_weight,
            "loss.high_frequency_weight"
        )

        self.consistency_weight = require_non_negative_float(
            consistency_weight,
            "loss.consistency_weight"
        )

        self.frequency_kernel_size = require_positive_int(
            frequency_kernel_size,
            "loss.frequency_kernel_size"
        )

        if self.frequency_kernel_size % 2 == 0:
            raise ValueError(
                "loss.frequency_kernel_size must be odd. "
                f"Got {self.frequency_kernel_size}."
            )

        self.frequency_sigma = require_positive_float(
            frequency_sigma,
            "loss.frequency_sigma"
        )

        self.consistency_config = build_photometric_perturbation_config(
            consistency_config
        )

    def forward(
        self,
        output: Tensor | dict[str, Tensor],
        target: Tensor
    ) -> Tensor:
        prediction = extract_prediction(output)
        loss = self.reconstruction_loss(prediction, target)

        if self.low_frequency_weight > 0.0:
            loss = loss + self.low_frequency_weight * self.compute_low_frequency_loss(
                prediction,
                target
            )

        if self.high_frequency_weight > 0.0:
            loss = loss + self.high_frequency_weight * self.compute_high_frequency_loss(
                prediction,
                target
            )

        consistency_prediction = extract_consistency_prediction(output)

        if consistency_prediction is not None and self.consistency_weight > 0.0:
            loss = loss + self.consistency_weight * F.l1_loss(
                consistency_prediction,
                prediction.detach()
            )

        return loss

    def compute_low_frequency_loss(
        self,
        prediction: Tensor,
        target: Tensor
    ) -> Tensor:
        prediction_low = self.compute_low_frequency(prediction)
        target_low = self.compute_low_frequency(target)

        return F.l1_loss(prediction_low, target_low)

    def compute_high_frequency_loss(
        self,
        prediction: Tensor,
        target: Tensor
    ) -> Tensor:
        prediction_low = self.compute_low_frequency(prediction)
        target_low = self.compute_low_frequency(target)

        prediction_high = prediction - prediction_low
        target_high = target - target_low

        return F.l1_loss(prediction_high, target_high)

    def compute_low_frequency(self, image: Tensor) -> Tensor:
        return gaussian_blur_2d(
            image,
            kernel_size=self.frequency_kernel_size,
            sigma=self.frequency_sigma
        )


class ECLFARLoss(nn.Module):
    def __init__(
        self,
        *,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        initial_weight: float = 0.2,
        consistency_weight: float = 0.02,
        window_size: int = 11
    ) -> None:
        super().__init__()

        self.reconstruction_loss = L1SSIMLoss(
            l1_weight=l1_weight,
            ssim_weight=ssim_weight,
            window_size=window_size
        )

        self.initial_weight = require_non_negative_float(
            initial_weight,
            "loss.initial_weight"
        )

        self.consistency_weight = require_non_negative_float(
            consistency_weight,
            "loss.consistency_weight"
        )

    def forward(
        self,
        output: Tensor | dict[str, Tensor],
        target: Tensor
    ) -> Tensor:
        prediction = extract_prediction(output)
        loss = self.reconstruction_loss(prediction, target)

        initial_prediction = extract_initial_prediction(output)

        if initial_prediction is not None and self.initial_weight > 0.0:
            loss = loss + self.initial_weight * self.reconstruction_loss(
                initial_prediction,
                target
            )

        consistency_prediction = extract_ecl_far_consistency_prediction(output)

        if (
            initial_prediction is not None
            and consistency_prediction is not None
            and self.consistency_weight > 0.0
        ):
            loss = loss + self.consistency_weight * F.l1_loss(
                consistency_prediction,
                initial_prediction.detach()
            )

        return loss


def extract_prediction(output: Tensor | dict[str, Tensor]) -> Tensor:
    if isinstance(output, Tensor):
        return output

    return output["prediction"]


def extract_consistency_prediction(
    output: Tensor | dict[str, Tensor]
) -> Tensor | None:
    if isinstance(output, Tensor):
        return None

    return output.get("consistency_prediction")


def extract_initial_prediction(
    output: Tensor | dict[str, Tensor]
) -> Tensor | None:
    if isinstance(output, Tensor):
        return None

    return output.get("initial_prediction")


def extract_ecl_far_consistency_prediction(
    output: Tensor | dict[str, Tensor]
) -> Tensor | None:
    if isinstance(output, Tensor):
        return None

    return output.get("perturbed_prediction")


def build_loss(config: dict[str, Any]) -> nn.Module:
    loss_name = config["name"]

    if loss_name == "l1_ssim":
        return L1SSIMLoss(
            l1_weight=config["l1_weight"],
            ssim_weight=config["ssim_weight"]
        )

    if loss_name == "fac_ecl":
        return FACECLLoss(
            l1_weight=config["l1_weight"],
            ssim_weight=config["ssim_weight"],
            low_frequency_weight=config["low_frequency_weight"],
            high_frequency_weight=config["high_frequency_weight"],
            consistency_weight=config["consistency_weight"],
            frequency_kernel_size=config["frequency_kernel_size"],
            frequency_sigma=config["frequency_sigma"],
            consistency_config=config.get("consistency")
        )

    if loss_name == "ecl_far":
        return ECLFARLoss(
            l1_weight=config["l1_weight"],
            ssim_weight=config["ssim_weight"],
            initial_weight=config["initial_weight"],
            consistency_weight=config["consistency_weight"]
        )

    raise ValueError(f"Unsupported loss: {loss_name!r}.")
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.low_light_enhancement.framework.config_validation import (
    require_non_negative_float,
    require_non_negative_range,
    require_positive_range
)


DEFAULT_PHOTOMETRIC_CONFIG = {
    "exposure_min": 0.85,
    "exposure_max": 1.15,
    "gamma_min": 0.85,
    "gamma_max": 1.15,
    "color_temperature_delta": 0.05,
    "noise_sigma_min": 0.0,
    "noise_sigma_max": 0.015
}


def build_photometric_perturbation_config(
    config: dict[str, Any] | None
) -> dict[str, float]:
    if config is None:
        config = {}

    merged = dict(DEFAULT_PHOTOMETRIC_CONFIG)
    merged.update(config)

    exposure_min, exposure_max = require_positive_range(
        merged["exposure_min"],
        merged["exposure_max"],
        "loss.consistency.exposure"
    )

    gamma_min, gamma_max = require_positive_range(
        merged["gamma_min"],
        merged["gamma_max"],
        "loss.consistency.gamma"
    )

    noise_sigma_min, noise_sigma_max = require_non_negative_range(
        merged["noise_sigma_min"],
        merged["noise_sigma_max"],
        "loss.consistency.noise_sigma"
    )

    color_temperature_delta = require_non_negative_float(
        merged["color_temperature_delta"],
        "loss.consistency.color_temperature_delta"
    )

    return {
        "exposure_min": exposure_min,
        "exposure_max": exposure_max,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "color_temperature_delta": color_temperature_delta,
        "noise_sigma_min": noise_sigma_min,
        "noise_sigma_max": noise_sigma_max
    }


def apply_photometric_perturbation(
    image: Tensor,
    config: dict[str, float]
) -> Tensor:
    perturbed = apply_exposure_scaling(
        image,
        min_value=config["exposure_min"],
        max_value=config["exposure_max"]
    )

    perturbed = apply_gamma_variation(
        perturbed,
        min_value=config["gamma_min"],
        max_value=config["gamma_max"]
    )

    perturbed = apply_color_temperature_shift(
        perturbed,
        max_delta=config["color_temperature_delta"]
    )

    perturbed = apply_low_light_noise(
        perturbed,
        min_sigma=config["noise_sigma_min"],
        max_sigma=config["noise_sigma_max"]
    )

    return perturbed.clamp(0.0, 1.0)


def apply_exposure_scaling(
    image: Tensor,
    *,  # next options are keyword-only
    min_value: float,
    max_value: float
) -> Tensor:
    factor = sample_uniform_like_batch(image, min_value, max_value)
    return image * factor


def apply_gamma_variation(
    image: Tensor,
    *,
    min_value: float,
    max_value: float
) -> Tensor:
    gamma = sample_uniform_like_batch(image, min_value, max_value)
    return image.clamp_min(1e-6).pow(gamma)


def apply_color_temperature_shift(
    image: Tensor,
    *,
    max_delta: float
) -> Tensor:
    if max_delta == 0.0:
        return image

    delta = sample_uniform_like_batch(image, -max_delta, max_delta)

    channel_scale = torch.cat(
        [
            1.0 + delta,
            torch.ones_like(delta),
            1.0 - delta
        ],
        dim=1
    )

    return image * channel_scale.clamp_min(0.0)


def apply_low_light_noise(
    image: Tensor,
    *,
    min_sigma: float,
    max_sigma: float
) -> Tensor:
    if max_sigma == 0.0:
        return image

    sigma = sample_uniform_like_batch(image, min_sigma, max_sigma)
    return image + torch.randn_like(image) * sigma


def sample_uniform_like_batch(
    image: Tensor,
    min_value: float,
    max_value: float
) -> Tensor:
    shape = (image.shape[0], 1, 1, 1)

    return torch.empty(
        shape,
        device=image.device,
        dtype=image.dtype
    ).uniform_(min_value, max_value)
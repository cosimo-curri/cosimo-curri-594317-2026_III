from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def compute_luminance(image: Tensor) -> Tensor:
    red = image[:, 0:1]
    green = image[:, 1:2]
    blue = image[:, 2:3]

    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def compute_laplacian(luminance: Tensor) -> Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=luminance.device,
        dtype=luminance.dtype
    ).view(1, 1, 3, 3)

    return F.conv2d(luminance, kernel, padding=1)


def compute_ssim(
    prediction: Tensor,
    target: Tensor,
    *,  # next options are keyword-only
    window_size: int = 11
) -> Tensor:
    channel_count = prediction.shape[1]
    padding = window_size // 2

    window = torch.ones(
        (channel_count, 1, window_size, window_size),
        device=prediction.device,
        dtype=prediction.dtype
    )

    window = window / float(window_size * window_size)

    mu_prediction = F.conv2d(
        prediction,
        window,
        padding=padding,
        groups=channel_count
    )

    mu_target = F.conv2d(
        target,
        window,
        padding=padding,
        groups=channel_count
    )

    mu_prediction_sq = mu_prediction.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_prediction_target = mu_prediction * mu_target

    sigma_prediction_sq = F.conv2d(
        prediction * prediction,
        window,
        padding=padding,
        groups=channel_count
    ) - mu_prediction_sq

    sigma_target_sq = F.conv2d(
        target * target,
        window,
        padding=padding,
        groups=channel_count
    ) - mu_target_sq

    sigma_prediction_target = F.conv2d(
        prediction * target,
        window,
        padding=padding,
        groups=channel_count
    ) - mu_prediction_target

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = (
        (2.0 * mu_prediction_target + c1)
        * (2.0 * sigma_prediction_target + c2)
    ) / (
        (mu_prediction_sq + mu_target_sq + c1)
        * (sigma_prediction_sq + sigma_target_sq + c2)
    )

    return ssim_map.mean()


def gaussian_blur_2d(
    image: Tensor,
    *,
    kernel_size: int,
    sigma: float
) -> Tensor:
    if kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be odd for Gaussian blur. "
            f"Got {kernel_size}."
        )

    coords = torch.arange(
        kernel_size,
        device=image.device,
        dtype=image.dtype
    ) - (kernel_size - 1) / 2.0

    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    channel_count = image.shape[1]
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.repeat(channel_count, 1, 1, 1)

    padding = kernel_size // 2

    # Reflection padding avoids artificial dark borders in the illumination map
    padded = F.pad(
        image,
        (padding, padding, padding, padding),
        mode="reflect"
    )

    return F.conv2d(padded, kernel_2d, groups=channel_count)
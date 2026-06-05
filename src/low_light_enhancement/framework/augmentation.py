from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from src.low_light_enhancement.framework.config_validation import (
    require_bool,
    require_non_negative_range,
    require_output_size,
    require_positive_range,
    require_probability,
    require_scale_range
)


DEFAULT_AUGMENTATION_ENABLED = False
RESAMPLE_MODE = Image.Resampling.BILINEAR


def is_augmentation_enabled(config: dict[str, Any] | None) -> bool:
    if not config:
        return DEFAULT_AUGMENTATION_ENABLED

    return require_bool(
        config.get("enabled", DEFAULT_AUGMENTATION_ENABLED),
        "augmentation.enabled"
    )


def apply_train_augmentation(
    input_image: Image.Image,
    target_image: Image.Image | None,
    config: dict[str, Any] | None
) -> tuple[Image.Image, Image.Image | None]:
    if not is_augmentation_enabled(config):
        return input_image, target_image

    config = config or {}

    if target_image is not None and input_image.size != target_image.size:
        raise ValueError(
            "Paired augmentation requires input and target images with the "
            f"same size. Got input={input_image.size}, target={target_image.size}."
        )

    input_image, target_image = apply_horizontal_flip(
        input_image,
        target_image,
        config.get("horizontal_flip", {})
    )

    input_image, target_image = apply_random_resized_crop(
        input_image,
        target_image,
        config.get("random_resized_crop", {})
    )

    input_image = apply_color_jitter(
        input_image,
        config.get("color_jitter", {})
    )

    input_image = apply_gamma_jitter(
        input_image,
        config.get("gamma_jitter", {})
    )

    input_image = apply_sensor_noise(
        input_image,
        config.get("sensor_noise", {})
    )

    return input_image, target_image


def apply_horizontal_flip(
    input_image: Image.Image,
    target_image: Image.Image | None,
    config: dict[str, Any]
) -> tuple[Image.Image, Image.Image | None]:
    if not is_transform_enabled(config, "horizontal_flip.enabled"):
        return input_image, target_image

    probability = require_probability(config.get("p", 0.5), "horizontal_flip.p")

    if random.random() >= probability:
        return input_image, target_image

    input_image = ImageOps.mirror(input_image)

    if target_image is not None:
        target_image = ImageOps.mirror(target_image)

    return input_image, target_image


def apply_random_resized_crop(
    input_image: Image.Image,
    target_image: Image.Image | None,
    config: dict[str, Any]
) -> tuple[Image.Image, Image.Image | None]:
    if not is_transform_enabled(config, "random_resized_crop.enabled"):
        return input_image, target_image

    output_size = require_output_size(
        config.get("output_size", input_image.size[0]),
        "random_resized_crop.output_size"
    )

    scale = require_scale_range(
        config.get("scale_min", 0.85),
        config.get("scale_max", 1.0),
        "random_resized_crop.scale"
    )

    ratio = require_positive_range(
        config.get("ratio_min", 0.95),
        config.get("ratio_max", 1.05),
        "random_resized_crop.ratio"
    )

    left, top, width, height = sample_random_resized_crop_params(
        image_width=input_image.size[0],
        image_height=input_image.size[1],
        scale=scale,
        ratio=ratio
    )

    crop_box = (left, top, left + width, top + height)
    input_image = input_image.crop(crop_box).resize(output_size, RESAMPLE_MODE)

    if target_image is not None:
        target_image = target_image.crop(crop_box).resize(
            output_size,
            RESAMPLE_MODE
        )

    return input_image, target_image


def apply_color_jitter(
    image: Image.Image,
    config: dict[str, Any]
) -> Image.Image:
    if not is_transform_enabled(config, "color_jitter.enabled"):
        return image

    probability = require_probability(config.get("p", 0.3), "color_jitter.p")

    if random.random() >= probability:
        return image

    operations = [
        (
            ImageEnhance.Brightness,
            require_non_negative_range(
                config.get("brightness_min", 0.9),
                config.get("brightness_max", 1.1),
                "color_jitter.brightness"
            )
        ),
        (
            ImageEnhance.Contrast,
            require_non_negative_range(
                config.get("contrast_min", 0.9),
                config.get("contrast_max", 1.1),
                "color_jitter.contrast"
            )
        ),
        (
            ImageEnhance.Color,
            require_non_negative_range(
                config.get("saturation_min", 0.95),
                config.get("saturation_max", 1.05),
                "color_jitter.saturation"
            )
        )
    ]

    random.shuffle(operations)

    for enhancer_cls, value_range in operations:
        # Expand (min, max) into random.uniform(min, max)
        factor = random.uniform(*value_range)
        image = enhancer_cls(image).enhance(factor)

    return image


def apply_gamma_jitter(
    image: Image.Image,
    config: dict[str, Any]
) -> Image.Image:
    if not is_transform_enabled(config, "gamma_jitter.enabled"):
        return image

    probability = require_probability(config.get("p", 0.3), "gamma_jitter.p")

    if random.random() >= probability:
        return image

    gamma = random.uniform(
        *require_positive_range(
            config.get("gamma_min", 0.85),
            config.get("gamma_max", 1.15),
            "gamma_jitter.gamma"
        )
    )

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.power(np.clip(image_array, 0.0, 1.0), gamma)

    return array_to_rgb_image(image_array)


def apply_sensor_noise(
    image: Image.Image,
    config: dict[str, Any]
) -> Image.Image:
    if not is_transform_enabled(config, "sensor_noise.enabled"):
        return image

    probability = require_probability(config.get("p", 0.2), "sensor_noise.p")

    if random.random() >= probability:
        return image

    sigma = random.uniform(
        *require_non_negative_range(
            config.get("sigma_min", 0.0),
            config.get("sigma_max", 0.015),
            "sensor_noise.sigma"
        )
    )

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=sigma, size=image_array.shape)
    image_array = np.clip(image_array + noise, 0.0, 1.0)

    return array_to_rgb_image(image_array)


def sample_random_resized_crop_params(
    *,  # next options are keyword-only
    image_width: int,
    image_height: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    max_attempts: int = 10
) -> tuple[int, int, int, int]:
    image_area = image_width * image_height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    for _ in range(max_attempts):
        target_area = image_area * random.uniform(*scale)
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        width = int(round(math.sqrt(target_area * aspect_ratio)))
        height = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < width <= image_width and 0 < height <= image_height:
            left = random.randint(0, image_width - width)
            top = random.randint(0, image_height - height)

            return left, top, width, height

    return center_crop_params(
        image_width=image_width,
        image_height=image_height,
        ratio=ratio
    )


def center_crop_params(
    *,
    image_width: int,
    image_height: int,
    ratio: tuple[float, float]
) -> tuple[int, int, int, int]:
    image_ratio = image_width / image_height

    if image_ratio < ratio[0]:
        width = image_width
        height = int(round(width / ratio[0]))
    elif image_ratio > ratio[1]:
        height = image_height
        width = int(round(height * ratio[1]))
    else:
        width = image_width
        height = image_height

    width = min(width, image_width)
    height = min(height, image_height)
    left = (image_width - width) // 2
    top = (image_height - height) // 2

    return left, top, width, height


def array_to_rgb_image(image_array: np.ndarray) -> Image.Image:
    image_array = (image_array * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_array, mode="RGB")


def is_transform_enabled(config: dict[str, Any], name: str) -> bool:
    return require_bool(config.get("enabled", False), name)
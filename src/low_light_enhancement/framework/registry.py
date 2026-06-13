from __future__ import annotations

from typing import Any

from src.low_light_enhancement.models.fac_ecl_wrapper import FACECLUNetWrapper
from src.low_light_enhancement.models.illumination_guided_unet_wrapper import (
    IlluminationGuidedUNetWrapper
)
from src.low_light_enhancement.models.unet_wrapper import UNetWrapper


MODEL_REGISTRY = {
    "unet": UNetWrapper,
    "illumination_guided_unet": IlluminationGuidedUNetWrapper,
    "fac_ecl": FACECLUNetWrapper
}


def build_model_wrapper(model_name: str) -> Any:
    try:
        wrapper_class = MODEL_REGISTRY[model_name]
    except KeyError as error:
        available_models = ", ".join(sorted(MODEL_REGISTRY))

        raise ValueError(
            f"Unsupported model family: {model_name!r}. "
            f"Available families: {available_models}."
        ) from error

    return wrapper_class()
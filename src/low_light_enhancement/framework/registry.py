from __future__ import annotations

from src.low_light_enhancement.models.unet_wrapper import UNetWrapper


MODEL_REGISTRY = {
    "unet": UNetWrapper
}


def build_model_wrapper(model_name: str) -> UNetWrapper:
    wrapper_class = MODEL_REGISTRY[model_name]
    return wrapper_class()
from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from src.low_light_enhancement.framework.losses import build_loss
from src.low_light_enhancement.models.unet import build_unet


class UNetWrapper:
    def build_model(self, model_config: dict[str, Any]) -> nn.Module:
        config = dict(model_config)
        config.pop("name", None)

        return build_unet(config)

    def build_loss(self, loss_config: dict[str, Any]) -> nn.Module:
        return build_loss(loss_config)

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, Any]
    ) -> Tensor:
        return model(batch["input"])

    def get_prediction(self, output: Tensor) -> Tensor:
        return output
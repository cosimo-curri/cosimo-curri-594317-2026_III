from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from src.low_light_enhancement.framework.losses import (
    FACECLLoss,
    build_loss
)
from src.low_light_enhancement.framework.photometric import (
    apply_photometric_perturbation
)
from src.low_light_enhancement.models.unet import build_unet


# Dictionary output used during FAC-ECL training to store both the
# prediction on the original input and the prediction on the perturbed input
class FACECLOutput(dict[str, Tensor]):
    pass


class FACECLUNetWrapper:
    def __init__(self) -> None:
        self.consistency_config: dict[str, float] | None = None
        self.use_consistency = False

    def build_model(self, model_config: dict[str, Any]) -> nn.Module:
        config = dict(model_config)
        config.pop("name", None)

        return build_unet(config)

    def build_loss(self, loss_config: dict[str, Any]) -> nn.Module:
        loss = build_loss(loss_config)

        if not isinstance(loss, FACECLLoss):
            raise ValueError(
                "FACECLUNetWrapper expects loss.name=\"fac_ecl\". "
                f"Got {loss_config['name']!r}."
            )

        self.consistency_config = loss.consistency_config
        self.use_consistency = loss.consistency_weight > 0.0

        return loss

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, Any]
    ) -> FACECLOutput:
        input_rgb = batch["input"]
        output = FACECLOutput(prediction=model(input_rgb))

        if model.training and self.use_consistency:
            if self.consistency_config is None:
                raise RuntimeError(
                    "FACECLUNetWrapper consistency configuration was not set."
                )

            perturbed_input = apply_photometric_perturbation(
                input_rgb,
                self.consistency_config
            )

            output["consistency_prediction"] = model(perturbed_input)

        return output

    def get_prediction(self, output: FACECLOutput) -> Tensor:
        return output["prediction"]

    def compute_loss(
        self,
        loss_function: nn.Module,
        output: FACECLOutput,
        batch: dict[str, Any]
    ) -> Tensor:
        return loss_function(output, batch["target"])
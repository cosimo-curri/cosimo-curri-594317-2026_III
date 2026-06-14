from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from src.low_light_enhancement.framework.losses import (
    ECLFARLoss,
    build_loss
)
from src.low_light_enhancement.framework.photometric import (
    apply_fixed_photometric_perturbation,
    apply_photometric_perturbation,
    build_fixed_photometric_perturbation_config,
    build_photometric_perturbation_config
)
from src.low_light_enhancement.models.ecl_far import (
    ECLFAROutput,
    build_ecl_far_unet
)


class ECLFARUNetWrapper:
    def __init__(self) -> None:
        self.train_sensitivity_config: dict[str, float] | None = None
        self.eval_sensitivity_config: dict[str, float] | None = None
        self.use_consistency = False

    def build_model(self, model_config: dict[str, Any]) -> nn.Module:
        config = dict(model_config)
        config.pop("name", None)

        self.train_sensitivity_config = build_photometric_perturbation_config(
            config.pop("sensitivity_train", None)
        )

        self.eval_sensitivity_config = build_fixed_photometric_perturbation_config(
            config.pop("sensitivity_eval", None)
        )

        return build_ecl_far_unet(config)

    def build_loss(self, loss_config: dict[str, Any]) -> nn.Module:
        loss = build_loss(loss_config)

        if not isinstance(loss, ECLFARLoss):
            raise ValueError(
                "ECLFARUNetWrapper expects loss.name=\"ecl_far\". "
                f"Got {loss_config['name']!r}."
            )

        self.use_consistency = loss.consistency_weight > 0.0

        return loss

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, Any]
    ) -> ECLFAROutput:
        input_rgb = batch["input"]
        perturbed_input = self.build_sensitivity_input(model, input_rgb)

        return model(
            input_rgb,
            perturbed_input,
            train_perturbed_prediction=self.use_consistency
        )

    def get_prediction(self, output: ECLFAROutput) -> Tensor:
        return output["prediction"]

    def compute_loss(
        self,
        loss_function: nn.Module,
        output: ECLFAROutput,
        batch: dict[str, Any]
    ) -> Tensor:
        return loss_function(output, batch["target"])

    def build_sensitivity_input(
        self,
        model: nn.Module,
        input_rgb: Tensor
    ) -> Tensor:
        if model.training:
            if self.train_sensitivity_config is None:
                raise RuntimeError(
                    "ECLFARUNetWrapper training sensitivity configuration was not set."
                )

            return apply_photometric_perturbation(
                input_rgb,
                self.train_sensitivity_config
            )

        if self.eval_sensitivity_config is None:
            raise RuntimeError(
                "ECLFARUNetWrapper evaluation sensitivity configuration was not set."
            )

        return apply_fixed_photometric_perturbation(
            input_rgb,
            self.eval_sensitivity_config
        )
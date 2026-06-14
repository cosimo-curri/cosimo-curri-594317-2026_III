from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from src.low_light_enhancement.framework.config_validation import (
    require_positive_float,
    require_positive_int
)
from src.low_light_enhancement.models.unet import DoubleConv, build_unet


# Dictionary output used by ECL-FAR to store the first enhancement, the
# perturbed-input enhancement and the refined final prediction
class ECLFAROutput(dict[str, Tensor]):
    pass


class ECLFARUNet(nn.Module):
    def __init__(
        self,
        *,  # next options are keyword-only
        backbone_config: Mapping[str, Any],
        refinement_channels: int = 32,
        residual_scale: float = 0.15
    ) -> None:
        super().__init__()

        self.enhancer = build_unet(backbone_config)

        self.refinement = ECLFARRefinementModule(
            refinement_channels=refinement_channels,
            residual_scale=residual_scale,
            normalization=str(backbone_config.get("normalization", "none")),
            activation=str(backbone_config.get("activation", "relu")),
            group_count=int(backbone_config.get("group_count", 8))
        )

    def forward(
        self,
        input_rgb: Tensor,
        perturbed_input: Tensor,
        *,
        train_perturbed_prediction: bool = True
    ) -> ECLFAROutput:
        initial_prediction = self.enhancer(input_rgb)

        if train_perturbed_prediction:
            perturbed_prediction = self.enhancer(perturbed_input)
        else:
            with torch.no_grad():
                perturbed_prediction = self.enhancer(perturbed_input)

        instability_map = compute_instability_map(
            initial_prediction,
            perturbed_prediction
        )

        prediction, residual, gate = self.refinement(
            input_rgb=input_rgb,
            initial_prediction=initial_prediction,
            instability_map=instability_map
        )

        return ECLFAROutput(
            prediction=prediction,
            initial_prediction=initial_prediction,
            perturbed_prediction=perturbed_prediction,
            instability_map=instability_map,
            residual=residual,
            gate=gate
        )


class ECLFARRefinementModule(nn.Module):
    def __init__(
        self,
        *,
        refinement_channels: int = 32,
        residual_scale: float = 0.15,
        normalization: str = "none",
        activation: str = "relu",
        group_count: int = 8
    ) -> None:
        super().__init__()

        self.refinement_channels = require_positive_int(
            refinement_channels,
            "model.refinement_channels"
        )

        self.residual_scale = require_positive_float(
            residual_scale,
            "model.residual_scale"
        )

        # RGB input + first RGB prediction + one-channel instability map
        self.features = nn.Sequential(
            DoubleConv(
                7,
                self.refinement_channels,
                normalization=normalization,
                activation=activation,
                group_count=group_count
            ),
            DoubleConv(
                self.refinement_channels,
                self.refinement_channels,
                normalization=normalization,
                activation=activation,
                group_count=group_count
            )
        )

        # Three residual channels and one gate channel
        self.output_conv = nn.Conv2d(
            self.refinement_channels,
            4,
            kernel_size=1
        )

        self.reset_output_conv()

    def forward(
        self,
        *,
        input_rgb: Tensor,
        initial_prediction: Tensor,
        instability_map: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        refinement_input = torch.cat(
            [input_rgb, initial_prediction, instability_map],
            dim=1
        )

        correction = self.output_conv(self.features(refinement_input))
        residual_logits = correction[:, :3]
        gate_logits = correction[:, 3:4]

        residual = torch.tanh(residual_logits) * self.residual_scale
        gate = torch.sigmoid(gate_logits)

        prediction = (initial_prediction + gate * residual).clamp(0.0, 1.0)

        return prediction, residual, gate

    def reset_output_conv(self) -> None:
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)


def build_ecl_far_unet(config: Mapping[str, Any] | None = None) -> ECLFARUNet:
    if config is None:
        config = {}

    config = dict(config)

    refinement_channels = config.pop("refinement_channels", 32)
    residual_scale = config.pop("residual_scale", 0.15)

    return ECLFARUNet(
        backbone_config=config,
        refinement_channels=refinement_channels,
        residual_scale=residual_scale
    )


def compute_instability_map(
    initial_prediction: Tensor,
    perturbed_prediction: Tensor
) -> Tensor:
    if initial_prediction.shape != perturbed_prediction.shape:
        raise ValueError(
            "ECL-FAR instability requires predictions with the same shape. "
            f"Got {tuple(initial_prediction.shape)} and "
            f"{tuple(perturbed_prediction.shape)}."
        )

    return (initial_prediction - perturbed_prediction).abs().mean(
        dim=1,
        keepdim=True
    ).detach()
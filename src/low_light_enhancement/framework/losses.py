from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F

from src.low_light_enhancement.framework.image_ops import compute_ssim


class L1SSIMLoss(nn.Module):
    def __init__(
        self,
        *,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        window_size: int = 11
    ) -> None:
        super().__init__()

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


def build_loss(config: dict[str, float | str]) -> nn.Module:
    loss_name = config["name"]

    if loss_name == "l1_ssim":
        return L1SSIMLoss(
            l1_weight=float(config["l1_weight"]),
            ssim_weight=float(config["ssim_weight"])
        )

    raise ValueError(f"Unsupported loss: {loss_name!r}.")
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Standard U-Net block
# Two consecutive convolutions let each U-Net level learn richer local features
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,  # next options are keyword-only
        normalization: str = "none",
        activation: str = "relu",
        group_count: int = 8
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            _conv_norm_activation(
                in_channels,
                out_channels,
                normalization=normalization,
                activation=activation,
                group_count=group_count
            ),
            _conv_norm_activation(
                out_channels,
                out_channels,
                normalization=normalization,
                activation=activation,
                group_count=group_count
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# Downsampling
# 2x2 max-pooling followed by DoubleConv
class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        normalization: str = "none",
        activation: str = "relu",
        group_count: int = 8
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            DoubleConv(
                in_channels,
                out_channels,
                normalization=normalization,
                activation=activation,
                group_count=group_count
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# Upsampling with skip connections
class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        normalization: str = "none",
        activation: str = "relu",
        group_count: int = 8,
        upsampling: str = "bilinear"
    ) -> None:
        super().__init__()

        self.upsampling = upsampling

        if upsampling == "bilinear":  # upsamples spatially without learnable weights
            conv_in_channels = in_channels + skip_channels
        elif upsampling == "transposed":  # transposed convolution with learnable weights
            self.up = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2
            )

            conv_in_channels = out_channels + skip_channels
        else:
            raise ValueError(
                "Unsupported upsampling. "
                "Expected one of {\"bilinear\", \"transposed\"}, "
                f"got {upsampling!r}."
            )

        self.conv = DoubleConv(
            conv_in_channels,
            out_channels,
            normalization=normalization,
            activation=activation,
            group_count=group_count
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        if self.upsampling == "bilinear":
            x = F.interpolate(  # match the decoder feature map to the skip resolution
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        else:
            x = self.up(x)

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(  # match the decoder feature map to the skip resolution
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        normalization: str = "none",
        activation: str = "relu",
        group_count: int = 8,
        upsampling: str = "bilinear"
    ) -> None:
        super().__init__()

        self.in_channels = _validate_positive_int(in_channels, "in_channels")
        self.out_channels = _validate_positive_int(out_channels, "out_channels")
        self.channels = _validate_channels(channels)

        self.input_block = DoubleConv(
            self.in_channels,
            self.channels[0],
            normalization=normalization,
            activation=activation,
            group_count=group_count
        )

        self.down_blocks = nn.ModuleList(
            DownBlock(
                current_channels,
                next_channels,
                normalization=normalization,
                activation=activation,
                group_count=group_count
            )
            for current_channels, next_channels in zip(
                self.channels[:-1],
                self.channels[1:],
                strict=True
            )
        )

        self.up_blocks = nn.ModuleList(
            UpBlock(
                in_channels=current_channels,
                skip_channels=skip_channels,
                out_channels=skip_channels,
                normalization=normalization,
                activation=activation,
                group_count=group_count,
                upsampling=upsampling
            )
            for current_channels, skip_channels in zip(
                reversed(self.channels[1:]),
                reversed(self.channels[:-1]),
                strict=True
            )
        )

        self.output_conv = nn.Conv2d(self.channels[0], self.out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skips: list[Tensor] = []

        x = self.input_block(x)
        skips.append(x)

        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)

        skips.pop()  # no skip connection is needed at the bottom of the U

        for up_block, skip in zip(self.up_blocks, reversed(skips), strict=True):
            x = up_block(x, skip)

        x = self.output_conv(x)
        return torch.sigmoid(x)


def build_unet(config: Mapping[str, Any] | None = None) -> UNet:  # from config files
    if config is None:
        config = {}

    return UNet(**dict(config))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# Conv (used twice in DoubleConv)
# Convolution -> optional normalization -> activation
def _conv_norm_activation(
    in_channels: int,
    out_channels: int,
    *,
    normalization: str,
    activation: str,
    group_count: int
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=_uses_bias(normalization)
        )
    ]

    norm_layer = _build_normalization(
        out_channels,
        normalization=normalization,
        group_count=group_count
    )

    if norm_layer is not None:
        layers.append(norm_layer)

    layers.append(_build_activation(activation))
    return nn.Sequential(*layers)


def _build_normalization(
    num_channels: int,
    *,
    normalization: str,
    group_count: int
) -> nn.Module | None:
    if normalization == "none":
        return None

    if normalization == "batch":
        return nn.BatchNorm2d(num_channels)

    if normalization == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)

    if normalization == "group":
        group_count = _validate_positive_int(group_count, "group_count")

        if num_channels % group_count != 0:
            raise ValueError(
                "group_count must divide the number of channels. "
                f"Got num_channels={num_channels} and group_count={group_count}."
            )

        return nn.GroupNorm(num_groups=group_count, num_channels=num_channels)

    raise ValueError(
        "Unsupported normalization. "
        "Expected one of {\"none\", \"batch\", \"instance\", \"group\"}, "
        f"got {normalization!r}."
    )


def _build_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=True)

    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)

    raise ValueError(
        "Unsupported activation. "
        "Expected one of {\"relu\", \"leaky_relu\"}, "
        f"got {activation!r}."
    )


# Normalization layers already learn a shift, so Conv2d bias is used only without normalization
def _uses_bias(normalization: str) -> bool:
    return normalization == "none"


def _validate_channels(channels: Sequence[int]) -> tuple[int, ...]:
    if len(channels) < 2:
        raise ValueError(
            "channels must contain at least two values: "
            "one encoder width and one bottleneck width."
        )

    validated_channels = tuple(
        _validate_positive_int(channel, f"channels[{index}]")
        for index, channel in enumerate(channels)
    )

    return validated_channels


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    return value
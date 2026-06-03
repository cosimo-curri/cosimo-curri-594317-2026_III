from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def image_to_tensor(image: Image.Image) -> Tensor:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))

    return torch.from_numpy(image_array)


def tensor_to_image(tensor: Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    image_array = tensor.permute(1, 2, 0).numpy()
    image_array = (image_array * 255.0).round().astype(np.uint8)

    return Image.fromarray(image_array, mode="RGB")
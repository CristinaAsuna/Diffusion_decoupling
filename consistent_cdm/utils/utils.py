from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    array = tensor.numpy()
    if array.shape[0] == 1:
        return (array[0] * 255.0).round().astype(np.uint8)
    image = np.transpose(array, (1, 2, 0))
    return (image * 255.0).round().astype(np.uint8)


def save_tensor_png(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(tensor)).save(path)


def save_tensor_npy(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor.detach().float().cpu().numpy())


def save_tensor_gif(
    tensors: list[torch.Tensor],
    path: str | Path,
    duration_ms: int = 120,
    loop: int = 0,
) -> None:
    if not tensors:
        raise ValueError("tensors must not be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(tensor_to_uint8_image(tensor)) for tensor in tensors]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
    )


def save_tensor_grid_png(
    tensors: list[torch.Tensor],
    path: str | Path,
    columns: int = 5,
    padding: int = 4,
) -> None:
    if not tensors:
        raise ValueError("tensors must not be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(tensor_to_uint8_image(tensor)) for tensor in tensors]
    width, height = images[0].size
    columns = max(1, columns)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new(
        "L" if images[0].mode == "L" else "RGB",
        (
            columns * width + max(columns - 1, 0) * padding,
            rows * height + max(rows - 1, 0) * padding,
        ),
    )
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = col * (width + padding)
        y = row * (height + padding)
        canvas.paste(image, (x, y))
    canvas.save(path)

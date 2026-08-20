from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessResult:
    tensor: np.ndarray
    scale: float
    pad_x: int
    pad_y: int


def preprocess(image: Image.Image, size: int = 640) -> PreprocessResult:
    """
    Convert an input image into the NCHW FP32 tensor expected by YOLOv8.

    Returns:
        tensor: [1, 3, size, size] float32 tensor in the [0, 1] range.
        scale: Resize scale applied to the original image.
        pad_x: Horizontal letterbox padding.
        pad_y: Vertical letterbox padding.
    """

    image = image.convert("RGB")

    width, height = image.size

    scale = min(size / width, size / height)

    resized_width = round(width * scale)
    resized_height = round(height * scale)

    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    canvas = Image.new(
        "RGB",
        (size, size),
        (114, 114, 114),
    )

    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2

    canvas.paste(resized, (pad_x, pad_y))

    array = np.asarray(canvas, dtype=np.float32)

    # HWC -> CHW
    array = array.transpose(2, 0, 1)

    # Add batch dimension: CHW -> NCHW
    array = array[None, ...]

    # Normalize [0, 255] -> [0, 1]
    array /= 255.0

    # Ensure contiguous memory for ONNX Runtime
    tensor = np.ascontiguousarray(array)

    return PreprocessResult(
        tensor=tensor,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
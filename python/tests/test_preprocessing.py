import numpy as np
from PIL import Image

from services.inference.preprocessing import preprocess


def test_preprocess_output_shape_and_dtype():
    image = Image.new("RGB", (1920, 1080))

    result = preprocess(image, 640)

    assert result.tensor.shape == (1, 3, 640, 640)
    assert result.tensor.dtype == np.float32


def test_preprocess_output_range():
    image = Image.new("RGB", (800, 600), color=(255, 128, 0))

    result = preprocess(image, 640)

    assert result.tensor.min() >= 0.0
    assert result.tensor.max() <= 1.0


def test_preprocess_returns_scaling_information():
    image = Image.new("RGB", (1920, 1080))

    result = preprocess(image, 640)

    assert result.scale == 640 / 1920
    assert result.pad_x == 0
    assert result.pad_y == 140
import numpy as np
from PIL import Image

from services.inference.detector import letterbox, nms


def test_letterbox_makes_normalized_nchw_tensor():
    tensor, scale, padding = letterbox(Image.new("RGB", (400, 200), "white"), 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert scale == 1.6 and padding == (0, 160)


def test_nms_discards_overlapping_low_score_box():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [30, 30, 40, 40]], dtype=np.float32)
    assert nms(boxes, np.array([0.9, 0.7, 0.8]), 0.45) == [0, 2]

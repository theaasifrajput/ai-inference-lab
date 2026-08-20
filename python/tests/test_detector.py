from pathlib import Path

import numpy as np
from PIL import Image

from services.inference.detector import YoloOnnxDetector
from services.inference.preprocessing import preprocess


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "yolov8n.onnx"


def test_yolo_onnx_inference():
    image = Image.new(
        "RGB",
        (640, 640),
        color=(128, 128, 128),
    )

    preprocessing_result = preprocess(image, 640)

    detector = YoloOnnxDetector(MODEL_PATH)

    output, elapsed_ms = detector.infer(
        preprocessing_result.tensor
    )

    assert output.shape == (1, 84, 8400)
    assert output.dtype == np.float32
    assert elapsed_ms > 0
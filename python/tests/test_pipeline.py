from pathlib import Path

from PIL import Image

from services.inference.pipeline import InferencePipeline


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "yolov8n.onnx"


def test_end_to_end_inference():
    image = Image.new(
        "RGB",
        (640, 640),
        color=(128, 128, 128),
    )

    pipeline = InferencePipeline(
        model_path=MODEL_PATH,
        input_size=640,
        confidence_threshold=0.35,
        nms_threshold=0.45,
    )

    detections, metrics = pipeline.infer(image)

    assert isinstance(detections, list)

    assert metrics["preprocess_ms"] >= 0
    assert metrics["inference_ms"] > 0
    assert metrics["postprocess_ms"] >= 0
    assert metrics["total_ms"] > 0
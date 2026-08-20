from pathlib import Path

from PIL import Image

from services.inference.pipeline import InferencePipeline


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "yolov8n.onnx"
IMAGE_PATH = Path(__file__).parent / "data" / "person.jpg"


def test_real_person_detection():
    image = Image.open(IMAGE_PATH)

    pipeline = InferencePipeline(
        model_path=MODEL_PATH,
        input_size=640,
        confidence_threshold=0.35,
        nms_threshold=0.45,
    )

    detections, metrics = pipeline.infer(image)

    persons = [
        detection
        for detection in detections
        if detection.label == "person"
    ]

    print(f"\nPreprocess:   {metrics['preprocess_ms']:.2f} ms")
    print(f"Inference:    {metrics['inference_ms']:.2f} ms")
    print(f"Postprocess:  {metrics['postprocess_ms']:.2f} ms")
    print(f"Total:        {metrics['total_ms']:.2f} ms")

    print("\nDetections:")

    for detection in persons:
        print(
            f"  {detection.label}: "
            f"{detection.confidence:.3f} "
            f"{detection.box}"
        )

    assert len(persons) > 0

    width, height = image.size

    for detection in persons:
        x1, y1, x2, y2 = detection.box

        assert 0 <= x1 <= width
        assert 0 <= y1 <= height
        assert 0 <= x2 <= width
        assert 0 <= y2 <= height

        assert x2 > x1
        assert y2 > y1
        assert detection.confidence >= 0.35
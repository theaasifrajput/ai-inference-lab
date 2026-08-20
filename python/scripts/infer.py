from pathlib import Path
import sys

from PIL import Image, ImageDraw

from services.inference.pipeline import InferencePipeline


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT.parent / "models" / "yolov8n.onnx"


def draw_detections(image: Image.Image, detections):
    output = image.copy()
    draw = ImageDraw.Draw(output)

    for detection in detections:
        x1, y1, x2, y2 = detection.box

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        label = (
            f"{detection.label} "
            f"{detection.confidence:.2f}"
        )

        draw.text(
            (x1, max(0, y1 - 20)),
            label,
            fill="red",
        )

    return output


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.infer <image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")

    pipeline = InferencePipeline(
        model_path=MODEL_PATH,
        input_size=640,
        confidence_threshold=0.35,
        nms_threshold=0.45,
    )

    detections, metrics = pipeline.infer(image)

    print(f"\nImage: {image_path}")
    print(f"Size:  {image.size}")

    print("\nPerformance:")
    print(f"  Preprocess:  {metrics['preprocess_ms']:.2f} ms")
    print(f"  Inference:   {metrics['inference_ms']:.2f} ms")
    print(f"  Postprocess: {metrics['postprocess_ms']:.2f} ms")
    print(f"  Total:       {metrics['total_ms']:.2f} ms")

    print("\nDetections:")

    if not detections:
        print("  No detections")
        return

    for detection in detections:
        print(
            f"  {detection.label}: "
            f"{detection.confidence:.3f} "
            f"box={detection.box}"
        )

    annotated = draw_detections(
        image,
        detections,
    )

    output_path = (
        image_path.parent
        / f"{image_path.stem}_detected.jpg"
    )

    annotated.save(output_path)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
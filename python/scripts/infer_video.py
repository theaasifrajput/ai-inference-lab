from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
from PIL import Image

from services.inference.pipeline import InferencePipeline


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT.parent / "models" / "yolov8n.onnx"


def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.box)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        label = (
            f"{detection.label} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python -m scripts.infer_video <video>")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.is_file():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    output_path = (
        video_path.parent
        / f"{video_path.stem}_detected.mp4"
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    pipeline = InferencePipeline(
        model_path=MODEL_PATH,
        input_size=640,
        confidence_threshold=0.35,
        nms_threshold=0.45,
        providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
    )

    print(f"Input:       {video_path}")
    print(f"Resolution:  {width}x{height}")
    print(f"FPS:         {fps:.2f}")
    print(f"Frames:      {frame_count}")
    print(f"Output:      {output_path}")

    frame_index = 0
    total_inference_time = 0.0

    started = time.perf_counter()

    while True:
        success, frame = cap.read()

        if not success:
            break

        # OpenCV uses BGR.
        # PIL expects RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(rgb_frame)

        detections, metrics = pipeline.infer(image)

        total_inference_time += metrics["total_ms"]

        draw_detections(
            frame,
            detections,
        )

        writer.write(frame)

        frame_index += 1

        if frame_index % 30 == 0:
            print(
                f"Processed {frame_index}/"
                f"{frame_count} frames"
            )

    elapsed = time.perf_counter() - started

    cap.release()
    writer.release()

    average_ms = (
        total_inference_time / frame_index
        if frame_index
        else 0
    )

    processing_fps = (
        frame_index / elapsed
        if elapsed > 0
        else 0
    )

    print("\n========== Video Inference ==========")
    print(f"Processed frames:    {frame_index}")
    print(f"Total time:          {elapsed:.2f} sec")
    print(f"Average inference:   {average_ms:.2f} ms/frame")
    print(f"Processing FPS:      {processing_fps:.2f}")
    print(f"Video FPS:           {fps:.2f}")
    print(f"Saved:               {output_path}")


if __name__ == "__main__":
    main()
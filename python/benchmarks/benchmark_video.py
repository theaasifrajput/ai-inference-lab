from __future__ import annotations

import statistics
import time
from pathlib import Path

import cv2
from PIL import Image

from services.inference.pipeline import InferencePipeline


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT.parent / "models" / "yolov8n.onnx"
VIDEO_PATH = ROOT / "tests" / "data" / "walking_person.mp4"

WARMUP_FRAMES = 10
BENCHMARK_FRAMES = 100


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)

    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * weight
    )


def print_stats(name: str, values: list[float]) -> None:
    print(f"\n{name}")

    print(f"  Mean: {statistics.mean(values):.3f} ms")
    print(f"  P50:  {percentile(values, 0.50):.3f} ms")
    print(f"  P95:  {percentile(values, 0.95):.3f} ms")
    print(f"  P99:  {percentile(values, 0.99):.3f} ms")


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("========== Video Benchmark ==========")
    print(f"Video FPS:       {video_fps:.2f}")
    print(f"Total frames:    {frame_count}")
    print(f"Benchmark frames: {BENCHMARK_FRAMES}")

    pipeline = InferencePipeline(
        model_path=MODEL_PATH,
        input_size=640,
        confidence_threshold=0.35,
        nms_threshold=0.45,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    # -------------------------------------------------
    # Warmup
    # -------------------------------------------------

    print("\nRunning warmup...")

    for _ in range(WARMUP_FRAMES):
        success, frame = cap.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(rgb_frame)

        pipeline.infer(image)

    # Restart video
    cap.release()

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    # -------------------------------------------------
    # Benchmark
    # -------------------------------------------------

    decode_times = []
    conversion_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_pipeline_times = []

    total_times = []

    for frame_index in range(BENCHMARK_FRAMES):

        # ---------------------------------------------
        # Frame decode
        # ---------------------------------------------

        start = time.perf_counter()

        success, frame = cap.read()

        decode_ms = (
            time.perf_counter() - start
        ) * 1000

        if not success:
            break

        decode_times.append(decode_ms)

        # ---------------------------------------------
        # BGR → RGB + PIL
        # ---------------------------------------------

        start = time.perf_counter()

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(rgb_frame)

        conversion_ms = (
            time.perf_counter() - start
        ) * 1000

        conversion_times.append(conversion_ms)

        # ---------------------------------------------
        # Full pipeline
        # ---------------------------------------------

        start = time.perf_counter()

        _, metrics = pipeline.infer(image)

        total_pipeline_ms = (
            time.perf_counter() - start
        ) * 1000

        total_pipeline_times.append(
            total_pipeline_ms
        )

        preprocess_times.append(
            metrics["preprocess_ms"]
        )

        inference_times.append(
            metrics["inference_ms"]
        )

        postprocess_times.append(
            metrics["postprocess_ms"]
        )

        # ---------------------------------------------
        # Complete frame time
        # ---------------------------------------------

        total_ms = (
            decode_ms
            + conversion_ms
            + total_pipeline_ms
        )

        total_times.append(total_ms)

    cap.release()

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print("\n========== Results ==========")

    print_stats(
        "Frame Decode",
        decode_times,
    )

    print_stats(
        "BGR → RGB + PIL",
        conversion_times,
    )

    print_stats(
        "Preprocessing",
        preprocess_times,
    )

    print_stats(
        "ONNX Runtime",
        inference_times,
    )

    print_stats(
        "Post-processing",
        postprocess_times,
    )

    print_stats(
        "Pipeline",
        total_pipeline_times,
    )

    print_stats(
        "Complete Frame",
        total_times,
    )

    mean_total = statistics.mean(total_times)

    throughput = 1000.0 / mean_total

    print("\nThroughput")
    print(f"  {throughput:.2f} frames/sec")

    print("\nVideo")
    print(f"  Source FPS:     {video_fps:.2f}")
    print(f"  Processing FPS: {throughput:.2f}")


if __name__ == "__main__":
    main()
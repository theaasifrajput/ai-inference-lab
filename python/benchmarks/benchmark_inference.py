from __future__ import annotations

import statistics
import time
from pathlib import Path

from PIL import Image

from services.inference.pipeline import InferencePipeline


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT.parent / "models" / "yolov8n.onnx"
IMAGE_PATH = ROOT / "tests" / "data" / "person.jpg"

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)

    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower]
        + weight * (values[upper] - values[lower])
    )


def print_stats(name: str, values: list[float]) -> None:
    print(f"\n{name}")

    print(f"  Mean: {statistics.mean(values):.3f} ms")
    print(f"  P50:  {percentile(values, 0.50):.3f} ms")
    print(f"  P95:  {percentile(values, 0.95):.3f} ms")
    print(f"  P99:  {percentile(values, 0.99):.3f} ms")


def main() -> None:
    image = Image.open(IMAGE_PATH).convert("RGB")

    pipeline = InferencePipeline(
        model_path=MODEL_PATH,
        input_size=640,
        confidence_threshold=0.35,
        nms_threshold=0.45,
    )

    print("Running warmup...")

    for _ in range(WARMUP_RUNS):
        pipeline.infer(image)

    preprocess_times: list[float] = []
    inference_times: list[float] = []
    postprocess_times: list[float] = []
    total_times: list[float] = []

    print(
        f"Running benchmark "
        f"({BENCHMARK_RUNS} iterations)..."
    )

    for _ in range(BENCHMARK_RUNS):
        _, metrics = pipeline.infer(image)

        preprocess_times.append(
            metrics["preprocess_ms"]
        )

        inference_times.append(
            metrics["inference_ms"]
        )

        postprocess_times.append(
            metrics["postprocess_ms"]
        )

        total_times.append(
            metrics["total_ms"]
        )

    print("\n========== Python Inference Benchmark ==========")

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
        "End-to-end",
        total_times,
    )

    mean_total = statistics.mean(total_times)

    throughput = 1000.0 / mean_total

    print("\nThroughput")
    print(f"  {throughput:.2f} images/sec")


if __name__ == "__main__":
    main()
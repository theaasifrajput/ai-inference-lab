from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import onnxruntime as ort
import yaml
from PIL import Image

from services.inference.pipeline import InferencePipeline


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AI inference video benchmark"
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to benchmark configuration",
    )

    parser.add_argument(
        "--provider",
        choices=[
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ],
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to benchmark result JSON",
    )

    return parser.parse_args()


def percentile(
    values: list[float],
    p: float,
) -> float:
    values = sorted(values)

    if not values:
        return 0.0

    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * weight
    )


def calculate_stats(
    values: list[float],
) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def print_stats(
    name: str,
    values: list[float],
) -> None:
    stats = calculate_stats(values)

    print(f"\n{name}")

    print(f"  Mean: {stats['mean']:.3f} ms")
    print(f"  P50:  {stats['p50']:.3f} ms")
    print(f"  P95:  {stats['p95']:.3f} ms")
    print(f"  P99:  {stats['p99']:.3f} ms")
    print(f"  Min:  {stats['min']:.3f} ms")
    print(f"  Max:  {stats['max']:.3f} ms")


def get_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        lines = result.stdout.strip().splitlines()

        if lines:
            return lines[0]

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        pass

    return None


def get_cuda_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.splitlines():
            if "CUDA Version" in line:
                return line.split(
                    "CUDA Version:"
                )[-1].strip()

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        pass

    return None


def load_config(path: Path) -> dict:
    with path.open("r") as file:
        return yaml.safe_load(file)


def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()
    output_path = args.output.resolve()

    config = load_config(config_path)

    benchmark_config = config["benchmark"]
    model_config = config["model"]
    workload_config = config["workload"]

    model_path = (
        REPO_ROOT / model_config["path"]
    )

    video_path = (
        REPO_ROOT / workload_config["input"]
    )

    input_size = workload_config[
        "input_size"
    ]

    precision = workload_config[
        "precision"
    ]

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    print(
        "========== Video Benchmark =========="
    )

    print(
        f"Benchmark:        "
        f"{benchmark_config['name']}"
    )

    print(
        f"Model:            "
        f"{model_config['name']}"
    )

    print(
        f"Model path:       "
        f"{model_path}"
    )

    print(
        f"Video:            "
        f"{video_path}"
    )

    print(
        f"Provider:         "
        f"{args.provider}"
    )

    print(
        f"Precision:        "
        f"{precision}"
    )

    print(
        f"Input size:       "
        f"{input_size['width']}x"
        f"{input_size['height']}"
    )

    # -------------------------------------------------
    # Validate paths
    # -------------------------------------------------

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # -------------------------------------------------
    # ONNX Runtime
    # -------------------------------------------------

    available_providers = (
        ort.get_available_providers()
    )

    print(
        "\nAvailable ORT providers: "
        f"{available_providers}"
    )

    if args.provider not in available_providers:
        raise RuntimeError(
            f"Requested provider "
            f"'{args.provider}' is not available. "
            f"Available providers: "
            f"{available_providers}"
        )

    # -------------------------------------------------
    # Open video
    # -------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Video FPS:        "
        f"{video_fps:.2f}"
    )

    print(
        f"Total frames:     "
        f"{total_frames}"
    )

    # -------------------------------------------------
    # Create inference pipeline
    # -------------------------------------------------

    pipeline = InferencePipeline(
        model_path=model_path,
        input_size=input_size["width"],
        confidence_threshold=0.35,
        nms_threshold=0.45,
        providers=[
            args.provider,
        ],
    )

    # -------------------------------------------------
    # Benchmark storage
    # -------------------------------------------------

    decode_times: list[float] = []

    preprocess_times: list[float] = []

    inference_times: list[float] = []

    postprocess_times: list[float] = []

    end_to_end_times: list[float] = []

    # -------------------------------------------------
    # Benchmark
    # -------------------------------------------------

    print(
        "\nRunning benchmark..."
    )

    processed_frames = 0

    while True:

        # ---------------------------------------------
        # Decode
        # ---------------------------------------------

        decode_start = time.perf_counter()

        success, frame = cap.read()

        decode_ms = (
            time.perf_counter()
            - decode_start
        ) * 1000.0

        if not success:
            break

        decode_times.append(
            decode_ms
        )

        # ---------------------------------------------
        # BGR → RGB + PIL
        # ---------------------------------------------

        conversion_start = (
            time.perf_counter()
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            rgb_frame
        )

        conversion_ms = (
            time.perf_counter()
            - conversion_start
        ) * 1000.0

        # ---------------------------------------------
        # Inference pipeline
        # ---------------------------------------------

        _, metrics = pipeline.infer(
            image
        )

        # ---------------------------------------------
        # Preprocessing
        #
        # Includes:
        #
        #   BGR → RGB
        #   PIL conversion
        #   model preprocessing
        # ---------------------------------------------

        preprocess_ms = (
            conversion_ms
            + metrics["preprocess_ms"]
        )

        preprocess_times.append(
            preprocess_ms
        )

        # ---------------------------------------------
        # Inference
        # ---------------------------------------------

        inference_ms = (
            metrics["inference_ms"]
        )

        inference_times.append(
            inference_ms
        )

        # ---------------------------------------------
        # Post-processing
        # ---------------------------------------------

        postprocess_ms = (
            metrics["postprocess_ms"]
        )

        postprocess_times.append(
            postprocess_ms
        )

        # ---------------------------------------------
        # End-to-end
        #
        # decode
        # + preprocess
        # + inference
        # + postprocess
        # ---------------------------------------------

        end_to_end_ms = (
            decode_ms
            + preprocess_ms
            + inference_ms
            + postprocess_ms
        )

        end_to_end_times.append(
            end_to_end_ms
        )

        processed_frames += 1

    cap.release()

    if not end_to_end_times:
        raise RuntimeError(
            "No frames were processed"
        )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print(
        "\n========== Results =========="
    )

    print_stats(
        "Frame Decode",
        decode_times,
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
        "End-to-end",
        end_to_end_times,
    )

    # -------------------------------------------------
    # Throughput
    # -------------------------------------------------

    mean_end_to_end = (
        statistics.mean(
            end_to_end_times
        )
    )

    throughput = (
        1000.0 /
        mean_end_to_end
    )

    print(
        "\nThroughput"
    )

    print(
        f"  {throughput:.2f} frames/sec"
    )

    print(
        "\nFrames"
    )

    print(
        f"  Total:     {total_frames}"
    )

    print(
        f"  Processed: {processed_frames}"
    )

    # -------------------------------------------------
    # Result JSON
    # -------------------------------------------------

    result = {
        "benchmark": {
            "name": benchmark_config["name"],
            "version": benchmark_config["version"],
        },

        "environment": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "gpu": get_gpu_name(),
            "cuda_version": get_cuda_version(),
        },

        "implementation": {
            "language": "python",
            "runtime": "onnxruntime",
            "execution_provider": args.provider,
        },

        "model": {
            "name": model_config["name"],
            "path": model_config["path"],
            "input_width": input_size["width"],
            "input_height": input_size["height"],
            "precision": precision,
        },

        "workload": {
            "type": workload_config["type"],
            "input": workload_config["input"],
            "total_frames": total_frames,
            "processed_frames": processed_frames,
        },

        "latency_ms": {
            "decode": calculate_stats(
                decode_times
            ),

            "preprocess": calculate_stats(
                preprocess_times
            ),

            "inference": calculate_stats(
                inference_times
            ),

            "postprocess": calculate_stats(
                postprocess_times
            ),

            "end_to_end": calculate_stats(
                end_to_end_times
            ),
        },

        "throughput": {
            "fps": throughput,
        },
    }

    # -------------------------------------------------
    # Write result
    # -------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        f"\nResults written to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
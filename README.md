# AI Inference Lab

Production-oriented AI inference experiments focused on **latency, throughput, C++, CUDA, ONNX Runtime, and inference systems**.

The project uses YOLOv8n as a reproducible computer-vision workload and provides both Python and C++ inference implementations. The goal is to establish a correct reference pipeline first, then progressively optimize the complete inference path.

## Current Status

* YOLOv8n model exported to ONNX
* Python inference pipeline using ONNX Runtime
* Image preprocessing: RGB conversion, aspect-ratio-preserving resize, letterbox padding, HWC → CHW, NCHW FP32 normalization
* YOLOv8 post-processing with confidence filtering, box conversion, coordinate restoration, and NMS
* Stage-level latency measurements for preprocessing, inference, post-processing, and end-to-end execution
* Image and video inference entry points
* C++20 inference implementation using OpenCV + ONNX Runtime
* CMake + Ninja + vcpkg build configuration for Linux
* CPU/CUDA execution-provider experimentation through ONNX Runtime

## Architecture

```text
                         AI Inference Lab
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
          Python Inference              C++ Inference
                 │                             │
          ONNX Runtime                 ONNX Runtime
                 │                             │
             CPU / CUDA                  CPU / CUDA
                 │                             │
          Post-processing             Post-processing
                 │                             │
               Result                       Result
```

The Python implementation is the reference pipeline. The C++ implementation provides a lower-overhead path for performance experiments while keeping the model contract and inference stages aligned.

## Repository Layout

```text
ai-inference-lab/
├── cpp/                    # C++20 inference implementation
│   ├── include/            # C++ headers
│   ├── src/                # Preprocess, inference, postprocess, utilities
│   └── tests/              # C++ test data
├── python/
│   ├── benchmarks/         # Image and video benchmarks
│   ├── scripts/            # CLI inference entry points
│   └── services/
│       ├── gateway/        # API gateway layer
│       └── inference/      # Preprocess, runtime, postprocess, models
├── docs/                   # Technical documentation
├── models/                 # Local model artifacts; not committed
├── output/                 # Generated inference output
├── CMakeLists.txt          # Top-level CMake project
├── CMakePresets.json       # Linux Debug/Release presets
└── vcpkg.json              # C++ dependencies
```

## Quick Start — Python

### 1. Clone

```bash
git clone https://github.com/theaasifrajput/ai-inference-lab.git
cd ai-inference-lab
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

From the `python/` directory:

```bash
cd python
python -m pip install -r requirements.txt
```

The inference implementation uses NumPy, Pillow, OpenCV, and ONNX Runtime.

### 4. Prepare the Model

From the `python/` directory:

```bash
python scripts/download_model.py
python scripts/export_onnx.py
python scripts/validate_onnx.py
```

The generated model artifacts are stored under `models/` and are intentionally ignored by Git.

### 5. Run Image Inference

```bash
python -m scripts.infer tests/data/person.jpg
```

The command reports:

* Preprocessing latency
* ONNX Runtime latency
* Post-processing latency
* End-to-end latency
* Detected objects

An annotated image is written beside the input image.

### 6. Run Video Inference

```bash
python -m scripts.infer_video path/to/input.mp4
```

The video pipeline reports:

* Average inference time
* Processing FPS
* Input video FPS
* Number of processed frames

An annotated MP4 is written beside the input video.

### 7. Run the Python Benchmark

```bash
python -m benchmarks.benchmark_inference
```

The benchmark performs warm-up runs and reports:

* Mean latency
* P50 latency
* P95 latency
* P99 latency
* Throughput

Measurements are reported separately for preprocessing, ONNX Runtime, post-processing, and end-to-end execution.

## C++ Build

The C++ implementation uses:

* C++20
* CMake
* Ninja
* OpenCV
* ONNX Runtime
* vcpkg

Initialize the vcpkg submodule:

```bash
git submodule update --init --recursive
```

Configure and build the Linux Release preset:

```bash
cmake --preset linux-release
cmake --build --preset linux-release
```

For a debug build:

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
```

## Inference Flow

```text
Input Image
     ↓
RGB Conversion
     ↓
Aspect-Ratio-Preserving Resize
     ↓
Letterbox to 640 × 640
     ↓
HWC → CHW
     ↓
Add Batch Dimension
     ↓
FP32 Normalization [0, 1]
     ↓
NCHW Tensor [1, 3, 640, 640]
     ↓
ONNX Runtime
     ↓
Raw YOLOv8 Output [1, 84, 8400]
     ↓
Confidence Filtering
     ↓
XYWH → XYXY
     ↓
Map Coordinates Back to Source Image
     ↓
Non-Maximum Suppression
     ↓
Detections
```

The Python detector creates the ONNX Runtime session once and reuses it across inference calls.

The pipeline can be configured with CPU and CUDA execution providers. The providers actually available depend on the installed ONNX Runtime package and the local hardware/software environment.

## Performance Engineering Roadmap

The project will progressively explore:

```text
Python + ONNX Runtime
        ↓
C++ + ONNX Runtime
        ↓
CUDA Execution Provider
        ↓
TensorRT
        ↓
FP16 / INT8
        ↓
Memory Reuse / Pooling
        ↓
Concurrent Execution
        ↓
CUDA Streams / CUDA Graphs
```

Future optimization areas include:

* Memory pooling
* Tensor reuse
* Lock-free queues
* Concurrent inference
* Dynamic batching
* CUDA streams
* CUDA Graphs
* FP16 / INT8 inference
* CPU/GPU data-transfer optimization
* Profiling with Nsight and other performance tools

Every optimization should be supported by benchmark and profiling data rather than assumptions.

## Documentation

* [Setup](docs/setup.md)
* [Architecture](docs/architecture.md)
* [Model](docs/model.md)
* [Inference](docs/inference.md)
* [Benchmarking](docs/benchmarking.md)
* [API](docs/api.md)

## Project Goal

AI Inference Lab is a hands-on laboratory for understanding the complete production inference path:

```text
Application
     ↓
Request Scheduling
     ↓
Memory Management
     ↓
Preprocessing
     ↓
Inference Runtime
     ↓
CPU / GPU Execution
     ↓
Post-Processing
     ↓
Response
```

The long-term objective is to understand how production inference systems trade off:

* Latency
* Throughput
* Memory usage
* Concurrency
* CPU/GPU utilization
* Implementation complexity

using a reproducible workload and measurable performance data.

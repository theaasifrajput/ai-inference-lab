# Setup

This guide explains how to set up AI Inference Lab, prepare the YOLOv8n model, run Python inference, and build the C++ implementation.

---

## Prerequisites

### Python

- Python 3.10+
- pip
- Git
- Virtual environment support

### C++

- C++20-compatible compiler
- CMake 3.20+
- Ninja
- Git
- OpenCV
- ONNX Runtime
- vcpkg

The repository includes vcpkg as a Git submodule.

---

# 1. Clone the Repository

```bash
git clone https://github.com/theaasifrajput/ai-inference-lab.git
cd ai-inference-lab

Initialize the vcpkg submodule:

git submodule update --init --recursive
2. Python Environment

Create a virtual environment:

python3 -m venv .venv

Activate it:

Linux / WSL
source .venv/bin/activate
Windows
.venv\Scripts\activate

Upgrade pip:

python -m pip install --upgrade pip
3. Install Python Dependencies

Move into the Python project:

cd python

Install dependencies:

python -m pip install -r requirements.txt

The inference pipeline uses packages including:

NumPy
Pillow
OpenCV
ONNX Runtime
4. Prepare the YOLOv8 Model

The project currently uses YOLOv8n as the reference model.

The model preparation pipeline is:

YOLOv8
   ↓
PyTorch Model
   ↓
YOLOv8n .pt
   ↓
ONNX Export
   ↓
YOLOv8n .onnx
   ↓
ONNX Runtime
Download the Model

From the python/ directory:

python scripts/download_model.py

This creates:

models/yolov8n.pt
5. Export to ONNX

Run:

python scripts/export_onnx.py

This creates:

models/yolov8n.onnx

The ONNX model is the primary model artifact used by the inference pipeline.

6. Validate the ONNX Model

Run:

python scripts/validate_onnx.py

Validation checks:

ONNX model structure
ONNX Runtime compatibility
Model input information
Model output information

The expected input is:

Name:   images
Shape:  [1, 3, 640, 640]
Type:   float32
Layout: NCHW

The output is:

Name:   output0
Shape:  [1, 84, 8400]
Type:   float32

For the COCO-trained YOLOv8 model:

84 = 4 bounding-box values + 80 class scores
7. Run Image Inference

The repository contains a sample image under:

python/tests/data/person.jpg

Run:

python -m scripts.infer tests/data/person.jpg

The pipeline performs:

Input Image
     ↓
RGB Conversion
     ↓
Resize
     ↓
Letterbox
     ↓
HWC → CHW
     ↓
NCHW
     ↓
Normalization
     ↓
ONNX Runtime
     ↓
Post-processing
     ↓
Detections

The inference command reports latency for each major stage.

8. Run Video Inference

The video inference script accepts a video path:

python -m scripts.infer_video path/to/input.mp4

The pipeline processes each frame independently.

The output video is written as:

input_detected.mp4

The script reports:

Processed frames
Total processing time
Average inference latency
Processing FPS
Input video FPS
9. Run the Benchmark

Run:

python -m benchmarks.benchmark_inference

The benchmark first performs warm-up iterations.

It then measures:

Preprocessing
ONNX Runtime
Post-processing
End-to-end

For each stage, the benchmark reports:

Mean
P50
P95
P99

It also reports estimated throughput:

images/sec

This benchmark should be used as the baseline before making performance optimizations.

10. CPU vs CUDA Execution Provider

The inference pipeline can be configured with ONNX Runtime execution providers.

Example:

providers=[
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
]

The actual available providers depend on:

Installed ONNX Runtime package
NVIDIA driver
CUDA installation
GPU availability
CUDA/cuDNN compatibility

Check available providers with:

import onnxruntime as ort

print(ort.get_available_providers())

For example:

[
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

The exact list will vary by environment.

11. C++ Build

Return to the repository root:

cd ..

Verify the vcpkg submodule:

git submodule update --init --recursive

The project uses CMake presets.

Available presets:

linux-debug
linux-release
Release Build

Configure:

cmake --preset linux-release

Build:

cmake --build --preset linux-release
Debug Build

Configure:

cmake --preset linux-debug

Build:

cmake --build --preset linux-debug
12. C++ Dependencies

The C++ project currently depends on:

OpenCV
ONNX Runtime

These dependencies are declared through:

vcpkg.json

The project is compiled using:

C++20
13. Build Directory

CMake generates build artifacts under:

cpp/build/

The repository ignores build artifacts through .gitignore.

Typical structure:

cpp/
└── build/
    ├── linux-debug/
    └── linux-release/
14. Model Artifacts

Model files are intentionally not committed to Git.

Ignored model formats include:

*.pt
*.onnx
*.bin

Therefore a fresh clone requires the model preparation steps before inference can run.

15. Recommended Development Workflow

For new inference optimizations, use the following workflow:

1. Establish baseline
        ↓
2. Measure latency
        ↓
3. Profile
        ↓
4. Identify bottleneck
        ↓
5. Implement optimization
        ↓
6. Benchmark again
        ↓
7. Compare results

Avoid optimizing based only on theoretical performance.

The benchmark results should demonstrate whether an optimization actually improves:

Latency
Throughput
CPU utilization
GPU utilization
Memory usage
16. Troubleshooting
ONNX Runtime Cannot Load Model

Verify that the model exists:

ls models/

Expected:

yolov8n.pt
yolov8n.onnx
CUDA Provider Is Missing

Check:

import onnxruntime as ort

print(ort.get_available_providers())

If only:

CPUExecutionProvider

is available, the installed ONNX Runtime package does not currently provide CUDA execution.

CMake Cannot Find ONNX Runtime

Make sure the vcpkg submodule has been initialized:

git submodule update --init --recursive

Then configure again:

cmake --preset linux-release

Build and Run
cmake --preset linux-release
cmake --build --preset linux-release

./cpp/build/linux-release/ai_inference \
    tests/data/test.mp4 \
    output/cpp_detected.mp4
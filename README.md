# AI Inference Lab

Production-oriented AI inference experiments focused on
latency, throughput, C++, CUDA and inference runtimes.

## Quick Start

### 1. Clone

git clone ...
cd ai-inference-lab

### 2. Create virtual environment

...

### 3. Install dependencies

python -m pip install -r services/requirements.txt

### 4. Download model

python scripts/download_model.py

### 5. Export ONNX

python scripts/export_onnx.py

### 6. Validate ONNX

python scripts/validate_onnx.py

## Documentation

- [Architecture](docs/architecture.md)
- [Model](docs/model.md)
- [Inference](docs/inference.md)
- [API](docs/api.md)
- [Benchmarking](docs/benchmarking.md)
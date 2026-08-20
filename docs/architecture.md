# Architecture

AI Inference Lab is an inference-engineering project focused on building, measuring, and optimizing production-oriented AI inference systems.

The repository contains separate Python and C++ implementations so that different inference approaches can be developed and benchmarked using the same models and workloads.

## High-Level Architecture

```text
                         AI Inference Lab
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
          Python Inference              C++ Inference
                 │                             │
          ONNX Runtime                 ONNX Runtime
                 │                             │
                CPU                       CPU / GPU
                                               │
                                            CUDA
                                               │
                                          TensorRT


Python Inference

The Python implementation provides the initial reference inference pipeline.

Client
  ↓
FastAPI Gateway
  ↓
Bounded Request Queue
  ↓
Inference Worker
  ↓
YOLOv8 ONNX
  ↓
ONNX Runtime
  ↓
CPU
  ↓
Post-processing
  ↓
Result

The Python implementation is responsible for establishing:

Correct model execution
Correct preprocessing
Correct post-processing
API behavior
Queue behavior
Performance measurements
Components
Gateway

The gateway handles:

HTTP requests
Image validation
Request IDs
Queue admission
Result retrieval
Health checks
Readiness checks
Metrics

Inference Engine

The inference engine handles:

Image preprocessing
Tensor creation
Model execution
Raw model output
Post-processing
Queue

The request queue provides bounded buffering between incoming requests and inference workers.

This prevents unlimited requests from accumulating in memory.

Benchmarking

The benchmarking system measures:

Latency
Throughput
Queue behavior
Concurrency
Runtime performance

Python and C++ Separation

Python and C++ implementations are intentionally kept separate.

python/
    ↓
Python inference implementation


cpp/
    ↓
C++ inference implementation

Both implementations should use the same:

Model
Input format
Preprocessing behavior
Output interpretation
Detection format
Benchmark methodology

This allows correctness and performance comparisons between the two implementations.

Evolution of the Inference Engine

The project will progressively explore different inference implementations:

Python
  ↓
C++ + ONNX Runtime
  ↓
C++ + ONNX Runtime + CUDA
  ↓
C++ + TensorRT
  ↓
TensorRT FP16 / INT8

The system can then be extended with inference-engineering techniques such as:

Memory pooling
      ↓
Concurrent execution
      ↓
Dynamic batching
      ↓
CUDA streams
      ↓
CUDA Graphs

Design Goal

The long-term goal is to understand how production inference systems are built and optimized across the complete execution path:

Application
    ↓
Request scheduling
    ↓
Memory management
    ↓
Preprocessing
    ↓
Inference runtime
    ↓
GPU execution
    ↓
Post-processing
    ↓
Response

Performance improvements should be supported by benchmarks and profiling rather than assumptions.


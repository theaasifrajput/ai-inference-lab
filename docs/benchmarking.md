# Benchmarking

AI inference performance should be measured using consistent workloads and clearly defined metrics.

The goal is to understand where time is spent and quantify the impact of each optimization.

## Metrics

The inference pipeline should measure the following:

| Metric | Description |
|---|---|
| Preprocessing latency | Time spent preparing the input tensor |
| Model execution latency | Time spent executing the ONNX model |
| Post-processing latency | Time spent processing model output |
| Queue latency | Time spent waiting before inference |
| End-to-end latency | Total request processing time |
| Throughput | Number of requests processed per second |

Latency should be reported using:

```text
p50
p95
p99

Why Percentiles Matter

Average latency can hide slow requests.

For example:

Average: 20 ms
p50:      18 ms
p95:      31 ms
p99:      75 ms

This tells us that most requests are fast, but a small number experience significantly higher latency.

For an inference service, tail latency is therefore important.

Benchmark Pipeline

The benchmark should measure the complete pipeline:
Request
   ↓
Queue
   ↓
Preprocessing
   ↓
Model Execution
   ↓
Post-processing
   ↓
Response

Each stage should be measured independently where possible.

Benchmark Parameters

The benchmark should allow us to vary:

Number of requests
Concurrency
Worker count
Input image
Model
Execution provider

For example:

workers = 1
workers = 2
workers = 4
workers = 8

This allows us to understand how the system behaves as concurrency increases.
Initial Baseline

The initial inference configuration is:

Model:               YOLOv8n
Input:               640 × 640
Precision:           FP32
Runtime:             ONNX Runtime
Execution Provider:  CPUExecutionProvider

Performance numbers should be measured on the actual machine running the benchmark.

They should not be hard-coded into the documentation.

Benchmark Results

Results should eventually be recorded in a table such as:

Configuration	Throughput	p50	p95	p99
1 worker	-	-	-	-
2 workers	-	-	-	-
4 workers	-	-	-	-
8 workers	-	-	-	-

Actual values will be added after the benchmark implementation is complete.

Future Comparisons

The benchmark framework should eventually allow us to compare:

Python + ONNX Runtime CPU
        ↓
C++ + ONNX Runtime CPU
        ↓
C++ + ONNX Runtime CUDA
        ↓
C++ + TensorRT
        ↓
C++ + TensorRT FP16

The same model, input data, and measurement methodology should be used for each implementation.

This allows performance improvements to be attributed to specific runtime or system-level changes.


## Current Baseline

The initial Python inference baseline was measured using:

- Model: YOLOv8n
- Input: 640 × 640
- Precision: FP32
- Runtime: ONNX Runtime
- Execution Provider: CPUExecutionProvider
- Warmup iterations: 10
- Benchmark iterations: 100

Results:

| Stage           | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|-----------------|-----------|----------|----------|----------|
| Preprocessing   | 19.313    | 18.022   | 28.928   | 30.389   |
| ONNX Runtime    | 56.998    | 51.412   | 80.254   | 104.599  |
| Post-processing | 0.272     | 0.235    | 0.408    | 0.774    |
| End-to-end      | 76.598    | 70.908   | 112.862  | 134.737  |

Throughput:

```text
13.06 images/sec
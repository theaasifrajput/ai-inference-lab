# Model

AI Inference Lab currently uses **YOLOv8n** as its reference computer-vision model.

The model is exported from PyTorch to ONNX and executed using **ONNX Runtime**.

The purpose of using YOLOv8n is not only object detection. It provides a reproducible workload for studying the complete inference stack:

```text
Model
  ↓
Preprocessing
  ↓
Tensor creation
  ↓
ONNX Runtime
  ↓
CPU / GPU execution
  ↓
Raw model output
  ↓
Post-processing
  ↓
Application detections
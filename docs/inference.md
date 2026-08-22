# Inference

This document describes how an input image moves through the AI Inference Lab pipeline, from application input to final detections.

The current reference implementation uses:

- Python
- ONNX Runtime
- YOLOv8n
- NumPy
- Pillow
- OpenCV

The same inference flow is being implemented in C++ for performance-oriented experiments.

---

# 1. High-Level Pipeline

```text
Input Image
     ↓
Preprocessing
     ↓
Tensor Creation
     ↓
ONNX Runtime
     ↓
Model Execution
     ↓
Raw Output Tensor
     ↓
Post-processing
     ↓
Detections


A more detailed representation is:


                    Input Image
                         │
                         ▼
                RGB Conversion
                         │
                         ▼
             Aspect-Ratio Resize
                         │
                         ▼
                 Letterboxing
                         │
                         ▼
                    HWC → CHW
                         │
                         ▼
                Add Batch Dimension
                         │
                         ▼
                   Normalize
                         │
                         ▼
              [1, 3, 640, 640]
                         │
                         ▼
                 ONNX Runtime
                         │
                  ┌──────┴──────┐
                  │             │
                 CPU           CUDA
                  │             │
                  └──────┬──────┘
                         │
                         ▼
                [1, 84, 8400]
                         │
                         ▼
              Confidence Filtering
                         │
                         ▼
                   XYWH → XYXY
                         │
                         ▼
              Restore Image Coordinates
                         │
                         ▼
                        NMS
                         │
                         ▼
                    Detections
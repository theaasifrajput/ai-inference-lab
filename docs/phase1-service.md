# Phase 1 service

The request path is browser → FastAPI → bounded `asyncio.Queue` → one ONNX Runtime CPU worker. Image preprocessing is RGB decode, aspect-ratio-preserving letterbox, normalization, HWC-to-CHW conversion, and contiguous FP32 tensor construction. YOLOv8 output is filtered to COCO class zero (`person`) and run through NMS before original-image coordinates are returned.

The local queue is intentionally bounded for a correct, easily testable CPU-first vertical slice. A production Phase 4 scheduler can replace the dispatcher with Redis Streams without changing the HTTP submit/status contract. GPU execution will add a provider selection policy (TensorRT → ORT CUDA → ORT CPU), pinned host buffers only where measurements justify them, and batching bounded by maximum batch size and wait time.

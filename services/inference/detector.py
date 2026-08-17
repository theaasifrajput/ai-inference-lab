"""YOLOv8 ONNX CPU execution and person-only postprocessing."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: list[float]


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - resized.width) // 2, (size - resized.height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), scale, (pad_x, pad_y)


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(box[0], others[:, 0]), np.maximum(box[1], others[:, 1])
    x2, y2 = np.minimum(box[2], others[:, 2]), np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]) - intersection
    return intersection / np.maximum(union, 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    keep, order = [], scores.argsort()[::-1]
    while len(order):
        current = order[0]
        keep.append(int(current))
        order = order[1:]
        if len(order):
            order = order[_iou(boxes[current], boxes[order]) <= threshold]
    return keep


class YoloOnnxDetector:
    def __init__(self, model_path: str, size: int, confidence: float, nms_threshold: float):
        self.model_path, self.size = Path(model_path), size
        self.confidence, self.nms_threshold, self.session = confidence, nms_threshold, None

    def detect(self, image: Image.Image) -> tuple[list[Detection], float]:
        if self.session is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Model missing: {self.model_path}. See scripts/export_onnx.py.")
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        width, height = image.size
        tensor, scale, (pad_x, pad_y) = letterbox(image, self.size)
        started = time.perf_counter()
        output = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        prediction = np.squeeze(output).T
        prediction = prediction[prediction[:, 4] >= self.confidence]  # COCO class 0: person.
        if not len(prediction):
            return [], elapsed_ms
        xywh, scores = prediction[:, :4], prediction[:, 4]
        boxes = np.column_stack((xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2))
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / scale).clip(0, width)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / scale).clip(0, height)
        return [Detection("person", float(scores[i]), [round(float(v), 1) for v in boxes[i]]) for i in nms(boxes, scores, self.nms_threshold)], elapsed_ms

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from services.inference.detector import YoloOnnxDetector
from services.inference.models import Detection
from services.inference.postprocessing import postprocess
from services.inference.preprocessing import preprocess


class InferencePipeline:
    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
    ) -> None:
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.detector = YoloOnnxDetector(model_path)

    def infer(
        self,
        image: Image.Image,
    ) -> tuple[list[Detection], dict[str, float]]:
        total_started = time.perf_counter()

        original_width, original_height = image.size

        # -------------------------
        # Preprocessing
        # -------------------------
        started = time.perf_counter()

        preprocessing_result = preprocess(
            image,
            self.input_size,
        )

        preprocess_ms = (
            time.perf_counter() - started
        ) * 1000

        # -------------------------
        # Model execution
        # -------------------------
        output, inference_ms = self.detector.infer(
            preprocessing_result.tensor
        )

        # -------------------------
        # Post-processing
        # -------------------------
        started = time.perf_counter()

        detections = postprocess(
            output=output,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            scale=preprocessing_result.scale,
            pad_x=preprocessing_result.pad_x,
            pad_y=preprocessing_result.pad_y,
            original_width=original_width,
            original_height=original_height,
        )

        postprocess_ms = (
            time.perf_counter() - started
        ) * 1000

        total_ms = (
            time.perf_counter() - total_started
        ) * 1000

        metrics = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
        }

        return detections, metrics
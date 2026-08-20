from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.providers = providers or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(
        self,
        tensor: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Execute the ONNX model.

        Args:
            tensor: NCHW FP32 input tensor.

        Returns:
            output: Raw model output.
            elapsed_ms: Model execution time in milliseconds.
        """

        started = time.perf_counter()

        output = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )[0]

        elapsed_ms = (time.perf_counter() - started) * 1000

        return output, elapsed_ms
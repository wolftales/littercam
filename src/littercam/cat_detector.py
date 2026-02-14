"""TFLite-based cat detection for LitterCam."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO class ID for "cat"
CAT_CLASS_ID = 17


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # ymin, xmin, ymax, xmax


class CatDetector:
    """Detect cats using a quantized MobileNet SSD v2 COCO TFLite model."""

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        confidence_threshold: float = 0.5,
        enabled: bool = True,
    ) -> None:
        self._threshold = confidence_threshold
        self._enabled = enabled
        self._interpreter: Optional[object] = None
        self._labels: dict[int, str] = {}
        self._input_size = (300, 300)

        if not enabled:
            logger.info("Cat detection disabled by config")
            return

        model_file = Path(model_path)
        labels_file = Path(labels_path)

        if not model_file.exists():
            logger.warning("Model file not found: %s — cat detection disabled", model_path)
            self._enabled = False
            return

        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                logger.warning("tflite_runtime not installed — cat detection disabled")
                self._enabled = False
                return

        self._labels = self._load_labels(labels_file)
        self._interpreter = tflite.Interpreter(
            model_path=str(model_file), num_threads=4
        )
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        logger.info("Cat detector loaded: %s", model_path)

    @staticmethod
    def _load_labels(path: Path) -> dict[int, str]:
        labels: dict[int, str] = {}
        if not path.exists():
            logger.warning("Labels file not found: %s", path)
            return labels
        for line in path.read_text().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1]
        return labels

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a frame. Returns only cat detections above threshold."""
        if not self._enabled or self._interpreter is None:
            return []

        # Resize to model input size
        from PIL import Image

        img = Image.fromarray(frame).resize(self._input_size)
        input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        # SSD MobileNet v2 outputs: boxes, classes, scores, count
        boxes = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
        scores = self._interpreter.get_tensor(self._output_details[2]["index"])[0]

        detections: list[Detection] = []
        for i, score in enumerate(scores):
            class_id = int(classes[i])
            if class_id == CAT_CLASS_ID and score >= self._threshold:
                label = self._labels.get(class_id, "cat")
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=label,
                        confidence=float(score),
                        bbox=tuple(boxes[i]),
                    )
                )
        return detections

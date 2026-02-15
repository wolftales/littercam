"""ONNX Runtime-based cat detection for LitterCam."""
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
    """Detect cats using SSD MobileNet v1 COCO via ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        confidence_threshold: float = 0.5,
        enabled: bool = True,
    ) -> None:
        self._threshold = confidence_threshold
        self._enabled = enabled
        self._session = None
        self._labels: dict[int, str] = {}
        self._input_name: Optional[str] = None
        self._input_shape: Optional[tuple] = None

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
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed — cat detection disabled")
            self._enabled = False
            return

        self._labels = self._load_labels(labels_file)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self._session = ort.InferenceSession(
            str(model_file),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        # ONNX may report dynamic dims as strings; default to 300x300
        raw_shape = inp.shape
        h = raw_shape[1] if isinstance(raw_shape[1], int) else 300
        w = raw_shape[2] if isinstance(raw_shape[2], int) else 300
        self._input_shape = (h, w)
        logger.info("Cat detector loaded: %s (input %s)", model_path, self._input_shape)

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
        if not self._enabled or self._session is None:
            return []

        from PIL import Image

        h, w = self._input_shape
        # Handle YUV420 (2D) frames — use Y plane as grayscale, convert to RGB
        if frame.ndim == 2:
            # Y plane is the top portion (full height of the requested resolution)
            y_height = int(frame.shape[0] / 1.5)
            y_plane = frame[:y_height, :frame.shape[1]]
            img = Image.fromarray(y_plane, mode="L").convert("RGB").resize((w, h))
        else:
            img = Image.fromarray(frame).convert("RGB").resize((w, h))
        input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

        outputs = self._session.run(None, {self._input_name: input_data})

        # SSD MobileNet v1 ONNX outputs: boxes, classes, scores, num_detections
        boxes = outputs[0][0]
        classes = outputs[1][0]
        scores = outputs[2][0]

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

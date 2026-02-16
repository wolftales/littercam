"""YOLOv8n-based cat detection for LitterCam."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO 80-class index for "cat" (0-indexed, no background class)
CAT_CLASS_ID = 15


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized 0-1)


class CatDetector:
    """Detect cats using YOLOv8n via ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        confidence_threshold: float = 0.3,
        enabled: bool = True,
    ) -> None:
        self._threshold = confidence_threshold
        self._enabled = enabled
        self._session = None
        self._labels: dict[int, str] = {}
        self._input_name: Optional[str] = None
        self._input_hw: tuple[int, int] = (640, 640)

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
        # YOLOv8 input: [1, 3, H, W]
        raw_shape = inp.shape
        h = raw_shape[2] if isinstance(raw_shape[2], int) else 640
        w = raw_shape[3] if isinstance(raw_shape[3], int) else 640
        self._input_hw = (h, w)
        logger.info("Cat detector loaded: %s (input %dx%d)", model_path, h, w)

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

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
        """Simple non-maximum suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a frame. Returns only cat detections above threshold."""
        if not self._enabled or self._session is None:
            return []

        from PIL import Image

        h, w = self._input_hw

        # Handle YUV420 (2D) frames — use Y plane as grayscale, convert to RGB
        if frame.ndim == 2:
            y_height = int(frame.shape[0] / 1.5)
            y_plane = frame[:y_height, :frame.shape[1]]
            img = Image.fromarray(y_plane, mode="L").convert("RGB").resize((w, h))
        else:
            img = Image.fromarray(frame).convert("RGB").resize((w, h))

        # YOLOv8 input: [1, 3, H, W], float32, 0-1 range
        img_array = np.array(img, dtype=np.float32) / 255.0
        input_data = img_array.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> NCHW

        outputs = self._session.run(None, {self._input_name: input_data})

        # YOLOv8 output: [1, 84, 8400] -> transpose to [8400, 84]
        raw = outputs[0][0].T  # [8400, 84]
        # First 4: cx, cy, w, h; remaining 80: class scores
        cx, cy, bw, bh = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        class_scores = raw[:, 4:]  # [8400, 80]

        # Get best class per detection
        class_ids = class_scores.argmax(axis=1)
        max_scores = class_scores.max(axis=1)

        # Filter by confidence
        mask = max_scores >= self._threshold
        if not mask.any():
            return []

        filtered_cx = cx[mask]
        filtered_cy = cy[mask]
        filtered_bw = bw[mask]
        filtered_bh = bh[mask]
        filtered_ids = class_ids[mask]
        filtered_scores = max_scores[mask]

        # Convert cx,cy,w,h to x1,y1,x2,y2 (pixel coords)
        x1 = filtered_cx - filtered_bw / 2
        y1 = filtered_cy - filtered_bh / 2
        x2 = filtered_cx + filtered_bw / 2
        y2 = filtered_cy + filtered_bh / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep = self._nms(boxes, filtered_scores)

        # Log all detections for debugging
        detections: list[Detection] = []
        for idx in keep:
            class_id = int(filtered_ids[idx])
            score = float(filtered_scores[idx])
            label = self._labels.get(class_id, f"class_{class_id}")
            if score >= 0.2:
                logger.debug("Detection: %s (id=%d) %.1f%%", label, class_id, score * 100)
            if class_id == CAT_CLASS_ID:
                # Normalize bbox to 0-1 range
                bbox = (
                    float(boxes[idx, 0]) / w,
                    float(boxes[idx, 1]) / h,
                    float(boxes[idx, 2]) / w,
                    float(boxes[idx, 3]) / h,
                )
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=label,
                        confidence=score,
                        bbox=bbox,
                    )
                )
        return detections

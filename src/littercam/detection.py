"""Frame-based motion detection (camera-independent)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionConfig:
    threshold: float
    downscale_width: int
    downscale_height: int
    trigger_frames: int


class MotionDetector:
    """Motion detection using frame differencing on externally-provided frames."""

    def __init__(self, config: MotionConfig) -> None:
        self._config = config
        self._previous: Optional[np.ndarray] = None
        self._trigger_count = 0

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            # YUV420: full frame is 1.5x height (Y + UV planes).
            # Extract just the Y (luminance) plane and crop padding.
            y_height = self._config.downscale_height
            y_width = self._config.downscale_width
            return frame[:y_height, :y_width].astype("float32")
        return frame[:, :, 0].astype("float32")

    def _frame_diff(self, frame: np.ndarray) -> float:
        gray = self._to_gray(frame)
        if self._previous is None:
            self._previous = gray
            return 0.0
        diff = np.abs(gray - self._previous)
        self._previous = gray
        return float(np.mean(diff))

    def analyze(self, frame: np.ndarray) -> Optional[float]:
        """Analyze a frame for motion. Returns trigger score if threshold met, else None."""
        score = self._frame_diff(frame)
        if score >= self._config.threshold:
            self._trigger_count += 1
            logger.debug(
                "Motion score %s (%s/%s)",
                score,
                self._trigger_count,
                self._config.trigger_frames,
            )
        else:
            self._trigger_count = 0
        if self._trigger_count >= self._config.trigger_frames:
            self._trigger_count = 0
            return score
        return None

    def current_score(self, frame: np.ndarray) -> float:
        """Check motion score for a frame without affecting trigger count."""
        gray = self._to_gray(frame)
        if self._previous is None:
            return 0.0
        diff = np.abs(gray - self._previous)
        return float(np.mean(diff))

    def reset(self) -> None:
        """Clear state between sessions."""
        self._previous = None
        self._trigger_count = 0

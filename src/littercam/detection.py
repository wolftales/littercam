"""Camera-based motion detection using libcamera via Picamera2."""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

Picamera2 = None
if importlib.util.find_spec("picamera2"):
    Picamera2 = importlib.import_module("picamera2").Picamera2


@dataclass
class MotionConfig:
    threshold: float
    downscale_width: int
    downscale_height: int
    trigger_frames: int


class MotionDetector:
    """Motion detection using frame differencing."""

    def __init__(self, config: MotionConfig) -> None:
        if Picamera2 is None:
            raise RuntimeError("Picamera2 is required for motion detection")
        self._config = config
        self._camera = Picamera2()
        self._camera.configure(
            self._camera.create_preview_configuration(
                main={"size": (config.downscale_width, config.downscale_height)}
            )
        )
        self._camera.start()
        time.sleep(1.0)
        self._previous: Optional[np.ndarray] = None
        self._trigger_count = 0

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()

    def _frame_diff(self, frame: np.ndarray) -> float:
        gray = frame[:, :, 0].astype("float32")
        if self._previous is None:
            self._previous = gray
            return 0.0
        diff = np.abs(gray - self._previous)
        self._previous = gray
        return float(np.mean(diff))

    def poll(self) -> Optional[float]:
        frame = self._camera.capture_array()
        score = self._frame_diff(frame)
        if score >= self._config.threshold:
            self._trigger_count += 1
            logger.debug("Motion score %s (%s/%s)", score, self._trigger_count, self._config.trigger_frames)
        else:
            self._trigger_count = 0
        if self._trigger_count >= self._config.trigger_frames:
            self._trigger_count = 0
            return score
        return None


def motion_loop(detector: MotionDetector, poll_interval: float = 0.25) -> Iterable[float]:
    """Yield motion scores when a trigger fires."""
    try:
        while True:
            score = detector.poll()
            if score is not None:
                yield score
            time.sleep(poll_interval)
    finally:
        detector.close()

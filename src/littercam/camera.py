"""Persistent dual-stream camera manager for LitterCam."""
from __future__ import annotations

import importlib.util
import io
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

Transform = None
if importlib.util.find_spec("libcamera"):
    Transform = importlib.import_module("libcamera").Transform

logger = logging.getLogger(__name__)

Picamera2 = None
if importlib.util.find_spec("picamera2"):
    Picamera2 = importlib.import_module("picamera2").Picamera2


@dataclass
class BufferedFrame:
    timestamp: float
    jpeg_bytes: bytes


class CameraManager:
    """Single persistent Picamera2 instance with dual streams and pre-roll buffer."""

    def __init__(
        self,
        main_width: int = 1920,
        main_height: int = 1080,
        lores_width: int = 320,
        lores_height: int = 240,
        jpeg_quality: int = 85,
        pre_roll_seconds: float = 5.0,
        capture_interval: float = 0.25,
    ) -> None:
        if Picamera2 is None:
            raise RuntimeError("Picamera2 is required for camera access")
        self._main_size = (main_width, main_height)
        self._lores_size = (lores_width, lores_height)
        self._jpeg_quality = jpeg_quality
        self._camera: Optional[object] = None
        maxlen = max(1, int(pre_roll_seconds / capture_interval))
        self._buffer: deque[BufferedFrame] = deque(maxlen=maxlen)

    def start(self) -> None:
        """Start the camera with dual-stream configuration."""
        self._camera = Picamera2()
        transform = Transform(vflip=True, hflip=True) if Transform else None
        config = self._camera.create_preview_configuration(
            main={"size": self._main_size},
            lores={"size": self._lores_size},
            transform=transform,
        )
        self._camera.configure(config)
        self._camera.start()
        time.sleep(1.0)
        logger.info(
            "Camera started: main=%s, lores=%s",
            self._main_size,
            self._lores_size,
        )

    def stop(self) -> None:
        """Stop and close the camera."""
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None
            logger.info("Camera stopped")

    def capture_lores(self) -> np.ndarray:
        """Capture a low-resolution frame for motion/cat detection."""
        return self._camera.capture_array("lores")

    def capture_main_jpeg(self) -> bytes:
        """Capture a high-resolution JPEG from the main stream."""
        frame = self._camera.capture_array("main")
        img = Image.fromarray(frame).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._jpeg_quality)
        return buf.getvalue()

    def buffer_frame(self) -> None:
        """Capture a main frame and push it to the pre-roll buffer."""
        jpeg = self.capture_main_jpeg()
        self._buffer.append(BufferedFrame(timestamp=time.time(), jpeg_bytes=jpeg))

    def drain_buffer(self) -> list[BufferedFrame]:
        """Return and clear the pre-roll buffer contents."""
        frames = list(self._buffer)
        self._buffer.clear()
        return frames

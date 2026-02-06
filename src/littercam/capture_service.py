"""Capture service for LitterCam."""
from __future__ import annotations

import importlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from littercam.config import AppConfig
from littercam.detection import MotionConfig, MotionDetector, motion_loop
from littercam.events import EventMeta, event_dir_for, write_meta
from littercam.logging import configure_logging

logger = logging.getLogger(__name__)

Picamera2 = None
if importlib.util.find_spec("picamera2"):
    Picamera2 = importlib.import_module("picamera2").Picamera2


class CaptureService:
    """Monitor for motion and capture event bursts."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._output_root = config.capture.output_root
        self._cooldown_seconds = config.capture.cooldown_seconds
        self._capture_seconds = config.capture.capture_seconds
        self._capture_interval = config.capture.capture_interval_seconds
        self._motion_config = MotionConfig(
            threshold=config.capture.motion_threshold,
            downscale_width=config.capture.downscale_width,
            downscale_height=config.capture.downscale_height,
            trigger_frames=config.capture.trigger_frames,
        )

    def _capture_burst(self, event_path: Path) -> int:
        if Picamera2 is None:
            raise RuntimeError("Picamera2 is required for capture")
        camera = Picamera2()
        camera.configure(camera.create_still_configuration())
        camera.start()
        time.sleep(0.5)
        image_count = 0
        start_time = time.time()
        try:
            while time.time() - start_time < self._capture_seconds:
                image_path = event_path / f"img-{image_count:03d}.jpg"
                camera.capture_file(str(image_path))
                thumbnail_path = event_path / f"thumb-{image_count:03d}.jpg"
                self._create_thumbnail(image_path, thumbnail_path)
                image_count += 1
                time.sleep(self._capture_interval)
        finally:
            camera.stop()
            camera.close()
        return image_count

    @staticmethod
    def _create_thumbnail(image_path: Path, thumbnail_path: Path) -> None:
        with Image.open(image_path) as image:
            image.thumbnail((320, 240))
            image.save(thumbnail_path, "JPEG")

    def run(self) -> None:
        configure_logging(self._config.logging)
        self._output_root.mkdir(parents=True, exist_ok=True)
        detector = MotionDetector(self._motion_config)
        logger.info("Starting motion detection")

        for score in motion_loop(detector):
            start_ts = datetime.now(timezone.utc)
            event_path = event_dir_for(start_ts, self._output_root)
            event_path.mkdir(parents=True, exist_ok=True)
            logger.info("Motion detected (score %.2f). Capturing burst.", score)
            image_count = self._capture_burst(event_path)
            end_ts = datetime.now(timezone.utc)
            meta = EventMeta(
                start_ts=start_ts.isoformat(),
                end_ts=end_ts.isoformat(),
                trigger_score=score,
                image_count=image_count,
                cat_tag=None,
            )
            write_meta(event_path, meta)
            logger.info("Captured event %s with %s images", event_path.name, image_count)
            logger.info("Cooldown for %s seconds", self._cooldown_seconds)
            time.sleep(self._cooldown_seconds)

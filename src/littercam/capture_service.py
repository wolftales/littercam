"""Capture service for LitterCam — state machine with adaptive recording."""
from __future__ import annotations

import enum
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from littercam.camera import CameraManager
from littercam.cat_detector import CatDetector
from littercam.config import AppConfig
from littercam.detection import MotionConfig, MotionDetector
from littercam.events import EventMeta, event_dir_for, write_meta
from littercam.logging import configure_logging

logger = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    COOLDOWN = "cooldown"


class CaptureService:
    """Monitor for motion and capture event bursts with adaptive recording."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        cap = config.capture
        self._output_root = cap.output_root
        self._cooldown_seconds = cap.cooldown_seconds
        self._capture_interval = cap.capture_interval_seconds
        self._post_roll_seconds = cap.post_roll_seconds
        self._max_capture_seconds = cap.max_capture_seconds
        self._motion_threshold = cap.motion_threshold

        self._motion_config = MotionConfig(
            threshold=cap.motion_threshold,
            downscale_width=cap.downscale_width,
            downscale_height=cap.downscale_height,
            trigger_frames=cap.trigger_frames,
        )
        self._detector = MotionDetector(self._motion_config)

        self._camera = CameraManager(
            main_width=cap.main_width,
            main_height=cap.main_height,
            lores_width=cap.downscale_width,
            lores_height=cap.downscale_height,
            jpeg_quality=cap.jpeg_quality,
            pre_roll_seconds=cap.pre_roll_seconds,
            capture_interval=cap.capture_interval_seconds,
        )

        cat_cfg = config.cat_detection
        self._cat_detector = CatDetector(
            model_path=cat_cfg.model_path,
            labels_path=cat_cfg.labels_path,
            confidence_threshold=cat_cfg.confidence_threshold,
            enabled=cat_cfg.enabled,
        )

    @staticmethod
    def _create_thumbnail(image_path: Path, thumbnail_path: Path) -> None:
        with Image.open(image_path) as image:
            image.thumbnail((320, 240))
            image.save(thumbnail_path, "JPEG")

    def _save_jpeg(self, event_path: Path, index: int, jpeg_bytes: bytes) -> None:
        image_path = event_path / f"img-{index:03d}.jpg"
        image_path.write_bytes(jpeg_bytes)
        thumbnail_path = event_path / f"thumb-{index:03d}.jpg"
        self._create_thumbnail(image_path, thumbnail_path)

    def _find_presence_window(self, event_path: Path, threshold_mult: float = 2.0) -> tuple[int, int]:
        """Use baseline diff to find when something entered/left the scene.

        Returns (start_idx, end_idx) of the presence window, or (0, n-1) if
        no baseline is available.
        """
        thumbs = sorted(event_path.glob("thumb-*.jpg"))
        n = len(thumbs)
        if n == 0:
            return 0, 0

        baseline_path = self._output_root / "baseline.jpg"
        if not baseline_path.exists():
            return 0, n - 1

        # Compute diff of each thumbnail against baseline
        ref = Image.open(baseline_path).convert("L")
        diffs = []
        for thumb_path in thumbs:
            img = Image.open(thumb_path).convert("L").resize(ref.size)
            gray = np.array(img, dtype=np.float32)
            ref_gray = np.array(ref, dtype=np.float32)
            diffs.append(float(np.mean(np.abs(gray - ref_gray))))

        # Find frames significantly above the baseline noise
        if not diffs:
            return 0, n - 1
        median_diff = sorted(diffs)[len(diffs) // 2]
        presence_threshold = max(median_diff * threshold_mult, 3.0)

        first = None
        last = None
        for i, d in enumerate(diffs):
            if d >= presence_threshold:
                if first is None:
                    first = i
                last = i

        if first is None:
            return 0, n - 1

        # Add a small buffer (3 frames before/after)
        first = max(0, first - 3)
        last = min(n - 1, last + 3)
        return first, last

    def _scan_for_cats(self, event_path: Path) -> tuple[bool, float, int, int | None, int | None]:
        """Post-capture cat detection using baseline-guided presence window.

        Returns (cat_detected, max_confidence, detection_count, first_cat_frame, last_cat_frame).
        """
        max_confidence = 0.0
        detection_count = 0
        first_cat_frame = None
        last_cat_frame = None

        images = sorted(event_path.glob("img-*.jpg"))
        if not images:
            return False, 0.0, 0, None, None

        # Use baseline to find when something was in the scene
        start, end = self._find_presence_window(event_path)
        candidates = images[start:end + 1]

        logger.info(
            "Presence window: frames %d-%d (%d/%d). Scanning for cats...",
            start, end, len(candidates), len(images),
        )
        for i, img_path in enumerate(candidates):
            frame_idx = start + i
            img = Image.open(img_path).convert("RGB")
            frame = np.array(img)
            cats = self._cat_detector.detect(frame)
            if cats:
                detection_count += 1
                best = max(cats, key=lambda d: d.confidence)
                logger.info("Cat in %s: %.1f%%", img_path.name, best.confidence * 100)
                if best.confidence > max_confidence:
                    max_confidence = best.confidence
                if first_cat_frame is None:
                    first_cat_frame = frame_idx
                last_cat_frame = frame_idx

        cat_detected = detection_count > 0
        if cat_detected:
            logger.info(
                "Cat detected in %d frames (frames %d-%d, best %.1f%%)",
                detection_count, first_cat_frame, last_cat_frame, max_confidence * 100,
            )
        else:
            logger.info("No cats detected in presence window")
        return cat_detected, max_confidence, detection_count, first_cat_frame, last_cat_frame

    def _record_event(self, trigger_score: float) -> None:
        """Run the RECORDING state: flush pre-roll, capture with motion tracking, then scan for cats."""
        start_ts = datetime.now().astimezone()
        event_path = event_dir_for(start_ts, self._output_root)
        event_path.mkdir(parents=True, exist_ok=True)
        logger.info("Motion detected (score %.2f). Recording to %s", trigger_score, event_path.name)

        image_count = 0

        # Flush pre-roll buffer
        pre_roll_frames = self._camera.drain_buffer()
        for bf in pre_roll_frames:
            self._save_jpeg(event_path, image_count, bf.jpeg_bytes)
            image_count += 1
        logger.info("Flushed %d pre-roll frames", len(pre_roll_frames))

        # Recording loop — compare consecutive lores frames for ongoing motion
        recording_start = time.time()
        last_activity = time.time()
        prev_lores = None

        while True:
            elapsed = time.time() - recording_start
            if elapsed >= self._max_capture_seconds:
                logger.info("Hit max capture time (%ds)", self._max_capture_seconds)
                break
            idle_time = time.time() - last_activity
            if idle_time >= self._post_roll_seconds:
                logger.info("No activity for %.1fs, stopping recording", idle_time)
                break

            # Capture main frame
            jpeg = self._camera.capture_main_jpeg()
            self._save_jpeg(event_path, image_count, jpeg)
            image_count += 1

            # Check motion by comparing consecutive lores frames
            lores = self._camera.capture_lores()
            gray = self._detector._to_gray(lores)
            if prev_lores is not None:
                score = float(np.mean(np.abs(gray - prev_lores)))
                if score >= self._motion_threshold:
                    last_activity = time.time()
            prev_lores = gray

            time.sleep(self._capture_interval)

        end_ts = datetime.now().astimezone()

        # Post-capture: baseline-guided cat detection
        cat_detected, max_confidence, detection_count, first_frame, last_frame = self._scan_for_cats(event_path)

        meta = EventMeta(
            start_ts=start_ts.isoformat(),
            end_ts=end_ts.isoformat(),
            trigger_score=trigger_score,
            image_count=image_count,
            cat_tag="auto:cat" if cat_detected else None,
            cat_detected=cat_detected if cat_detected else None,
            cat_confidence=round(max_confidence, 3) if cat_detected else None,
            detection_count=detection_count if cat_detected else None,
            cat_first_frame=first_frame,
            cat_last_frame=last_frame,
        )
        write_meta(event_path, meta)
        logger.info(
            "Event %s complete: %d images, cat=%s (%.1f%%, %d frames)",
            event_path.name,
            image_count,
            cat_detected,
            max_confidence * 100,
            detection_count,
        )

    def _write_snapshot(self) -> None:
        """Write the latest buffered frame as a snapshot for the live view."""
        if self._camera._buffer:
            latest = self._camera._buffer[-1]
            tmp = self._snapshot_path.with_suffix(".tmp")
            tmp.write_bytes(latest.jpeg_bytes)
            tmp.replace(self._snapshot_path)

    def _update_baseline(self) -> None:
        """Update baseline image when scene is stable (for background subtraction)."""
        baseline_path = self._output_root / "baseline.jpg"
        if self._camera._buffer:
            latest = self._camera._buffer[-1]
            tmp = baseline_path.with_suffix(".tmp")
            tmp.write_bytes(latest.jpeg_bytes)
            tmp.replace(baseline_path)

    def run(self) -> None:
        configure_logging(self._config.logging)
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._camera.start()
        logger.info("Capture service started (IDLE)")

        self._snapshot_path = self._output_root / "snapshot.jpg"

        state = State.IDLE
        last_motion_time = time.time()
        baseline_updated = False
        try:
            while True:
                if state == State.IDLE:
                    # Capture lores for motion detection
                    lores = self._camera.capture_lores()
                    trigger = self._detector.analyze(lores)

                    # Buffer a main frame for pre-roll and write snapshot
                    self._camera.buffer_frame()
                    self._write_snapshot()

                    if trigger is not None:
                        state = State.RECORDING
                        self._record_event(trigger)
                        self._detector.reset()
                        state = State.COOLDOWN
                        last_motion_time = time.time()
                        baseline_updated = False
                    else:
                        # Update baseline when scene is stable for 60s
                        if not baseline_updated and (time.time() - last_motion_time) > 60:
                            self._update_baseline()
                            baseline_updated = True
                            logger.debug("Baseline image updated")
                        time.sleep(self._capture_interval)

                elif state == State.COOLDOWN:
                    logger.info("Cooldown for %ds", self._cooldown_seconds)
                    time.sleep(self._cooldown_seconds)
                    state = State.IDLE
                    logger.info("Back to IDLE")
        finally:
            self._camera.stop()

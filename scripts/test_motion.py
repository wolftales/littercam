#!/usr/bin/env python3
"""Quick test script to see motion scores in real time."""
import time
import numpy as np
from littercam.config import load_config
from littercam.camera import CameraManager
from littercam.detection import MotionDetector, MotionConfig

cfg = load_config()
cap = cfg.capture
cam = CameraManager(
    main_width=cap.main_width,
    main_height=cap.main_height,
    lores_width=cap.downscale_width,
    lores_height=cap.downscale_height,
    jpeg_quality=cap.jpeg_quality,
    pre_roll_seconds=cap.pre_roll_seconds,
    capture_interval=cap.capture_interval_seconds,
)
det = MotionDetector(MotionConfig(
    threshold=cap.motion_threshold,
    downscale_width=cap.downscale_width,
    downscale_height=cap.downscale_height,
    trigger_frames=cap.trigger_frames,
))
cam.start()
print(f"Threshold: {cap.motion_threshold}, trigger_frames: {cap.trigger_frames}")

prev_gray = None
try:
    for i in range(120):
        frame = cam.capture_lores()
        if i == 0:
            print(f"Frame shape: {frame.shape}")
            print(f"Y plane: {cap.downscale_height}x{cap.downscale_width}")

        # Compute our own score for display
        y = frame[:cap.downscale_height, :cap.downscale_width].astype("float32")
        if prev_gray is not None:
            score = float(np.mean(np.abs(y - prev_gray)))
        else:
            score = 0.0
        prev_gray = y

        # Run the real analyzer
        result = det.analyze(frame)
        status = " TRIGGERED!" if result else ""
        print(f"score={score:.2f}{status}")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nStopped")
finally:
    cam.stop()

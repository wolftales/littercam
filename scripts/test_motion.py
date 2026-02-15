#!/usr/bin/env python3
"""Quick test script to see motion scores in real time."""
import time
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
try:
    for i in range(120):
        frame = cam.capture_lores()
        if i == 0:
            print(f"Frame shape: {frame.shape}")
        result = det.analyze(frame)
        score = det.current_score(frame)
        status = " TRIGGERED!" if result else ""
        print(f"score={score:.2f}{status}")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nStopped")
finally:
    cam.stop()

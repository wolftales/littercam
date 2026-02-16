#!/usr/bin/env python3
"""Re-scan all events with the current cat detection model and update meta.json."""
import sys
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from littercam.cat_detector import CatDetector
from littercam.config import load_config
from littercam.events import list_events, write_meta

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

cfg = load_config()
cat_cfg = cfg.cat_detection

detector = CatDetector(
    model_path=cat_cfg.model_path,
    labels_path=cat_cfg.labels_path,
    confidence_threshold=cat_cfg.confidence_threshold,
    enabled=cat_cfg.enabled,
)

if not detector._enabled:
    print("Cat detection is not available. Check model path and dependencies.")
    sys.exit(1)

# Load baseline if available for presence window detection
baseline_path = cfg.capture.output_root / "baseline.jpg"
baseline_gray = None
if baseline_path.exists():
    baseline_gray = np.array(Image.open(baseline_path).convert("L"), dtype=np.float32)
    print(f"Baseline loaded: {baseline_path}\n")
else:
    print("No baseline image found — scanning all frames per event\n")


def find_presence_window(event_path: Path) -> tuple[int, int]:
    """Use baseline diff to find when something entered/left the scene."""
    thumbs = sorted(event_path.glob("thumb-*.jpg"))
    n = len(thumbs)
    if n == 0 or baseline_gray is None:
        return 0, max(0, n - 1)

    diffs = []
    for thumb_path in thumbs:
        img = Image.open(thumb_path).convert("L").resize(
            (baseline_gray.shape[1], baseline_gray.shape[0])
        )
        gray = np.array(img, dtype=np.float32)
        diffs.append(float(np.mean(np.abs(gray - baseline_gray))))

    median_diff = sorted(diffs)[len(diffs) // 2]
    threshold = max(median_diff * 2.0, 3.0)

    first = None
    last = None
    for i, d in enumerate(diffs):
        if d >= threshold:
            if first is None:
                first = i
            last = i

    if first is None:
        return 0, n - 1

    # Buffer of 3 frames before/after
    return max(0, first - 3), min(n - 1, last + 3)


def scan_frames(images: list[Path], start: int, end: int) -> tuple[float, int, int | None, int | None]:
    """Run YOLO on images[start:end+1]. Returns (max_confidence, count, first, last)."""
    max_confidence = 0.0
    detection_count = 0
    first_cat = None
    last_cat = None

    for i, img_path in enumerate(images[start:end + 1]):
        frame_idx = start + i
        img = Image.open(img_path).convert("RGB")
        frame = np.array(img)
        cats = detector.detect(frame)
        if cats:
            detection_count += 1
            best = max(cats, key=lambda d: d.confidence)
            if best.confidence > max_confidence:
                max_confidence = best.confidence
            if first_cat is None:
                first_cat = frame_idx
            last_cat = frame_idx

    return max_confidence, detection_count, first_cat, last_cat


def scan_event(event_path: Path) -> tuple[bool, float, int, int | None, int | None]:
    """Run cat detection on frames within the presence window.

    Extends past the window if cat was still present at the tail (not leaving).
    """
    images = sorted(event_path.glob("img-*.jpg"))
    if not images:
        return False, 0.0, 0, None, None

    n = len(images)
    start, end = find_presence_window(event_path)

    # Phase 1: scan presence window
    max_conf, det_count, first_cat, last_cat = scan_frames(images, start, end)

    # Phase 2: extend if cat still present at tail of window (not leaving)
    if det_count > 0 and last_cat is not None and end < n - 1:
        cat_at_tail = last_cat >= end - 2
        if cat_at_tail:
            ext_conf, ext_count, _, ext_last = scan_frames(images, end + 1, n - 1)
            det_count += ext_count
            if ext_conf > max_conf:
                max_conf = ext_conf
            if ext_last is not None:
                last_cat = ext_last

    return det_count > 0, max_conf, det_count, first_cat, last_cat


events = list_events(cfg.capture.output_root)
print(f"Found {len(events)} events to scan\n")

updated = 0
detected = 0

for event in events:
    images = sorted(event.event_path.glob("img-*.jpg"))
    start, end = find_presence_window(event.event_path)
    cat_detected, confidence, count, first_frame, last_frame = scan_event(event.event_path)

    # Update meta
    event.meta.cat_detected = cat_detected if cat_detected else None
    event.meta.cat_confidence = round(confidence, 3) if cat_detected else None
    event.meta.detection_count = count if cat_detected else None
    event.meta.cat_first_frame = first_frame
    event.meta.cat_last_frame = last_frame
    if cat_detected and not event.meta.cat_tag:
        event.meta.cat_tag = "auto:cat"

    write_meta(event.event_path, event.meta)
    updated += 1

    if cat_detected:
        detected += 1
        print(f"  {event.event_id}: Cat {confidence:.0%} ({count}/{len(images)} frames, #{first_frame}-{last_frame})")
    else:
        print(f"  {event.event_id}: No cat (scanned {end - start + 1}/{len(images)} frames)")

print(f"\nDone: {updated} events scanned, {detected} cats detected")

#!/usr/bin/env python3
"""Re-scan all events with the current cat detection model and update meta.json."""
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from littercam.cat_detector import CatDetector
from littercam.config import load_config
from littercam.events import list_events, load_event, write_meta

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


def find_interesting_frames(event_path: Path, top_n: int = 10) -> list[int]:
    """Use cheap thumbnail diffs to find the most visually interesting frame indices."""
    thumbs = sorted(event_path.glob("thumb-*.jpg"))
    if len(thumbs) <= top_n:
        return list(range(len(thumbs)))

    scores: list[tuple[float, int]] = []
    prev_gray = None
    for i, thumb_path in enumerate(thumbs):
        img = Image.open(thumb_path).convert("L")
        gray = np.array(img, dtype=np.float32)
        if prev_gray is not None:
            score = float(np.mean(np.abs(gray - prev_gray)))
            scores.append((score, i))
        prev_gray = gray

    scores.sort(reverse=True)
    seen = set()
    result = []
    for _, idx in scores:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
            if len(result) >= top_n:
                break
    return sorted(result)


def scan_event(event_path: Path) -> tuple[bool, float, int]:
    """Run cat detection on ALL full-res frames (batch job, no time pressure)."""
    images = sorted(event_path.glob("img-*.jpg"))
    if not images:
        return False, 0.0, 0

    max_confidence = 0.0
    detection_count = 0

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        frame = np.array(img)
        cats = detector.detect(frame)
        if cats:
            detection_count += 1
            best = max(cats, key=lambda d: d.confidence)
            if best.confidence > max_confidence:
                max_confidence = best.confidence

    return detection_count > 0, max_confidence, detection_count


events = list_events(cfg.capture.output_root)
print(f"Found {len(events)} events to scan\n")

updated = 0
detected = 0

for event in events:
    cat_detected, confidence, count = scan_event(event.event_path)

    # Update meta
    event.meta.cat_detected = cat_detected if cat_detected else None
    event.meta.cat_confidence = round(confidence, 3) if cat_detected else None
    event.meta.detection_count = count if cat_detected else None
    if cat_detected and not event.meta.cat_tag:
        event.meta.cat_tag = "auto:cat"

    write_meta(event.event_path, event.meta)
    updated += 1

    status = f"Cat {confidence:.0%} ({count} frames)" if cat_detected else "No cat"
    print(f"  {event.event_id}: {status}")
    if cat_detected:
        detected += 1

print(f"\nDone: {updated} events scanned, {detected} cats detected")

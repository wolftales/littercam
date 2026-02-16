#!/usr/bin/env python3
"""Diagnostic script: run cat detector on an event's thumbnails and show ALL detections."""
import sys
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

# Usage: python scripts/test_cat_detect.py data/events/2026-02-16/event-20260216-050121
if len(sys.argv) < 2:
    print("Usage: python scripts/test_cat_detect.py <event_path>")
    print("Example: python scripts/test_cat_detect.py data/events/2026-02-16/event-20260216-050121")
    sys.exit(1)

event_path = Path(sys.argv[1])
if not event_path.exists():
    print(f"Path not found: {event_path}")
    sys.exit(1)

from littercam.config import load_config

cfg = load_config()
cat_cfg = cfg.cat_detection

# Load model directly to inspect outputs
import onnxruntime as ort

model_file = Path(cat_cfg.model_path)
if not model_file.exists():
    print(f"Model not found: {model_file}")
    sys.exit(1)

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession(
    str(model_file), sess_options=opts, providers=["CPUExecutionProvider"]
)

# Show model I/O info
print("=== Model Info ===")
inp = session.get_inputs()[0]
print(f"Input: {inp.name} shape={inp.shape} type={inp.type}")
for i, out in enumerate(session.get_outputs()):
    print(f"Output[{i}]: {out.name} shape={out.shape} type={out.type}")

# Build output map
output_map = {out.name.split(":")[0]: i for i, out in enumerate(session.get_outputs())}
print(f"Output map: {output_map}")

raw_shape = inp.shape
h = raw_shape[1] if isinstance(raw_shape[1], int) else 300
w = raw_shape[2] if isinstance(raw_shape[2], int) else 300
print(f"Input size: {h}x{w}")

# Load labels
labels = {}
labels_file = Path(cat_cfg.labels_path)
if labels_file.exists():
    for line in labels_file.read_text().splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            labels[int(parts[0])] = parts[1]

# Scan thumbnails
thumbs = sorted(event_path.glob("thumb-*.jpg"))
if not thumbs:
    thumbs = sorted(event_path.glob("img-*.jpg"))
print(f"\n=== Scanning {len(thumbs)} images ===")

boxes_idx = output_map.get("detection_boxes", 0)
classes_idx = output_map.get("detection_classes", 1)
scores_idx = output_map.get("detection_scores", 2)
num_idx = output_map.get("num_detections", 3)

for thumb_path in thumbs:
    img = Image.open(thumb_path).convert("RGB").resize((w, h))
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)
    outputs = session.run(None, {inp.name: input_data})

    boxes = outputs[boxes_idx][0]
    classes = outputs[classes_idx][0]
    scores = outputs[scores_idx][0]
    num = int(outputs[num_idx][0]) if num_idx < len(outputs) else len(scores)

    # Show all detections above 10%
    hits = []
    for i in range(min(num, len(scores))):
        score = float(scores[i])
        class_id = int(classes[i])
        if score >= 0.1:
            label = labels.get(class_id, f"class_{class_id}")
            hits.append(f"{label}({class_id})={score:.0%}")

    if hits:
        print(f"{thumb_path.name}: {', '.join(hits)}")
    else:
        print(f"{thumb_path.name}: (nothing above 10%)")

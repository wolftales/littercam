#!/usr/bin/env python3
"""Diagnostic script: run YOLOv8n cat detector on an event's images and show ALL detections."""
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

raw_shape = inp.shape
h = raw_shape[2] if len(raw_shape) >= 4 and isinstance(raw_shape[2], int) else 640
w = raw_shape[3] if len(raw_shape) >= 4 and isinstance(raw_shape[3], int) else 640
print(f"Input size: {h}x{w}")

# Load labels
labels = {}
labels_file = Path(cat_cfg.labels_path)
if labels_file.exists():
    for line in labels_file.read_text().splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            labels[int(parts[0])] = parts[1]

# Scan full-res images (fall back to thumbnails)
images = sorted(event_path.glob("img-*.jpg"))
if not images:
    images = sorted(event_path.glob("thumb-*.jpg"))
print(f"\n=== Scanning {len(images)} images ===")

for img_path in images:
    img = Image.open(img_path).convert("RGB").resize((w, h))
    img_array = np.array(img, dtype=np.float32) / 255.0
    input_data = img_array.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> NCHW

    outputs = session.run(None, {inp.name: input_data})

    # YOLOv8 output: [1, 84, 8400] -> [8400, 84]
    raw = outputs[0][0].T
    class_scores = raw[:, 4:]  # [8400, 80]
    class_ids = class_scores.argmax(axis=1)
    max_scores = class_scores.max(axis=1)

    # Show all detections above 15%
    mask = max_scores >= 0.15
    hits = []
    if mask.any():
        for i in np.where(mask)[0]:
            cid = int(class_ids[i])
            score = float(max_scores[i])
            label = labels.get(cid, f"class_{cid}")
            hits.append(f"{label}({cid})={score:.0%}")
        # Deduplicate by class, show best score per class
        best_per_class = {}
        for i in np.where(mask)[0]:
            cid = int(class_ids[i])
            score = float(max_scores[i])
            if cid not in best_per_class or score > best_per_class[cid]:
                best_per_class[cid] = score
        hits = [f"{labels.get(c, f'class_{c}')}({c})={s:.0%}" for c, s in sorted(best_per_class.items(), key=lambda x: -x[1])]

    if hits:
        print(f"{img_path.name}: {', '.join(hits)}")
    else:
        print(f"{img_path.name}: (nothing above 15%)")

#!/usr/bin/env bash
# Export YOLOv8s to ONNX format for cat detection.
#
# YOLOv8s is larger than YOLOv8n (~22MB vs ~6MB) but significantly better
# at detecting difficult cases (e.g. black cat in black litter box).
# Post-capture scanning isn't time-critical so the extra ~50ms is fine.
#
# Requires: pip install ultralytics (one-time, pulls in torch)
# At runtime only onnxruntime is needed — ultralytics/torch can be removed after export.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_DIR="models"
MODEL_FILE="$MODEL_DIR/yolov8s.onnx"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
  echo "Model already exists at $MODEL_FILE"
else
  echo "==> Exporting YOLOv8s to ONNX..."
  echo "    (requires: pip install ultralytics)"
  python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.export(format='onnx', imgsz=640, simplify=True)
import shutil, os
shutil.move('yolov8s.onnx', '$MODEL_FILE')
for f in ['yolov8s.pt']:
    if os.path.exists(f):
        os.remove(f)
"
  echo "    Exported to $MODEL_FILE"
fi

echo "==> Creating COCO 80-class labels file..."
cat > "$MODEL_DIR/coco_labels.txt" << 'LABELS'
0 person
1 bicycle
2 car
3 motorcycle
4 airplane
5 bus
6 train
7 truck
8 boat
9 traffic light
10 fire hydrant
11 stop sign
12 parking meter
13 bench
14 bird
15 cat
16 dog
17 horse
18 sheep
19 cow
20 elephant
21 bear
22 zebra
23 giraffe
24 backpack
25 umbrella
26 handbag
27 tie
28 suitcase
29 frisbee
30 skis
31 snowboard
32 sports ball
33 kite
34 baseball bat
35 baseball glove
36 skateboard
37 surfboard
38 tennis racket
39 bottle
40 wine glass
41 cup
42 fork
43 knife
44 spoon
45 bowl
46 banana
47 apple
48 sandwich
49 orange
50 broccoli
51 carrot
52 hot dog
53 pizza
54 donut
55 cake
56 chair
57 couch
58 potted plant
59 bed
60 dining table
61 toilet
62 tv
63 laptop
64 mouse
65 remote
66 keyboard
67 cell phone
68 microwave
69 oven
70 toaster
71 sink
72 refrigerator
73 book
74 clock
75 vase
76 scissors
77 teddy bear
78 hair drier
79 toothbrush
LABELS

echo "==> Model files ready in $MODEL_DIR/"
ls -lh "$MODEL_FILE"
echo "Done!"

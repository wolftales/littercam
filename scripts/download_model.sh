#!/usr/bin/env bash
# Download the quantized MobileNet SSD v2 COCO model for cat detection.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_DIR="models"
MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
LABELS_URL="https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt"

mkdir -p "$MODEL_DIR"

echo "==> Downloading quantized SSD MobileNet v2 COCO model..."
TMP_ZIP=$(mktemp)
curl -fSL "$MODEL_URL" -o "$TMP_ZIP"
unzip -o "$TMP_ZIP" -d "$MODEL_DIR"
mv "$MODEL_DIR/detect.tflite" "$MODEL_DIR/ssd_mobilenet_v2.tflite" 2>/dev/null || true
rm -f "$TMP_ZIP"

echo "==> Creating COCO labels file..."
cat > "$MODEL_DIR/coco_labels.txt" << 'LABELS'
0 background
1 person
2 bicycle
3 car
4 motorcycle
5 airplane
6 bus
7 train
8 truck
9 boat
10 traffic light
11 fire hydrant
13 stop sign
14 parking meter
15 bench
16 bird
17 cat
18 dog
19 horse
20 sheep
21 cow
22 elephant
23 bear
24 zebra
25 giraffe
27 backpack
28 umbrella
31 handbag
32 tie
33 suitcase
34 frisbee
35 skis
36 snowboard
37 sports ball
38 kite
39 baseball bat
40 baseball glove
41 skateboard
42 surfboard
43 tennis racket
44 bottle
46 wine glass
47 cup
48 fork
49 knife
50 spoon
51 bowl
52 banana
53 apple
54 sandwich
55 orange
56 broccoli
57 carrot
58 hot dog
59 pizza
60 donut
61 cake
62 chair
63 couch
64 potted plant
65 bed
67 dining table
70 toilet
72 tv
73 laptop
74 mouse
75 remote
76 keyboard
77 cell phone
78 microwave
79 oven
80 toaster
81 sink
82 refrigerator
84 book
85 clock
86 vase
87 scissors
88 teddy bear
89 hair drier
90 toothbrush
LABELS

echo "==> Model files ready in $MODEL_DIR/"
ls -la "$MODEL_DIR/"
echo "Done!"

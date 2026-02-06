#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y libcamera-apps python3-picamera2 python3-venv python3-dev

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .[pi]

sudo mkdir -p /data/events
sudo mkdir -p /etc/littercam
sudo cp config/littercam.yaml /etc/littercam/config.yaml

sudo cp systemd/littercam-capture.service /etc/systemd/system/
sudo cp systemd/littercam-web.service /etc/systemd/system/
sudo cp systemd/littercam-prune.service /etc/systemd/system/
sudo cp systemd/littercam-prune.timer /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now littercam-capture.service
sudo systemctl enable --now littercam-web.service
sudo systemctl enable --now littercam-prune.timer

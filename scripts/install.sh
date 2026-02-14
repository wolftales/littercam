#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

echo "==> Installing system dependencies..."
sudo apt update
sudo apt install -y libcamera-apps python3-picamera2 python3-venv python3-dev libcap-dev

echo "==> Setting up Python environment..."
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .[pi]

echo "==> Installing tflite-runtime for cat detection..."
pip install tflite-runtime 2>/dev/null || \
  sudo apt install -y python3-tflite-runtime 2>/dev/null || \
  echo "NOTE: tflite-runtime not available for Python $(python3 --version 2>&1). Cat detection will be disabled — motion capture still works."

echo "==> Creating data directory..."
mkdir -p data/events

echo "==> Downloading cat detection model..."
bash "$PROJECT_DIR/scripts/download_model.sh"

echo "==> Installing user systemd services..."
mkdir -p ~/.config/systemd/user
cp systemd/littercam-capture.service ~/.config/systemd/user/
cp systemd/littercam-web.service ~/.config/systemd/user/
cp systemd/littercam-prune.service ~/.config/systemd/user/
cp systemd/littercam-prune.timer ~/.config/systemd/user/

systemctl --user daemon-reload

echo "==> Enabling services..."
systemctl --user enable littercam-capture.service
systemctl --user enable littercam-web.service
systemctl --user enable littercam-prune.timer

# Enable lingering so user services start at boot without login
sudo loginctl enable-linger "$USER"

echo ""
echo "Installation complete!"
echo ""
echo "Start services now with:"
echo "  systemctl --user start littercam-capture"
echo "  systemctl --user start littercam-web"
echo ""
echo "View logs with:"
echo "  journalctl --user -u littercam-capture -f"
echo "  journalctl --user -u littercam-web -f"
echo ""
echo "Web UI will be at: http://$(hostname -I | awk '{print $1}'):8000"

# LitterCam v1

LitterCam v1 is a LAN-only Raspberry Pi camera system that detects motion, captures bursts of stills, and serves a simple web UI to browse and tag events. It is designed for Raspberry Pi OS Bookworm on a Pi4 with a ribbon camera (libcamera via Picamera2).

## Repo layout

```
app/                # FastAPI web app
  main.py
  templates/
  static/
config/
  littercam.yaml    # Sample configuration
scripts/            # Helper scripts
src/
  littercam/        # Core package
systemd/            # Service/timer units
tests/              # Light unit tests
```

## Requirements

- Raspberry Pi OS Bookworm
- Python 3.11+
- libcamera + Picamera2

## Quick start (assumes repo cloned to ~/littercam)

```bash
cd ~/littercam
sudo mkdir -p /data/events
sudo mkdir -p /etc/littercam
sudo cp config/littercam.yaml /etc/littercam/config.yaml
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .[pi,test]
```

Run the capture service manually:

```bash
littercam-capture
```

Run the web server locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open from a LAN device: `http://<pi-ip>:8000/latest`.

## Installation on Raspberry Pi OS (Bookworm)

Install system packages:

```bash
sudo apt update
sudo apt install -y libcamera-apps python3-picamera2 python3-venv python3-dev
```

Install Python dependencies and enable services:

```bash
cd ~/littercam
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
```

## Configuration

Edit `/etc/littercam/config.yaml` to adjust capture duration, cooldown, thresholds, and retention. The web service reads the same config file.

## Logging

- Services log to journald by default.
- Optional file logging can be enabled in config (`logging.file_logging: true`). Logs will be stored in `/var/log/littercam/littercam.log` with rotation handled by the app (5 MB, 3 backups). For OS-level rotation, add a logrotate config.

## Data layout

```
/data/events/YYYY-MM-DD/event-YYYYMMDD-HHMMSS/
  img-000.jpg
  thumb-000.jpg
  meta.json
```

## Notes

- LitterCam is LAN-only and does not add authentication by default. You can front it with Nginx or Caddy if authentication is required.
- The capture service uses Picamera2, which wraps libcamera. Ensure the camera is enabled with `sudo raspi-config` if needed.

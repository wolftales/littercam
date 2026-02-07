# LitterCam

A self-contained Raspberry Pi camera system that detects motion, captures bursts of stills, and serves a web UI to browse and tag events. Designed for Raspberry Pi OS Bookworm with a ribbon camera.

## Project Structure

```
littercam/
├── app/                  # FastAPI web app
├── config/
│   └── littercam.yaml    # Configuration
├── data/events/          # Captured images (created on first run)
├── scripts/
│   └── install.sh        # Installation script
├── src/littercam/        # Core package
├── systemd/              # User service files
└── tests/
```

## Requirements

- Raspberry Pi OS Bookworm
- Python 3.11+
- Ribbon camera (libcamera compatible)

## Quick Install

```bash
cd ~/git/littercam
./scripts/install.sh
```

This will:
- Install system dependencies (libcamera, picamera2)
- Create Python virtual environment
- Install user systemd services (no root required for services)
- Enable auto-start on boot

## Manual Setup

```bash
cd ~/git/littercam

# Install system packages
sudo apt update
sudo apt install -y libcamera-apps python3-picamera2 python3-venv python3-dev

# Setup Python environment
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[pi]

# Create data directory
mkdir -p data/events
```

## Running

### Manual (foreground)

```bash
cd ~/git/littercam
. .venv/bin/activate

# Terminal 1: Capture service
littercam-capture

# Terminal 2: Web server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### As Services (background, auto-restart)

```bash
# Start
systemctl --user start littercam-capture
systemctl --user start littercam-web

# Stop
systemctl --user stop littercam-capture
systemctl --user stop littercam-web

# View logs
journalctl --user -u littercam-capture -f
journalctl --user -u littercam-web -f

# Check status
systemctl --user status littercam-capture
systemctl --user status littercam-web
```

## Web UI

Access from any device on your LAN:

```
http://<pi-ip>:8000
```

- `/latest` - Most recent event
- `/events` - Browse all events by date

## Configuration

Edit `config/littercam.yaml`:

```yaml
capture:
  output_root: ./data/events    # Where images are stored
  cooldown_seconds: 120         # Wait between capture bursts
  capture_seconds: 18           # Duration of each burst
  motion_threshold: 12.0        # Sensitivity (lower = more sensitive)
  trigger_frames: 3             # Consecutive frames to confirm motion

retention:
  days_to_keep: 7               # Auto-delete older events

web:
  host: 0.0.0.0
  port: 8000
```

## Data Layout

```
data/events/
└── 2025-02-07/
    └── event-20250207-143052/
        ├── img-000.jpg
        ├── thumb-000.jpg
        ├── meta.json
        └── ...
```

## Notes

- LAN-only, no authentication. Add a reverse proxy (nginx/caddy) if needed.
- Ensure camera is enabled: `sudo raspi-config` → Interface Options → Camera

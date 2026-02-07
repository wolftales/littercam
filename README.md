# LitterCam

A self-contained Raspberry Pi camera system that detects motion, captures bursts of stills, and serves a web UI to browse and tag events. Designed for Raspberry Pi OS Bookworm with a ribbon camera.

## Requirements

- Raspberry Pi OS Bookworm
- Python 3.11+
- Ribbon camera (libcamera compatible)
- Camera enabled: `sudo raspi-config` → Interface Options → Camera

## 1. Install

```bash
cd ~/git/littercam
./scripts/install.sh
```

This installs system packages (libcamera, picamera2, python3-venv), creates a Python virtual environment, and sets up user systemd services.

## 2. Test

Start both services:

```bash
systemctl --user start littercam-capture
systemctl --user start littercam-web
```

Check they're running:

```bash
systemctl --user status littercam-capture
systemctl --user status littercam-web
```

Open in browser from any device on your LAN:

```
http://<pi-ip>:8000
```

Wave your hand in front of the camera. After a few seconds of motion, it triggers a capture burst. Refresh the browser to see the event.

Check logs if something isn't working:

```bash
journalctl --user -u littercam-capture -f
journalctl --user -u littercam-web -f
```

## 3. Run (Production)

The install script already enabled auto-start on boot. Just leave it running:

```bash
# Check status
systemctl --user status littercam-capture
systemctl --user status littercam-web

# Restart if needed
systemctl --user restart littercam-capture
systemctl --user restart littercam-web

# Stop if needed
systemctl --user stop littercam-capture
systemctl --user stop littercam-web
```

Services auto-restart on failure and start on boot (via lingering).

## Web UI

- `http://<pi-ip>:8000/latest` - Most recent event
- `http://<pi-ip>:8000/events` - Browse all events by date

## Configuration

Edit `config/littercam.yaml` and restart services:

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

## Project Structure

```
littercam/
├── app/                  # FastAPI web app
├── config/
│   └── littercam.yaml    # Configuration
├── data/events/          # Captured images (created on first run)
├── scripts/
│   └── install.sh
├── src/littercam/        # Core package
├── systemd/              # User service files
└── tests/
```

## Data Layout

```
data/events/
└── 2025-02-07/
    └── event-20250207-143052/
        ├── img-000.jpg      # Full resolution
        ├── thumb-000.jpg    # Thumbnail
        ├── meta.json        # Event metadata
        └── ...
```

## Notes

- LAN-only, no authentication. Add a reverse proxy (nginx/caddy) if needed.
- Old events are automatically pruned daily based on `retention.days_to_keep`.

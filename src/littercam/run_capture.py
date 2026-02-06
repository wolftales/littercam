"""Entrypoint for the capture service."""
from __future__ import annotations

from pathlib import Path

from littercam.capture_service import CaptureService
from littercam.config import load_config


def main(config_path: Path | None = None) -> None:
    config = load_config(config_path)
    service = CaptureService(config)
    service.run()


if __name__ == "__main__":
    main()

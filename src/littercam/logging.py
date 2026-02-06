"""Logging helpers for LitterCam."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from littercam.config import LoggingConfig

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(config: LoggingConfig) -> None:
    """Configure logging for console/journald and optional file logging."""
    logging.basicConfig(level=config.level, format=DEFAULT_FORMAT)

    if not config.file_logging:
        return

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "littercam.log"
    handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    logging.getLogger().addHandler(handler)

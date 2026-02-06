"""Retention pruning for LitterCam."""
from __future__ import annotations

import logging
from pathlib import Path

from littercam.config import AppConfig, load_config
from littercam.events import prune_events
from littercam.logging import configure_logging

logger = logging.getLogger(__name__)


def run_prune(config_path: Path | None = None) -> None:
    config = load_config(config_path)
    configure_logging(config.logging)
    removed = prune_events(config.capture.output_root, config.retention.days_to_keep)
    logger.info("Pruned %s event folders", len(removed))


if __name__ == "__main__":
    run_prune()

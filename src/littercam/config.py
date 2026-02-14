"""Configuration loader for LitterCam."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path("/etc/littercam/config.yaml")


@dataclass(frozen=True)
class CaptureConfig:
    output_root: Path
    cooldown_seconds: int
    capture_seconds: int
    capture_interval_seconds: float
    motion_threshold: float
    downscale_width: int
    downscale_height: int
    trigger_frames: int
    pre_roll_seconds: int
    post_roll_seconds: int
    max_capture_seconds: int
    main_width: int
    main_height: int
    jpeg_quality: int


@dataclass(frozen=True)
class CatDetectionConfig:
    enabled: bool
    model_path: str
    labels_path: str
    confidence_threshold: float


@dataclass(frozen=True)
class RetentionConfig:
    days_to_keep: int


@dataclass(frozen=True)
class WebConfig:
    host: str
    port: int
    base_url: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    file_logging: bool
    log_dir: Path


@dataclass(frozen=True)
class AppConfig:
    capture: CaptureConfig
    retention: RetentionConfig
    web: WebConfig
    logging: LoggingConfig
    cat_detection: CatDetectionConfig


def _require(data: Dict[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"Missing required config key: {key}")
    return data[key]


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from YAML."""
    env_path = os.getenv("LITTERCAM_CONFIG")
    config_path = path or (Path(env_path) if env_path else None)
    if config_path is None:
        config_path = Path(Path.cwd() / "config" / "littercam.yaml")
    if not config_path.exists():
        config_path = DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text())
    capture_raw = _require(raw, "capture")
    retention_raw = _require(raw, "retention")
    web_raw = _require(raw, "web")
    logging_raw = _require(raw, "logging")
    cat_detection_raw = raw.get("cat_detection", {})

    # Accept capture_seconds as alias for max_capture_seconds
    max_capture = capture_raw.get(
        "max_capture_seconds",
        capture_raw.get("capture_seconds", 120),
    )

    return AppConfig(
        capture=CaptureConfig(
            output_root=Path(_require(capture_raw, "output_root")),
            cooldown_seconds=int(_require(capture_raw, "cooldown_seconds")),
            capture_seconds=int(
                capture_raw.get("capture_seconds", max_capture)
            ),
            capture_interval_seconds=float(
                _require(capture_raw, "capture_interval_seconds")
            ),
            motion_threshold=float(_require(capture_raw, "motion_threshold")),
            downscale_width=int(_require(capture_raw, "downscale_width")),
            downscale_height=int(_require(capture_raw, "downscale_height")),
            trigger_frames=int(_require(capture_raw, "trigger_frames")),
            pre_roll_seconds=int(capture_raw.get("pre_roll_seconds", 5)),
            post_roll_seconds=int(capture_raw.get("post_roll_seconds", 5)),
            max_capture_seconds=int(max_capture),
            main_width=int(capture_raw.get("main_width", 1920)),
            main_height=int(capture_raw.get("main_height", 1080)),
            jpeg_quality=int(capture_raw.get("jpeg_quality", 85)),
        ),
        retention=RetentionConfig(
            days_to_keep=int(_require(retention_raw, "days_to_keep"))
        ),
        web=WebConfig(
            host=str(_require(web_raw, "host")),
            port=int(_require(web_raw, "port")),
            base_url=str(_require(web_raw, "base_url")),
        ),
        logging=LoggingConfig(
            level=str(_require(logging_raw, "level")),
            file_logging=bool(_require(logging_raw, "file_logging")),
            log_dir=Path(_require(logging_raw, "log_dir")),
        ),
        cat_detection=CatDetectionConfig(
            enabled=bool(cat_detection_raw.get("enabled", True)),
            model_path=str(cat_detection_raw.get("model_path", "./models/ssd_mobilenet_v1.onnx")),
            labels_path=str(cat_detection_raw.get("labels_path", "./models/coco_labels.txt")),
            confidence_threshold=float(cat_detection_raw.get("confidence_threshold", 0.5)),
        ),
    )

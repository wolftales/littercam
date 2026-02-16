"""Event management for LitterCam."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d"
EVENT_ID_FORMAT = "%Y%m%d-%H%M%S"


@dataclass
class EventMeta:
    start_ts: str
    end_ts: str
    trigger_score: float
    image_count: int
    cat_tag: Optional[str] = None
    cat_detected: Optional[bool] = None
    cat_confidence: Optional[float] = None
    detection_count: Optional[int] = None

    def to_json(self) -> str:
        d = asdict(self)
        # Omit None values for clean JSON
        return json.dumps({k: v for k, v in d.items() if v is not None}, indent=2)


@dataclass
class Event:
    event_id: str
    event_path: Path
    date: str
    meta: EventMeta


def event_dir_for(ts: datetime, output_root: Path) -> Path:
    date_path = ts.strftime(DATE_FORMAT)
    event_id = ts.strftime(EVENT_ID_FORMAT)
    return output_root / date_path / f"event-{event_id}"


def load_event(event_path: Path) -> Optional[Event]:
    meta_path = event_path / "meta.json"
    if not meta_path.exists():
        logger.warning("Missing meta.json in %s", event_path)
        return None
    try:
        meta_data = json.loads(meta_path.read_text())
        # Filter to only known fields for backward compatibility
        known = {f.name for f in EventMeta.__dataclass_fields__.values()}
        meta = EventMeta(**{k: v for k, v in meta_data.items() if k in known})
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to parse %s: %s", meta_path, exc)
        return None
    event_id = event_path.name.replace("event-", "")
    date = event_path.parent.name
    return Event(event_id=event_id, event_path=event_path, date=date, meta=meta)


def list_events(output_root: Path) -> list[Event]:
    events: list[Event] = []
    if not output_root.exists():
        return events
    for date_dir in sorted(output_root.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for event_dir in sorted(date_dir.iterdir(), reverse=True):
            if not event_dir.is_dir():
                continue
            event = load_event(event_dir)
            if event:
                events.append(event)
    return events


def latest_event(output_root: Path) -> Optional[Event]:
    events = list_events(output_root)
    return events[0] if events else None


def write_meta(event_path: Path, meta: EventMeta) -> None:
    meta_path = event_path / "meta.json"
    meta_path.write_text(meta.to_json())


def _should_keep(event_dir: Path) -> bool:
    """Check if an event should be preserved from pruning."""
    event = load_event(event_dir)
    if event is None:
        return False
    # Keep events with detected cats
    if event.meta.cat_detected:
        return True
    # Keep events manually tagged
    if event.meta.cat_tag and event.meta.cat_tag != "auto:cat":
        return True
    return False


def _remove_event_dir(event_dir: Path) -> None:
    """Remove an event directory and all its contents."""
    for item in event_dir.rglob("*"):
        if item.is_file():
            item.unlink()
    for subdir in sorted(event_dir.rglob("*"), reverse=True):
        if subdir.is_dir():
            subdir.rmdir()
    event_dir.rmdir()


def prune_events(output_root: Path, days_to_keep: int) -> list[Path]:
    """Prune events older than the retention window.

    Preserves events with cat detections or manual tags.
    Returns a list of removed paths.
    """
    removed: list[Path] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
    if not output_root.exists():
        return removed
    for date_dir in output_root.iterdir():
        if not date_dir.is_dir():
            continue
        try:
            date = datetime.strptime(date_dir.name, DATE_FORMAT).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if date < cutoff:
            for event_dir in date_dir.iterdir():
                if event_dir.is_dir():
                    if _should_keep(event_dir):
                        logger.info("Keeping %s (cat/tagged)", event_dir.name)
                        continue
                    removed.append(event_dir)
                    _remove_event_dir(event_dir)
            if not any(date_dir.iterdir()):
                date_dir.rmdir()
    return removed


def iter_event_images(event_path: Path) -> Iterable[Path]:
    return sorted(event_path.glob("img-*.jpg"))

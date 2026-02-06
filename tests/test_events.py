from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from littercam.events import EventMeta, event_dir_for, load_event, prune_events, write_meta


def test_event_dir_for() -> None:
    ts = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    output_root = Path("/data")
    path = event_dir_for(ts, output_root)
    assert path == Path("/data/2024-01-02/event-20240102-030405")


def test_load_event_roundtrip(tmp_path: Path) -> None:
    event_path = tmp_path / "2024-01-02" / "event-20240102-030405"
    event_path.mkdir(parents=True)
    meta = EventMeta(
        start_ts="2024-01-02T03:04:05Z",
        end_ts="2024-01-02T03:04:25Z",
        trigger_score=42.0,
        image_count=5,
        cat_tag=None,
    )
    write_meta(event_path, meta)
    event = load_event(event_path)
    assert event is not None
    assert event.event_id == "20240102-030405"
    assert event.meta.image_count == 5


def test_prune_events(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    old_date = (now - timedelta(days=10)).strftime("%Y-%m-%d")
    old_event = tmp_path / old_date / "event-20240101-000000"
    old_event.mkdir(parents=True)
    (old_event / "meta.json").write_text("{}")

    keep_date = now.strftime("%Y-%m-%d")
    keep_event = tmp_path / keep_date / "event-20240110-000000"
    keep_event.mkdir(parents=True)
    (keep_event / "meta.json").write_text("{}")

    removed = prune_events(tmp_path, days_to_keep=7)
    removed_paths = [path.name for path in removed]

    assert "event-20240101-000000" in removed_paths
    assert keep_event.exists()

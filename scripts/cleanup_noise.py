#!/usr/bin/env python3
"""One-time cleanup: remove all events without cat detections or manual tags."""
import sys
from pathlib import Path

from littercam.config import load_config
from littercam.events import list_events, _should_keep, _remove_event_dir

cfg = load_config()
events = list_events(cfg.capture.output_root)

keep = []
remove = []
for event in events:
    if _should_keep(event.event_path):
        keep.append(event)
    else:
        remove.append(event)

print(f"Total events: {len(events)}")
print(f"  Keeping:  {len(keep)} (cat detected or tagged)")
print(f"  Removing: {len(remove)} (noise)")

if not remove:
    print("\nNothing to remove.")
    sys.exit(0)

answer = input("\nProceed with deletion? [y/N] ")
if answer.strip().lower() != "y":
    print("Aborted.")
    sys.exit(0)

for event in remove:
    print(f"  Removing {event.event_id}...")
    _remove_event_dir(event.event_path)

# Clean up empty date directories
for date_dir in cfg.capture.output_root.iterdir():
    if date_dir.is_dir() and not any(date_dir.iterdir()):
        date_dir.rmdir()
        print(f"  Removed empty date dir {date_dir.name}")

print(f"\nDone. Removed {len(remove)} events, kept {len(keep)}.")

"""FastAPI web app for LitterCam."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from littercam.config import AppConfig, load_config
from littercam.events import Event, load_event, latest_event, list_events, write_meta


app = FastAPI(title="LitterCam")
config: AppConfig = load_config()

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def format_event_time(event_id: str) -> str:
    """Format event_id like '20260207-184013' to '6:40 PM'."""
    try:
        dt = datetime.strptime(event_id, "%Y%m%d-%H%M%S")
        return dt.strftime("%-I:%M %p")
    except ValueError:
        return event_id


def format_datetime(iso_str: str) -> str:
    """Format ISO datetime to readable format."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %-d, %Y %-I:%M:%S %p")
    except ValueError:
        return iso_str


def format_date_label(date_str: str) -> str:
    """Format date like '2026-02-16' to 'Feb 16'."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%b %-d")
    except ValueError:
        return date_str


TEMPLATES.env.filters["event_time"] = format_event_time
TEMPLATES.env.filters["datetime"] = format_datetime
TEMPLATES.env.filters["date_label"] = format_date_label

app.mount(
    "/data",
    StaticFiles(directory=str(config.capture.output_root), html=False),
    name="data",
)
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static"), html=False),
    name="static",
)


def _event_images(event: Event) -> List[Path]:
    return sorted(event.event_path.glob("img-*.jpg"))


def _event_thumbs(event: Event) -> List[Path]:
    thumbs = sorted(event.event_path.glob("thumb-*.jpg"))
    return thumbs if thumbs else _event_images(event)


def _best_frame_index(event_path: Path) -> Optional[int]:
    """Find the most visually interesting frame by diffing consecutive thumbnails."""
    thumbs = sorted(event_path.glob("thumb-*.jpg"))
    if not thumbs:
        return None
    if len(thumbs) <= 2:
        return 0

    best_idx = 0
    best_score = 0.0
    prev_gray = None

    for i, thumb_path in enumerate(thumbs):
        img = Image.open(thumb_path).convert("L")
        gray = np.array(img, dtype=np.float32)
        if prev_gray is not None:
            score = float(np.mean(np.abs(gray - prev_gray)))
            if score > best_score:
                best_score = score
                best_idx = i
        prev_gray = gray

    return best_idx


def _best_thumb(event: Event) -> Optional[str]:
    """Pick the most visually interesting thumbnail for an event."""
    thumbs = sorted(event.event_path.glob("thumb-*.jpg"))
    if not thumbs:
        images = sorted(event.event_path.glob("img-*.jpg"))
        if not images:
            return None
        idx = _best_frame_index(event.event_path)
        pick = images[idx] if idx is not None else images[0]
        return f"/data/{event.date}/event-{event.event_id}/{pick.name}"
    idx = _best_frame_index(event.event_path)
    pick = thumbs[idx] if idx is not None else thumbs[0]
    return f"/data/{event.date}/event-{event.event_id}/{pick.name}"


def _find_event(event_id: str) -> Optional[Event]:
    for event in list_events(config.capture.output_root):
        if event.event_id == event_id:
            return event
    return None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return RedirectResponse(url="/latest")


@app.get("/latest", response_class=HTMLResponse)
async def latest(request: Request) -> HTMLResponse:
    event = latest_event(config.capture.output_root)
    if not event:
        return TEMPLATES.TemplateResponse(
            "latest.html", {"request": request, "event": None}
        )
    images = _event_images(event)
    first_image = images[0] if images else None
    return TEMPLATES.TemplateResponse(
        "latest.html",
        {"request": request, "event": event, "first_image": first_image},
    )


@app.get("/live", response_class=HTMLResponse)
async def live(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse("live.html", {"request": request})


@app.get("/snapshot")
async def snapshot() -> Response:
    snapshot_path = config.capture.output_root / "snapshot.jpg"
    if not snapshot_path.exists():
        raise HTTPException(status_code=503, detail="No snapshot available — is the capture service running?")
    return Response(content=snapshot_path.read_bytes(), media_type="image/jpeg")


@app.get("/api/status")
async def api_status() -> JSONResponse:
    snapshot_path = config.capture.output_root / "snapshot.jpg"
    capture_ok = False
    if snapshot_path.exists():
        age = time.time() - snapshot_path.stat().st_mtime
        capture_ok = age < 10
    return JSONResponse({"capture": "ok" if capture_ok else "down", "web": "ok"})


@app.get("/highlights", response_class=HTMLResponse)
async def highlights(request: Request) -> HTMLResponse:
    all_events = list_events(config.capture.output_root)
    cards = []
    for event in all_events:
        idx = _best_frame_index(event.event_path)
        if idx is None:
            continue
        images = sorted(event.event_path.glob("img-*.jpg"))
        thumbs = sorted(event.event_path.glob("thumb-*.jpg"))
        if images:
            hero = f"/data/{event.date}/event-{event.event_id}/{images[idx].name}"
        elif thumbs:
            hero = f"/data/{event.date}/event-{event.event_id}/{thumbs[idx].name}"
        else:
            continue
        thumb = f"/data/{event.date}/event-{event.event_id}/{(thumbs or images)[idx].name}"
        cards.append({"event": event, "thumb": thumb, "hero": hero})
    return TEMPLATES.TemplateResponse(
        "highlights.html", {"request": request, "cards": cards}
    )


@app.get("/events", response_class=HTMLResponse)
async def events(request: Request) -> HTMLResponse:
    events = list_events(config.capture.output_root)
    grouped: Dict[str, List[Event]] = {}
    thumbs: Dict[str, str] = {}
    for event in events:
        grouped.setdefault(event.date, []).append(event)
        thumb = _best_thumb(event)
        if thumb:
            thumbs[event.event_id] = thumb
    return TEMPLATES.TemplateResponse(
        "events.html", {"request": request, "grouped": grouped, "thumbs": thumbs}
    )


@app.get("/events/{event_id}", response_class=HTMLResponse)
async def event_detail(request: Request, event_id: str) -> HTMLResponse:
    all_events = list_events(config.capture.output_root)
    event = None
    prev_event = None
    next_event = None
    for i, e in enumerate(all_events):
        if e.event_id == event_id:
            event = e
            # Events are sorted newest-first, so "next" is older (i+1), "prev" is newer (i-1)
            if i > 0:
                next_event = all_events[i - 1]
            if i < len(all_events) - 1:
                prev_event = all_events[i + 1]
            break
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    thumbs = _event_thumbs(event)
    images = _event_images(event)
    return TEMPLATES.TemplateResponse(
        "event_detail.html",
        {
            "request": request,
            "event": event,
            "thumbs": thumbs,
            "images": images,
            "prev_event": prev_event,
            "next_event": next_event,
        },
    )


@app.post("/events/{event_id}/tag")
async def tag_event(event_id: str, cat_tag: str = Form(...)) -> RedirectResponse:
    event = _find_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    event.meta.cat_tag = cat_tag
    write_meta(event.event_path, event.meta)
    return RedirectResponse(url=f"/events/{event_id}", status_code=303)

"""FastAPI web app for LitterCam."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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


TEMPLATES.env.filters["event_time"] = format_event_time
TEMPLATES.env.filters["datetime"] = format_datetime

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


@app.get("/events", response_class=HTMLResponse)
async def events(request: Request) -> HTMLResponse:
    events = list_events(config.capture.output_root)
    grouped: Dict[str, List[Event]] = {}
    for event in events:
        grouped.setdefault(event.date, []).append(event)
    return TEMPLATES.TemplateResponse(
        "events.html", {"request": request, "grouped": grouped}
    )


@app.get("/events/{event_id}", response_class=HTMLResponse)
async def event_detail(request: Request, event_id: str) -> HTMLResponse:
    event = _find_event(event_id)
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

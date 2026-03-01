"""FastAPI app exposing time-sliced sensor data."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from hcultctrl.routes import plants, species, devices, observations, timeseries


app = FastAPI(title="hcultctrl", version="0.1.0")
app.include_router(plants.router)
app.include_router(species.router)
app.include_router(devices.router)
app.include_router(observations.router)
app.include_router(timeseries.router)

logger = logging.getLogger(__name__)

from .routes import plants

_frontend_dir = Path(__file__).resolve().parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/frontend", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


@app.get("/")
def root():
    logger.info("GET /")
    return {"service": "hcultctrl", "status": "ok"}

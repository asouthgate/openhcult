"""FastAPI app exposing time-sliced sensor data."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from hcultctrl.routes import (
    plants,
    species,
    devices,
    observations,
    timeseries,
    plant_sensors,
    water_calibration,
)

app = FastAPI(title="hcultctrl", version="0.1.0")
app.include_router(plants.router)
app.include_router(species.router)
app.include_router(devices.router)
app.include_router(observations.router)
app.include_router(timeseries.router)
app.include_router(plant_sensors.router)
app.include_router(water_calibration.router)


logger = logging.getLogger(__name__)

_frontend_dir = Path(
    os.environ.get(
        "HCULT_FRONTEND_DIR",
        Path(__file__).resolve().parent.parent.parent / "frontend" / "dist",
    )
)
if _frontend_dir.is_dir():
    app.mount(
        "/frontend/app",
        StaticFiles(directory=_frontend_dir, html=True),
        name="frontend",
    )


@app.get("/")
def root():
    return RedirectResponse(url="/frontend/app/#/sensors")


@app.get("/status")
def status():
    return {"service": "hcultctrl", "status": "ok"}

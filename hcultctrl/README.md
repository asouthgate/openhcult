# hcultctrl

Small HTTP API for querying OpenHCult sensor readings from the Pi.

## Run

```bash
python -m hcultctrl
```

Defaults to `http://127.0.0.1:8000`.

## Endpoints

- `GET /timeseries` with query params:
  - `sensor` (optional, e.g. `sensor1`)
  - `device` (optional, device name or BLE address)
  - `start_ms` (optional, epoch ms)
  - `end_ms` (optional, epoch ms)
  - `start_utc` (optional, ISO 8601 UTC, e.g. `2024-01-16T12:00:00Z`)
  - `end_utc` (optional, ISO 8601 UTC, e.g. `2024-01-16T13:00:00Z`)
  - `limit` (optional, default 10000)
  - `format` (`json` or `csv`, default `json`)

Example:

```bash
curl "http://127.0.0.1:8000/timeseries?sensor=sensor1&start_ms=1700000000000&end_ms=1700003600000&format=csv"
```

Or with UTC timestamps:

```bash
curl "http://127.0.0.1:8000/timeseries?sensor=sensor1&start_utc=2026-01-16T12:00:00Z&end_utc=2026-01-16T13:00:00Z"
```

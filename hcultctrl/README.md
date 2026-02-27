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

## Config path

You can pass a config path with `-c`:

```bash
python -m hcultctrl -c /opt/openhcult/openhcult.conf
```

Bind host/port:

```bash
python -m hcultctrl --host 0.0.0.0 --port 8000
```

Or configure defaults in `openhcult.conf`:

```ini
[ctrl]
host=192.168.0.117
port=8000
```

## Run as a system service (systemd)

1) Create the service user:

```bash
sudo useradd --system --home /opt/openhcult --shell /usr/sbin/nologin hcult
```

2) Install the unit file:

```bash
sudo cp /opt/openhcult/hcultctrl/hcultctrl.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hcultctrl.service
```

Logs:

```bash
sudo journalctl -u hcultctrl -f
```

## Logging

By default logs go to `/var/log/hcult/hcultctrl.log`. To override, add to
`openhcult.conf`:

```ini
[logging]
path=/var/log/hcult
```

Or use a specific file:

```ini
[logging]
file=/var/log/hcult/hcultctrl.log
```

To disable file logging and keep stdout only:

```ini
[logging]
file=none
stdout=true
```

To disable stdout logging:

```ini
[logging]
stdout=false
```

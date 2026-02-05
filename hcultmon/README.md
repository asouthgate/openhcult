# hcultmon

BLE advertisement monitor for OpenHCult sensors (no GATT connections).

## Run as a system service (systemd)

1) Create the service user and grant Bluetooth access:

```bash
sudo useradd --system --home /opt/openhcult --shell /usr/sbin/nologin hcult
sudo usermod -aG bluetooth hcult
```

2) Install the unit file:

```bash
sudo cp /opt/openhcult/hcultmon/hcultmon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hcultmon.service
```

Logs:

```bash
sudo journalctl -u hcultmon -f
```

## Config path

You can pass a config path with `-c`:

```bash
python -m hcultmon -c /opt/openhcult/openhcult.conf
```

## Logging

By default logs go to `/var/log/hcult/hcultmon.log`. To override, add to
`openhcult.conf`:

```ini
[logging]
path=/var/log/hcult
```

Or use a specific file:

```ini
[logging]
file=/var/log/hcult/hcultmon.log
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

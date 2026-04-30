# OpenHCult

OpenHCult is robotics system for horticultural applications

## Development

Start everything with one command (postgres, ctrl, frontend, auth):

```
python3 dev.py start
```

This brings up all services in the background, sets up auth, and prints
URLs and credentials. Log files go to `.dev/`.

Run smoke tests against the live services:

```
python3 dev.py test
```

Stop everything:

```
python3 dev.py stop
```

For hot-reload development with live logs, run services individually:

```
python3 dev.py up        # postgres only (one-time, persists)
python3 dev.py ctrl      # ctrl in foreground with --reload
python3 dev.py frontend  # frontend in foreground with HMR
```

Other commands: `status`, `init-db`, `reset-db`. Run `python3 dev.py -h` for details.

## Running tests

Unit tests:

```
bash tests/run_unit_tests.sh
```

Smoke tests:

```
python3 dev.py test    # against running services
python3 dev.py ci      # full CI: build docker, test, teardown
```
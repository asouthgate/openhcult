# OpenHCult

OpenHCult is robotics system for horticultural applications

## Development

Bring up the database (one-time, persists across restarts):

```
python3 dev.py up
```

Start ctrl with hot-reload (separate terminal):

```
python3 dev.py ctrl
```

Start the frontend dev server with HMR (separate terminal):

```
python3 dev.py frontend
```

Check what's running:

```
python3 dev.py status
```

Run smoke tests against the live services (resets the database):

```
python3 dev.py test
```

Tear everything down:

```
python3 dev.py down
```

Other commands: `init-db`, `reset-db`. Run `python3 dev.py -h` for details.

## Running tests

Unit tests:

```
bash tests/run_unit_tests.sh
```

Smoke tests:

```
# Locally (requires postgres + ctrl running)
python3 dev.py test

# Full CI (builds docker, tests, tears down)
python3 dev.py ci
```
# OpenHCult

OpenHCult is robotics system for horticultural applications

## Running tests

To run the unit tests:

```
bash tests/run_unit_tests.sh
```

To run the smoke tests:

```
# Locally (requires ctrl running)
python3 dev.py test

# Full CI (builds docker, tests, tears down)
python3 dev.py ci
```

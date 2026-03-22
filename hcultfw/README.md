# hcultfw (ESP-IDF)

This project is the ESP-IDF port of the original Arduino-style `monitor.ino`.
It reads two ADC channels once at boot, advertises over BLE using a small
manufacturer payload, and goes to deep sleep after the advertising window.

## Requirements

- ESP-IDF v5.x installed
- An ESP32 connected over USB

## Build and Flash

source ~/esp/esp-idf/export.sh

1) Set up the ESP-IDF environment:

```bash
. $IDF_PATH/export.sh
```

2) Configure and build, passing the target board:

**FireBeetle ESP32-E:**
```bash
idf.py set-target esp32
idf.py -DBOARD=FIREBEETLE_ESP32E build
```

**FireBeetle 2 ESP32-C5:**
```bash
idf.py set-target esp32c5
idf.py -DBOARD=FIREBEETLE2_ESP32C5 build
```

3) Flash and monitor:

```bash
idf.py flash monitor
```

If you prefer `make`, there is a small wrapper that calls `idf.py`:

```bash
make build
make flash-monitor
```

## Flashing Notes

- To pick the serial port, pass `-p` to `idf.py`:

```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

- If you're not sure which port your ESP32 is on, unplug/replug and run:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

- Flash without opening the serial monitor:

```bash
idf.py -p /dev/ttyUSB0 flash
```

## Tests

### Host (no hardware required, runs in CI)

Covers pure-logic tests (BLE packet building).

```bash
cd test/host
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### Device (hardware required)

Covers packet tests and ADC range checks on real hardware. Sensors do not need to be connected — ADC reads will return floating values but still within the valid 0–4095 range.

```bash
cd test/device
idf.py set-target esp32          # or esp32c5 for the C5 board
idf.py -DBOARD=FIREBEETLE_ESP32E build flash monitor
```

Unity prints pass/fail per test and a final summary over serial.

## Notes

- Pin assignments are board-specific; see `main/pins.h`.
- Advertising window and sleep interval are set in `openhcult.conf`.

## Glossary

- RTOS (FreeRTOS): A real-time operating system; provides tasks, scheduling, and timers.
- GAP: BLE layer for advertising, discovery, and connection management.
- HCI: Host Controller Interface; command/event link between BLE host and controller.
- NimBLE: Lightweight BLE host stack used by ESP-IDF.
- NVS: Non-Volatile Storage; flash-backed key-value storage used by NimBLE.

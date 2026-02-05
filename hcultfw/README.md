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

2) Configure and build:

```bash
idf.py set-target esp32
idf.py build
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

## Notes

- ADC pins:
  - Sensor 1: GPIO34 (ADC1_CHANNEL_6)
  - Sensor 2: GPIO35 (ADC1_CHANNEL_7)
- Power pins:
  - Sensor power: GPIO5 and GPIO18
- Status LED: GPIO4
- Advertising window: 45s, then deep sleep for 60s.

## Glossary

- RTOS (FreeRTOS): A real-time operating system; provides tasks, scheduling, and timers.
- GAP: BLE layer for advertising, discovery, and connection management.
- HCI: Host Controller Interface; command/event link between BLE host and controller.
- NimBLE: Lightweight BLE host stack used by ESP-IDF.
- NVS: Non-Volatile Storage; flash-backed key-value storage used by NimBLE.

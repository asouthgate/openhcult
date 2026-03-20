"""BLE monitor loop and notification handling for hcultmon."""

import asyncio
import logging
import time

from bleak import BleakScanner
from bleak.exc import BleakDBusError
from hcultdb import queries

DEVICE_NAME_HINT = "ESP32_Sensor"
ADV_COMPANY_ID = 0xFFFF
ADV_MAGIC = b"HC"
ADV_VERSION = 1
ADV_PAYLOAD_LEN = 12

_last_adv_payload = {}


def _parse_adv_payload(data):
    """Parse advertise-only payload: magic(2), version, count, values, nonce."""
    if len(data) < ADV_PAYLOAD_LEN:
        return None
    if data[0:2] != ADV_MAGIC:
        return None
    version = data[2]
    if version != ADV_VERSION:
        logging.warning("Unsupported adv payload version %d", version)
        return None
    sensor_count = data[3]
    if sensor_count == 0:
        return None
    expected_len = 4 + (sensor_count * 2) + 4
    if len(data) < expected_len:
        return None
    values = []
    offset = 4
    for _ in range(sensor_count):
        value = int.from_bytes(data[offset : offset + 2], byteorder="little")
        values.append(value)
        offset += 2
    nonce = int.from_bytes(data[offset : offset + 4], byteorder="little")
    return {"sensor_count": sensor_count, "values": values, "nonce": nonce}


def _handle_adv_payload(payload, dbcon, device):
    """Persist advertise-only payload readings."""
    collection_time_ms = int(time.time() * 1000)
    logging.info(
        "Adv payload from %s: values=%s nonce=%d",
        device.address,
        payload["values"],
        payload["nonce"],
    )
    rows = []
    for i, value in enumerate(payload["values"], start=1):
        rows.append(
            (
                f"sensor{i}",
                value,
                collection_time_ms * 1000,
                collection_time_ms,
                collection_time_ms,
            )
        )
    device_id = queries.register_device(
        dbcon,
        device.name or DEVICE_NAME_HINT,
        device.address,
    )
    queries.write_sensor_readings(dbcon, device_id, rows)


async def run_monitor(db_con):
    """Continuously scan advertisements and store readings."""
    while True:
        logging.info("Starting BLE advertisement scan")
        found_payload = {}
        found_event = asyncio.Event()

        def _on_adv(device, advertisement_data):
            if found_event.is_set():
                return
            mfg_data = advertisement_data.manufacturer_data or {}
            data = mfg_data.get(ADV_COMPANY_ID)
            if not data:
                return
            payload = _parse_adv_payload(data)
            if payload is None:
                return
            found_payload["payload"] = payload
            found_payload["device"] = device
            found_event.set()

        try:
            async with BleakScanner(detection_callback=_on_adv):
                await asyncio.wait_for(found_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logging.info("No awake sensors could be found.")
            continue
        except BleakDBusError as e:
            logging.error(f"BLE scan failed: {e}")
            await asyncio.sleep(10.0)
            continue

        payload = found_payload.get("payload")
        device = found_payload.get("device")
        if payload is None or device is None:
            logging.info("No awake sensors could be found.")
            continue

        logging.info(
            "Found sensor device: %s, %s",
            device.name or DEVICE_NAME_HINT,
            device.address,
        )
        fingerprint = (payload["nonce"], tuple(payload["values"]))
        last = _last_adv_payload.get(device.address)
        if last == fingerprint:
            logging.info("Skipping duplicate payload from %s", device.address)
            continue
        _last_adv_payload[device.address] = fingerprint
        _handle_adv_payload(payload, db_con, device)

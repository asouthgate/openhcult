"""BLE monitor loop and notification handling for hcultmon."""

import asyncio
import logging

from bleak import BleakScanner, BleakClient
from bleak.exc import BleakDeviceNotFoundError, BleakDBusError

from . import config
from . import database

DEVICE_NAME_HINT = "ESP32_Sensor"
PAYLOAD_STRIDE_BYTES = 10
SENSOR_COUNT = 2


def _decode_payload(payload):
    """Decode a single payload block into (sensor, value, timestamp) tuples."""
    readings = []
    sensor_count = len(payload) // PAYLOAD_STRIDE_BYTES
    for i in range(sensor_count):
        offset = i * PAYLOAD_STRIDE_BYTES
        sensor_value = int.from_bytes(
            payload[offset : offset + 2], byteorder="little"
        )
        timestamp_us = int.from_bytes(
            payload[offset + 2 : offset + 10], byteorder="little", signed=False
        )
        readings.append((f"sensor{i + 1}", sensor_value, timestamp_us))
    return readings


def _split_payload_blocks(data):
    """Split a read buffer into one or more payload blocks."""
    block_size = PAYLOAD_STRIDE_BYTES * SENSOR_COUNT
    if block_size == 0 or len(data) % block_size != 0:
        return [data]
    return [
        data[offset : offset + block_size]
        for offset in range(0, len(data), block_size)
    ]


def _notification_handler(sender, data, dbcon, device_id):
    """Decode the payload and persist readings for a device."""
    payload_blocks = _split_payload_blocks(data)
    for payload_index, payload in enumerate(payload_blocks, start=1):
        readings = _decode_payload(payload)
        for sensor_name, sensor_value, timestamp_us in readings:
            logging.info(
                "Received data from %s [payload %d/%d]: %s=%d at %d us",
                sender,
                payload_index,
                len(payload_blocks),
                sensor_name,
                sensor_value,
                timestamp_us,
            )
        database.write_sensor_readings(dbcon, device_id, readings)


async def run_monitor(db_con, characteristic_uuid=None):
    """Continuously scan, connect, request data, and store readings."""
    if characteristic_uuid is None:
        characteristic_uuid = config.get_ble_characteristic_uuid()
    while True:
        logging.info("Starting BLE scan")
        try:
            devices = await BleakScanner.discover(timeout=10.0)
        except BleakDBusError as e:
            logging.error(f"BLE scan failed: {e}")
            await asyncio.sleep(10.0)
            continue
        esp32_device = None
        for d in devices:
            if d.name and DEVICE_NAME_HINT in d.name:
                esp32_device = d
                break

        if esp32_device is None:
            logging.info("No awake sensors could be found.")
            continue

        logging.info(f"Found sensor device: {esp32_device.name}, {esp32_device.address}")

        try:
            async with BleakClient(esp32_device.address) as client:
                if not client.is_connected:
                    logging.info("Failed to connect to the sensor device.")
                    continue
                logging.info("Connected to ESP32 device.")

                device_id = database.register_device(
                    db_con,
                    esp32_device.name or DEVICE_NAME_HINT,
                    esp32_device.address,
                )
                try:
                    data = await client.read_gatt_char(characteristic_uuid)
                except BleakDBusError as e:
                    logging.error(f"Failed to read characteristic: {e}")
                    continue
                _notification_handler(esp32_device.address, data, db_con, device_id)
        except BleakDeviceNotFoundError as e:
            logging.error(f"Device not found error: {e} (device probably went to sleep)")
        except EOFError as e:
            logging.error(f"Connection closed unexpectedly: {e}")

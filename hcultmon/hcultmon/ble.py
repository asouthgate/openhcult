"""BLE monitor loop and notification handling for hcultmon."""

import asyncio
import logging

from bleak import BleakScanner, BleakClient
from bleak.exc import BleakDeviceNotFoundError, BleakDBusError

from . import config
from . import database

DEVICE_NAME_HINT = "ESP32_Sensor"


def _notification_handler(sender, data, dbcon, device_id):
    """Decode the payload and persist readings for a device."""
    readings = []
    stride = 10
    sensor_count = len(data) // stride
    for i in range(sensor_count):
        offset = i * stride
        sensor_value = int.from_bytes(data[offset : offset + 2], byteorder="little")
        timestamp_us = int.from_bytes(
            data[offset + 2 : offset + 10], byteorder="little", signed=False
        )
        readings.append((f"sensor{i + 1}", sensor_value, timestamp_us))
    for sensor_name, sensor_value, timestamp_us in readings:
        logging.info(
            "Received data from %s: %s=%d at %d us",
            sender,
            sensor_name,
            sensor_value,
            timestamp_us,
        )
    database.write_sensor_readings(dbcon, device_id, readings)


async def run_monitor(db_con, characteristic_uuid=None):
    """Continuously scan, connect, request data, and store notifications."""
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

                # await client.start_notify(...) only waits for the subscription to be set up
                # (i.e., CCCD written / notifications enabled).
                # It does not wait for any notification data. The _handler runs
                # later, asynchronously, whenever a notification arrives.
                device_id = database.register_device(
                    db_con,
                    esp32_device.name or DEVICE_NAME_HINT,
                    esp32_device.address,
                )
                notify_event = asyncio.Event()

                def _handler(sender, data):
                    _notification_handler(sender, data, db_con, device_id)
                    notify_event.set()

                await client.start_notify(characteristic_uuid, _handler)
                try:
                    await client.write_gatt_char(
                        characteristic_uuid,
                        "a_message_here".encode("utf-8"),
                        response=False,
                    )
                except BleakDBusError as e:
                    logging.error(f"Failed to write to characteristic: {e}")
                try:
                    await asyncio.wait_for(notify_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logging.warning("Timed out waiting for sensor notification.")
                if client.is_connected:
                    await client.stop_notify(characteristic_uuid)
        except BleakDeviceNotFoundError as e:
            logging.error(f"Device not found error: {e} (device probably went to sleep)")
        except EOFError as e:
            logging.error(f"Connection closed unexpectedly: {e}")

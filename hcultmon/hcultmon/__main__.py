import asyncio
import logging
from bleak import BleakScanner, BleakClient
from bleak.exc import BleakDeviceNotFoundError, BleakDBusError

from . import database

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Use the TX characteristic (the one actually created on ESP32)
CHARACTERISTIC_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"

def notification_handler(sender, data, dbcon, device_id):
    sensor1 = int.from_bytes(data[0:2], byteorder="little")
    sensor2 = int.from_bytes(data[2:4], byteorder="little")
    logging.info(f"Received data from {sender}: Sensor 1: {sensor1}, Sensor 2: {sensor2}")
    readings = {"sensor1": sensor1, "sensor2": sensor2}
    database.write_sensor_readings(dbcon, device_id, readings)

async def _main():
    db_name = "sensor_readings.db"
    db_con = database.setup_db(db_name)
    while True:
        logging.info("Starting BLE scan")
        devices = await BleakScanner.discover(timeout=10.0)
        esp32_device = None
        for d in devices:
            if d.name and "ESP32_Sensor" in d.name:
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
                    esp32_device.name or "ESP32_Sensor",
                    esp32_device.address,
                )
                notify_event = asyncio.Event()

                def _handler(sender, data):
                    notification_handler(sender, data, db_con, device_id)
                    notify_event.set()

                await client.start_notify(CHARACTERISTIC_UUID, _handler)
                try:
                    await client.write_gatt_char(
                        CHARACTERISTIC_UUID,
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
                    await client.stop_notify(CHARACTERISTIC_UUID)
        except BleakDeviceNotFoundError as e:
            logging.error(f"Device not found error: {e} (device probably went to sleep)")

def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()

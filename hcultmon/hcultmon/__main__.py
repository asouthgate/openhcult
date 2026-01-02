import asyncio
import logging
from bleak import BleakScanner, BleakClient

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Use the TX characteristic (the one actually created on ESP32)
CHARACTERISTIC_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"

async def notification_handler(sender, data):
    sensor1 = int.from_bytes(data[0:2], byteorder="little")
    sensor2 = int.from_bytes(data[2:4], byteorder="little")
    logging.info(f"Received data from {sender}: Sensor 1: {sensor1}, Sensor 2: {sensor2}")

async def _main():
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

                await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
                await client.write_gatt_char(CHARACTERISTIC_UUID, "a_message_here".encode('utf-8'))
                await client.stop_notify(CHARACTERISTIC_UUID)
        except bleak.exc.BleakDeviceNotFoundError as e:
            logging.error(f"Device not found error: {e} (device probably went to sleep)")

def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()


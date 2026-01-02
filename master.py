import asyncio
from bleak import BleakScanner, BleakClient

# Use the TX characteristic (the one actually created on ESP32)
SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab"
CHARACTERISTIC_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"

async def notification_handler(sender, data):
    sensor1 = int.from_bytes(data[0:2], byteorder="little")
    sensor2 = int.from_bytes(data[2:4], byteorder="little")
    print(f"Sensor 1: {sensor1}, Sensor 2: {sensor2}")

async def main():
    # Step 1: Scan for devices
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=5.0)
    esp32_device = None
    for d in devices:
        if d.name and "ESP32_Sensor" in d.name:
            esp32_device = d
            break

    if esp32_device is None:
        print("ESP32 not found. Make sure it's advertising.")
        return

    print(f"Connecting to {esp32_device.name} [{esp32_device.address}]...")

    # Step 2: Connect to ESP32
    async with BleakClient(esp32_device.address) as client:
        if not client.is_connected:
            print("Failed to connect!")
            return
        print("Connected!")

        # Step 3: Subscribe to notifications
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        # Step 4: Send messages and receive temperature back
        print("Type any message to request temperature. Type 'exit' to quit.")
        while True:
            msg = input("> ")
            if msg.lower() == "exit":
                break
            await client.write_gatt_char(CHARACTERISTIC_UUID, msg.encode('utf-8'))
            # ESP32 will immediately respond with temperature via notification

        # Stop notifications when done
        await client.stop_notify(CHARACTERISTIC_UUID)

asyncio.run(main())


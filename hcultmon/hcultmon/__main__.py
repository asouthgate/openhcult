import asyncio
import logging

from . import ble
from . import database

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

async def _main():
    db_name = "sensor_readings.db"
    db_con = database.setup_db(db_name)
    await ble.run_monitor(db_con)

def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()

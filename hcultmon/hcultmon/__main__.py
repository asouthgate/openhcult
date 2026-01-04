import asyncio
import logging

from . import ble
from . import config
from . import database

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

async def _main():
    db_path = config.get_db_path()
    db_con = database.setup_db(str(db_path))
    await ble.run_monitor(db_con)

def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()

#pragma once

#include "host/ble_gatt.h"

#include "state.h"

extern struct ble_gatt_svc_def gatt_svcs[];

void ble_init(FirmwareState &state);
void ble_on_reset(int reason);
void ble_on_sync(void);
void ble_host_task(void *param);
void load_ble_uuids(FirmwareState &state);

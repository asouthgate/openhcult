#pragma once

#include "state.h"

void ble_init(FirmwareState &state);
bool init_ble_stack(FirmwareState &state);
void ble_on_reset(int reason);
void ble_on_sync(void);
void ble_host_task(void *param);

#pragma once

/* Shared firmware state referenced across modules. */
#include <stdint.h>

#include "esp_adc/adc_oneshot.h"
#include "host/ble_uuid.h"

extern const char *TAG;

extern ble_uuid128_t g_service_uuid;
extern ble_uuid128_t g_characteristic_uuid;
extern bool g_ble_uuid_ok;

extern uint16_t gatt_chr_handle;
extern int g_sensor_value_1;
extern int g_sensor_value_2;
extern int64_t boot_time_us;
extern adc_oneshot_unit_handle_t adc_handle;
extern uint8_t g_ble_addr_type;
extern volatile bool g_request_sleep;
extern volatile bool g_sent_payload;

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_adc/adc_oneshot.h"
#include "host/ble_uuid.h"

struct FirmwareState {
  ble_uuid128_t service_uuid;
  ble_uuid128_t characteristic_uuid;
  bool ble_uuid_ok;
  uint16_t gatt_chr_handle;
  int sensor_values[2];
  size_t sensor_value_count;
  int64_t boot_time_us;
  adc_oneshot_unit_handle_t adc_handle;
  uint8_t ble_addr_type;
  volatile bool request_sleep;
  volatile bool sent_payload;
};

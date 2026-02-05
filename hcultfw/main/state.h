#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_adc/adc_oneshot.h"
constexpr size_t kSensorCount = 2;

struct FirmwareState {
  int64_t boot_time_us;
  adc_oneshot_unit_handle_t adc_handle;
  uint8_t ble_addr_type;
  volatile bool request_sleep;
  float last_sensor_values[kSensorCount];
};

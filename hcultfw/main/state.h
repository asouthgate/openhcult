#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_attr.h"
#include "esp_adc/adc_oneshot.h"
#include "host/ble_uuid.h"

constexpr size_t kSensorBufferSize = 256;

extern RTC_DATA_ATTR int g_sensor_buffer[kSensorBufferSize];
extern RTC_DATA_ATTR size_t g_sensor_buffer_head;
extern RTC_DATA_ATTR size_t g_sensor_buffer_count;

void push_sensor_measurement(int value);

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

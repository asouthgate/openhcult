#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_adc/adc_oneshot.h"
#include "ble_config.h"

constexpr size_t kSensorBufferSize = SENSOR_BUFFER_SIZE;
constexpr size_t kSensorCount = 2;

extern int g_sensor_buffer[kSensorBufferSize];
extern int64_t g_sensor_time_buffer[kSensorBufferSize];
extern size_t g_sensor_buffer_head;
extern size_t g_sensor_buffer_count;
extern uint32_t g_sleep_cycle_count;

void push_sensor_measurement(int value, int64_t timestamp_us);
size_t copy_latest_measurements_with_time(
  int *values,
  int64_t *times,
  size_t capacity
);
void clear_sensor_buffer();

// TODO: FirmwareState should be separated from sensor buffer logic.
struct FirmwareState {
  int64_t boot_time_us;
  adc_oneshot_unit_handle_t adc_handle;
  uint8_t ble_addr_type;
  volatile bool request_sleep;
};

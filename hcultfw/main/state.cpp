#include "state.h"

// TODO: encapsulate the sensor buffer and give it a user-friendly interface.
RTC_DATA_ATTR int g_sensor_buffer[kSensorBufferSize] = {};
RTC_DATA_ATTR int64_t g_sensor_time_buffer[kSensorBufferSize] = {};
RTC_DATA_ATTR size_t g_sensor_buffer_head = 0;
RTC_DATA_ATTR size_t g_sensor_buffer_count = 0;
// TODO: sleep cycle should be separated
RTC_DATA_ATTR uint32_t g_sleep_cycle_count = 0;

// Push a new sensor measurement into the circular buffer.
void push_sensor_measurement(int value, int64_t timestamp_us) {
  g_sensor_buffer[g_sensor_buffer_head] = value;
  g_sensor_time_buffer[g_sensor_buffer_head] = timestamp_us;
  g_sensor_buffer_head = (g_sensor_buffer_head + 1) % kSensorBufferSize;
  if (g_sensor_buffer_count < kSensorBufferSize) {
    ++g_sensor_buffer_count;
  }
}

size_t copy_latest_measurements_with_time(
  int *values,
  int64_t *times,
  size_t capacity
) {
  if (capacity == 0 || g_sensor_buffer_count == 0) {
    return 0;
  }
  size_t count = g_sensor_buffer_count;
  if (count > capacity) {
    count = capacity;
  }
  size_t start = (g_sensor_buffer_head + kSensorBufferSize - count) % kSensorBufferSize;
  for (size_t i = 0; i < count; ++i) {
    size_t index = (start + i) % kSensorBufferSize;
    values[i] = g_sensor_buffer[index];
    times[i] = g_sensor_time_buffer[index];
  }
  return count;
}

void clear_sensor_buffer() {
  g_sensor_buffer_head = 0;
  g_sensor_buffer_count = 0;
}

#include "payload.h"
#include "state.h"

size_t build_sensor_payload(const int *readings,
                            const int64_t *timestamps_us,
                            size_t reading_count,
                            uint8_t *out, size_t out_capacity) {
  size_t count = reading_count;
  size_t max_count = out_capacity / kSensorPayloadStride;
  if (count > max_count) {
    count = max_count;
  }
  for (size_t i = 0; i < count; ++i) {
    size_t offset = i * kSensorPayloadStride;
    uint16_t value = static_cast<uint16_t>(readings[i]);
    uint64_t timestamp = static_cast<uint64_t>(timestamps_us[i]);
    out[offset + 0] = static_cast<uint8_t>(value & 0xFF);
    out[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
    for (size_t b = 0; b < sizeof(timestamp); ++b) {
      out[offset + 2 + b] =
          static_cast<uint8_t>((timestamp >> (8 * b)) & 0xFF);
    }
  }
  return count * kSensorPayloadStride;
}

size_t get_latest_payload(uint8_t *out, size_t out_capacity) {
  int latest_values[kSensorCount];
  int64_t latest_times[kSensorCount];
  size_t latest_count = copy_latest_measurements_with_time(
      latest_values, latest_times,
      sizeof(latest_values) / sizeof(latest_values[0]));
  return build_sensor_payload(
      latest_values, latest_times, latest_count,
      out, out_capacity);
}

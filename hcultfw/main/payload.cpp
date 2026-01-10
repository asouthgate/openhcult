#include "payload.h"
#include "state.h"

// TODO: replace pointers with something safe
size_t build_sensor_payload(
  const int *readings,
  const int64_t *timestamps_us,
  size_t reading_count,
  uint8_t *out,
  size_t out_capacity
) {
  size_t count = reading_count;
  size_t max_count = out_capacity / kSensorPayloadStride;
  if (count > max_count) {
    count = max_count;
  }
  for (size_t i = 0; i < count; ++i) {
    size_t offset = i * kSensorPayloadStride;
    uint16_t value = static_cast<uint16_t>(readings[i]);
    uint64_t timestamp = static_cast<uint64_t>(timestamps_us[i]);
    // We split into bytes so that we don't have to worry about endianness.
    // So that the payload is portable across different architectures.
    out[offset + 0] = static_cast<uint8_t>(value & 0xFF);
    out[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
    for (size_t b = 0; b < sizeof(timestamp); ++b) {
      out[offset + 2 + b] =
          static_cast<uint8_t>((timestamp >> (8 * b)) & 0xFF);
    }
  }
  return count * kSensorPayloadStride;
}

// TODO: incorrect responsibility; 'i' refers to an index in sensor readings.
//  Instead we should probably build payloads from readings, which are accessed
//  in the correct place to minimize the risks of global state.
size_t get_payload_i(size_t payload_index, uint8_t *out, size_t out_capacity) {
  if (out_capacity < kSensorCount * kSensorPayloadStride) {
    return 0;
  }
  if (g_sensor_buffer_count == 0) {
    return 0;
  }
  size_t full_count = (g_sensor_buffer_count / kSensorCount) * kSensorCount;
  if (full_count == 0) {
    return 0;
  }
  size_t payload_count = full_count / kSensorCount;
  if (payload_index >= payload_count) {
    return 0;
  }

  // TODO: this logic is required because ring buffer size is not necessarily
  //  a multiple of the sensor count. This seems like it should be removed.
  //  It's problem solving around something that should be fixed.
  size_t drop = g_sensor_buffer_count - full_count;
  size_t oldest =
      (g_sensor_buffer_head + kSensorBufferSize - g_sensor_buffer_count) %
      kSensorBufferSize;
  size_t start = (oldest + drop) % kSensorBufferSize;
  size_t base = (start + payload_index * kSensorCount) % kSensorBufferSize;

  int values[kSensorCount];
  int64_t times[kSensorCount];
  for (size_t i = 0; i < kSensorCount; ++i) {
    size_t index = (base + i) % kSensorBufferSize;
    values[i] = g_sensor_buffer[index];
    times[i] = g_sensor_time_buffer[index];
  }

  return build_sensor_payload(values, times, kSensorCount, out, out_capacity);
}

size_t build_payload_header(
  uint16_t payload_count,
  uint8_t *out,
  size_t out_capacity
) {
  if (out_capacity < kPayloadHeaderSize) {
    return 0;
  }
  out[0] = kPayloadHeaderMagic0;
  out[1] = kPayloadHeaderMagic1;
  out[2] = kPayloadHeaderVersion;
  out[3] = static_cast<uint8_t>(kSensorCount & 0xFF);
  out[4] = static_cast<uint8_t>(payload_count & 0xFF);
  out[5] = static_cast<uint8_t>((payload_count >> 8) & 0xFF);
  uint16_t stride = static_cast<uint16_t>(kSensorPayloadStride);
  out[6] = static_cast<uint8_t>(stride & 0xFF);
  out[7] = static_cast<uint8_t>((stride >> 8) & 0xFF);
  return kPayloadHeaderSize;
}

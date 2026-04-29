#pragma once

#include <stddef.h>
#include <stdint.h>

// Returns false if out_capacity is too small.
bool build_adv_mfg_data(
  uint8_t *out,
  size_t out_capacity,
  size_t *out_len,
  const int *sensor_raw,
  const uint16_t *sensor_mv,
  uint8_t sensor_count,
  uint32_t reading_token
);

// Required buffer size for a given sensor count.
constexpr size_t adv_mfg_data_size(uint8_t sensor_count) {
  return 2 + 4 + (size_t)sensor_count * 4 + 4;
}

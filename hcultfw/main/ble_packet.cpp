#include "ble_packet.h"

static constexpr uint16_t kCompanyId = 0xFFFF;
static constexpr uint8_t kMagic0 = 'H';
static constexpr uint8_t kMagic1 = 'C';
static constexpr uint8_t kVersion = 2;

bool build_adv_mfg_data(
  uint8_t *out,
  size_t out_capacity,
  size_t *out_len,
  const int *sensor_raw,
  const uint16_t *sensor_mv,
  uint8_t sensor_count,
  uint32_t reading_token
) {
  size_t required = adv_mfg_data_size(sensor_count);
  if (out_capacity < required) return false;

  out[0] = static_cast<uint8_t>(kCompanyId & 0xFF);
  out[1] = static_cast<uint8_t>((kCompanyId >> 8) & 0xFF);
  out[2] = kMagic0;
  out[3] = kMagic1;
  out[4] = kVersion;
  out[5] = sensor_count;

  size_t offset = 6;
  for (uint8_t i = 0; i < sensor_count; ++i) {
    uint16_t raw = static_cast<uint16_t>(sensor_raw[i]);
    uint16_t mv = sensor_mv[i];
    out[offset++] = static_cast<uint8_t>(raw & 0xFF);
    out[offset++] = static_cast<uint8_t>((raw >> 8) & 0xFF);
    out[offset++] = static_cast<uint8_t>(mv & 0xFF);
    out[offset++] = static_cast<uint8_t>((mv >> 8) & 0xFF);
  }

  out[offset++] = static_cast<uint8_t>(reading_token & 0xFF);
  out[offset++] = static_cast<uint8_t>((reading_token >> 8) & 0xFF);
  out[offset++] = static_cast<uint8_t>((reading_token >> 16) & 0xFF);
  out[offset++] = static_cast<uint8_t>((reading_token >> 24) & 0xFF);

  *out_len = required;
  return true;
}

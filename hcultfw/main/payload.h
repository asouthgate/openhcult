#pragma once

#include <stddef.h>
#include <stdint.h>

// Each sensor reading consists of a 2-byte value and an 8-byte timestamp.
constexpr size_t kSensorPayloadStride = sizeof(uint16_t) + sizeof(int64_t);
constexpr size_t kPayloadHeaderSize = 8;
// Magic bytes are used to identify valid payload headers.
constexpr uint8_t kPayloadHeaderMagic0 = 'H';
constexpr uint8_t kPayloadHeaderMagic1 = 'C';
constexpr uint8_t kPayloadHeaderVersion = 1;

size_t build_sensor_payload(
    const int *readings,
    const int64_t *timestamps_us,
    size_t reading_count,
    uint8_t *out, size_t out_capacity
);

size_t get_payload_i(size_t payload_index, uint8_t *out, size_t out_capacity);
size_t build_payload_header(
    uint16_t payload_count,
    uint8_t *out,
    size_t out_capacity
);

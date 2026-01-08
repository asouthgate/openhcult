#pragma once

#include <stddef.h>
#include <stdint.h>

constexpr size_t kSensorPayloadStride =
    sizeof(uint16_t) + sizeof(int64_t);

size_t build_sensor_payload(const int *readings,
                            const int64_t *timestamps_us,
                            size_t reading_count,
                            uint8_t *out, size_t out_capacity);

size_t get_latest_payload(uint8_t *out, size_t out_capacity);

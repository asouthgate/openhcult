#pragma once

#include <stddef.h>
#include <stdint.h>

size_t build_sensor_payload(const int *readings, size_t reading_count,
                            uint16_t *out, size_t out_capacity);

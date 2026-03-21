#pragma once

#include "esp_adc/adc_oneshot.h"
#include "state.h"

struct SensorReading {
  int raw;
  uint16_t voltage_mv;
};

SensorReading read_sensor(FirmwareState &state, adc_channel_t channel);

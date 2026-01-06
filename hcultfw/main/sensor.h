#pragma once

#include "esp_adc/adc_oneshot.h"

#include "state.h"

int read_sensor(FirmwareState *state, adc_channel_t channel);

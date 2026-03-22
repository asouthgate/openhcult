#pragma once

#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_oneshot.h"
#include "state.h"

constexpr adc_atten_t kAdcAtten = ADC_ATTEN_DB_12;
constexpr adc_bitwidth_t kAdcBitwidth = ADC_BITWIDTH_12;

struct SensorReading {
  int raw;
  uint16_t voltage_mv;
};

adc_cali_handle_t create_cali_handle(adc_channel_t channel);
SensorReading read_sensor(FirmwareState &state, adc_channel_t channel, adc_cali_handle_t cali);

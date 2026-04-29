#include "esp_log.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "pins.h"
#include "sensor.h"

static const char *TAG = "hcultfw";

void init_adc(FirmwareState &state) {
  adc_oneshot_unit_init_cfg_t unit_cfg = {};
  unit_cfg.unit_id = ADC_UNIT_1;
  ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_cfg, &state.adc_handle));

  const gpio_num_t sensor_gpios[kSensorCount] = {SENSOR_PIN_1, SENSOR_PIN_2};
  adc_oneshot_chan_cfg_t chan_cfg = {};
  chan_cfg.atten = kAdcAtten;
  chan_cfg.bitwidth = kAdcBitwidth;
  for (size_t i = 0; i < kSensorCount; ++i) {
    adc_unit_t unit;
    ESP_ERROR_CHECK(adc_oneshot_io_to_channel(sensor_gpios[i], &unit, &state.sensor_channels[i]));
    ESP_ERROR_CHECK(adc_oneshot_config_channel(state.adc_handle, state.sensor_channels[i], &chan_cfg));
    state.cali_handles[i] = create_cali_handle(state.sensor_channels[i]);
  }
}

adc_cali_handle_t create_cali_handle(adc_channel_t channel) {
  adc_cali_handle_t handle = nullptr;
#if defined(BOARD_FIREBEETLE2_ESP32C5)
  adc_cali_curve_fitting_config_t cfg = {
    .unit_id = ADC_UNIT_1,
    .chan = channel,
    .atten = kAdcAtten,
    .bitwidth = kAdcBitwidth,
  };
  adc_cali_create_scheme_curve_fitting(&cfg, &handle);
#else
  adc_cali_line_fitting_config_t cfg = {
    .unit_id = ADC_UNIT_1,
    .atten = kAdcAtten,
    .bitwidth = kAdcBitwidth,
    .default_vref = 1100,
  };
  adc_cali_create_scheme_line_fitting(&cfg, &handle);
#endif
  return handle;
}

SensorReading read_sensor(FirmwareState &state, adc_channel_t channel, adc_cali_handle_t cali) {
  constexpr int kDiscardSamples = 3;
  for (int i = 0; i < kDiscardSamples; ++i) {
    int raw = 0;
    esp_err_t rc = adc_oneshot_read(state.adc_handle, channel, &raw);
    if (rc != ESP_OK) {
      ESP_LOGW(TAG, "ADC read failed (discard): %d", rc);
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }

  const int nSamples = 10;
  int sum = 0;
  for (int i = 0; i < nSamples; ++i) {
    int raw = 0;
    esp_err_t rc = adc_oneshot_read(state.adc_handle, channel, &raw);
    if (rc != ESP_OK) {
      ESP_LOGW(TAG, "ADC read failed: %d", rc);
    } else {
      sum += raw;
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  int avg_raw = sum / nSamples;

  int voltage_mv = 0;
  if (cali) {
    adc_cali_raw_to_voltage(cali, avg_raw, &voltage_mv);
  }

  return {avg_raw, static_cast<uint16_t>(voltage_mv)};
}

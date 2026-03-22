#include "esp_log.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "sensor.h"

static const char *TAG = "hcultfw";

adc_cali_handle_t create_cali_handle(adc_channel_t channel) {
  adc_cali_handle_t handle = nullptr;
#if defined(BOARD_FIREBEETLE2_ESP32C5)
  adc_cali_curve_fitting_config_t cfg = {
    .atten = kAdcAtten,
    .bitwidth = kAdcBitwidth,
    .chan = channel,
    .unit_id = ADC_UNIT_1,
  };
  adc_cali_create_scheme_curve_fitting(&cfg, &handle);
#else
  adc_cali_line_fitting_config_t cfg = {
    .atten = kAdcAtten,
    .bitwidth = kAdcBitwidth,
    .default_vref = 1100,
    .unit_id = ADC_UNIT_1,
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

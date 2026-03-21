#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "sensor.h"
#include "esp_log.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

static const char *TAG = "hcultfw";

static adc_cali_handle_t _create_cali_handle(adc_channel_t channel) {
  adc_cali_handle_t handle = nullptr;
#if defined(BOARD_FIREBEETLE2_ESP32C5)
  adc_cali_curve_fitting_config_t cfg = {
    .unit_id = ADC_UNIT_1, .chan = channel,
    .atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_12,
  };
  adc_cali_create_scheme_curve_fitting(&cfg, &handle);
#else
  adc_cali_line_fitting_config_t cfg = {
    .unit_id = ADC_UNIT_1, .atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_12,
  };
  adc_cali_create_scheme_line_fitting(&cfg, &handle);
#endif
  return handle;
}

static void _delete_cali_handle(adc_cali_handle_t handle) {
  if (!handle) return;
#if defined(BOARD_FIREBEETLE2_ESP32C5)
  adc_cali_delete_scheme_curve_fitting(handle);
#else
  adc_cali_delete_scheme_line_fitting(handle);
#endif
}

SensorReading read_sensor(FirmwareState &state, adc_channel_t channel) {
  constexpr int kDiscardSamples = 3;
  for (int i = 0; i < kDiscardSamples; ++i) {
    int raw = 0;
    esp_err_t rc = adc_oneshot_read(state.adc_handle, channel, &raw);
    if (rc != ESP_OK) {
      ESP_LOGW(TAG, "ADC read failed (discard): %d", rc);
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }

  int sum = 0;
  for (int i = 0; i < 10; ++i) {
    int raw = 0;
    esp_err_t rc = adc_oneshot_read(state.adc_handle, channel, &raw);
    if (rc != ESP_OK) {
      ESP_LOGW(TAG, "ADC read failed: %d", rc);
    } else {
      sum += raw;
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  int avg_raw = sum / 10;

  int voltage_mv = 0;
  adc_cali_handle_t cali = _create_cali_handle(channel);
  if (cali) {
    adc_cali_raw_to_voltage(cali, avg_raw, &voltage_mv);
    _delete_cali_handle(cali);
  }

  return {avg_raw, static_cast<uint16_t>(voltage_mv)};
}

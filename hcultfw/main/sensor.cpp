#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "sensor.h"
#include "esp_log.h"

static const char *TAG = "hcultfw";

// Read the specified ADC channel multiple times and return the average.
int read_sensor(FirmwareState &state, adc_channel_t channel) {
  // Discard a few startup samples to let the sensor/ADC settle after power-on.
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
  return sum / 10;
}

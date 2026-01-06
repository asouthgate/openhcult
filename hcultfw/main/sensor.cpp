#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "sensor.h"
#include "esp_log.h"

static const char *TAG = "hcultfw";

/*
 * Reads a single ADC channel using multiple averaged samples.
 * This favors stable readings over speed because we only sample once per boot.
 */
int read_sensor(FirmwareState &state, adc_channel_t channel) {
  int sum = 0;
  for (int i = 0; i < 10; ++i) {
    int raw = 0;
    // Why oneshot? ESP-IDF also has a continuous
    // ADC driver, but it is heavier than needed here.
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

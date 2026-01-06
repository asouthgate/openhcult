/* Task creation and scheduling APIs built on top of FreeRTOS core types. */
// Q: tasks? so multithreading?
// A: FreeRTOS tasks are lightweight threads scheduled by the RTOS.
// A: On ESP32 they can run across the two cores, but they are still RTOS tasks.
#include "freertos/FreeRTOS.h"
/* Task creation and scheduling APIs built on top of FreeRTOS core types. */
#include "freertos/task.h"

#include "globals.h"
#include "sensor.h"
#include "esp_log.h"

/*
 * Reads a single ADC channel using multiple samples.
 * This favors stable readings over speed because we only sample once per boot.
 */
int read_sensor(adc_channel_t channel) {
  int sum = 0;
  for (int i = 0; i < 10; ++i) {
    int raw = 0;
    // Q: why do we use oneshot? We average manually, does that mean there is another function that could do the averaging for us?
    // A: Oneshot is simplest for infrequent reads. ESP-IDF also has a continuous
    // A: ADC driver, but it is heavier than needed here. Averaging is done manually
    // A: because the oneshot driver returns raw samples only.
    esp_err_t rc = adc_oneshot_read(adc_handle, channel, &raw);
    if (rc != ESP_OK) {
      ESP_LOGW(TAG, "ADC read failed: %d", rc);
    } else {
      sum += raw;
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  // Average multiple samples to smooth noise without extra DSP work.
  return sum / 10;
}

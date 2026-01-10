/* Task creation and scheduling APIs built on top of FreeRTOS core types. */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_timer.h"

#include "pins.h"
#include "sleep.h"
#include "ble_config.h"
#include "state.h"

static const char *TAG = "hcultfw";

void sleep_task(void *param) {
  FirmwareState &state = *static_cast<FirmwareState *>(param);
  // while(true) does get exited by the deep sleep call, so this 
  // is not an infinite loop.
  while (true) {
    bool should_sleep = state.request_sleep;
    int64_t elapsed_us = esp_timer_get_time() - state.boot_time_us;
    // If we've been advertising for too long, go to sleep
    if (elapsed_us > BLE_ADVERTISING_TIME_MS * 1000LL) {
      ESP_LOGI(TAG, "BLE window expired, sleeping");
      should_sleep = true;
    }

    if (should_sleep) {
      // Power-gate sensors and LED before deep sleep to minimize quiescent draw.
      gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 0);
      gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_1), 0);
      gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_2), 0);
      esp_deep_sleep(SLEEP_TIME_US);
    }
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

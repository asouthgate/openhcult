#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "ble_config.h"
#include "pins.h"
#include "sleep.h"
#include "state.h"

static const char *TAG = "hcultfw";

void sleep_task(void *param) {
  FirmwareState &state = *static_cast<FirmwareState *>(param);
  // Deep sleep causes this to end: not infinite loop
  while (true) {
    bool should_sleep = state.request_sleep;
    int64_t elapsed_us = esp_timer_get_time() - state.boot_time_us;

    if (elapsed_us > BLE_ADVERTISING_TIME_MS * 1000LL) {
      ESP_LOGI(TAG, "BLE window expired, sleeping");
      should_sleep = true;
    }

    if (should_sleep) {
      // TODO: consider encapsulation of this 'turn everything off'
      gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 0);
      gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_1), 0);
      gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_2), 0);
      uint32_t sleep_s = static_cast<uint32_t>((SLEEP_TIME_US + 500000ULL) / 1000000ULL);
      g_uptime_s += sleep_s;
      esp_deep_sleep(SLEEP_TIME_US);
    }
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

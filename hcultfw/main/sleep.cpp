/* Task creation and scheduling APIs built on top of FreeRTOS core types. */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* GPIO driver for pin direction and level control. */
#include "driver/gpio.h"
/* ESP-IDF logging macros (ESP_LOGI/W/E). */
#include "esp_log.h"
/* Deep sleep APIs for low-power operation. */
#include "esp_sleep.h"
/* High-resolution timer for microsecond timestamps. */
#include "esp_timer.h"

#include "globals.h"
#include "pins.h"
#include "sleep.h"
#include "ble_config.h"

/*
 * Background task that decides when to enter deep sleep.
 * It sleeps either when the fixed advertising window expires or once data has
 * been successfully notified to a client.
 */
void sleep_task(void *param) {
  // Q: this is while (true): are we always in this loop then? How do we ever exit?
  // A: Yes, the task loops forever. Deep sleep stops the CPU and resets on wake,
  // A: so the loop never returns; the chip restarts instead.
  while (true) {
    bool should_sleep = g_request_sleep;
    int64_t elapsed_us = esp_timer_get_time() - boot_time_us;
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

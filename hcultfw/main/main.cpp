#include <stdint.h>

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_random.h"
#include "esp_bt.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "ble.h"
#include "ble_config.h"
#include "pins.h"
#include "sensor.h"
#include "sleep.h"
#include "state.h"

static const char *TAG = "hcultfw"; // For logging

static void init_power_pins() {
  gpio_config_t io_conf = {};
  io_conf.intr_type = GPIO_INTR_DISABLE; // No interrupts?
  io_conf.mode = GPIO_MODE_OUTPUT;
  // pin_bit_mask specifies which pins this configuration struct applies to.
  io_conf.pin_bit_mask = (1ULL << LED_PIN) |
                        (1ULL << SENSOR_POWER_PIN_1) |
                        (1ULL << SENSOR_POWER_PIN_2);
  // TODO: internal pulldown/pullup: should these be disabled?
  io_conf.pull_down_en = GPIO_PULLDOWN_ENABLE;
  io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
  ESP_ERROR_CHECK(gpio_config(&io_conf));

  gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 0);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_1), 0);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_2), 0);
}

// This is probably unnecessary superstition, we only use BLE
// Apparently makes a difference to memory and is 'defensive'
static void disable_unused_radios() {
  esp_err_t err = esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT);
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESP_LOGW(TAG, "Failed to release BT classic memory: %d", err);
  }
  err = esp_wifi_stop();
  if (err != ESP_OK && err != ESP_ERR_WIFI_NOT_INIT) {
    ESP_LOGW(TAG, "Failed to stop Wi-Fi: %d", err);
  }
  err = esp_wifi_deinit();
  if (err != ESP_OK && err != ESP_ERR_WIFI_NOT_INIT) {
    ESP_LOGW(TAG, "Failed to deinit Wi-Fi: %d", err);
  }
}

// We need this for PHY calibration
static void init_nvs_storage() {
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ESP_ERROR_CHECK(nvs_flash_init());
  } else {
    ESP_ERROR_CHECK(ret);
  }
}

static void take_sensor_readings(FirmwareState &state) {
  const gpio_num_t sensor_power_pins[kSensorCount] = {
      static_cast<gpio_num_t>(SENSOR_POWER_PIN_1),
      static_cast<gpio_num_t>(SENSOR_POWER_PIN_2),
  };
  SensorReading sensor_readings[kSensorCount];
  for (size_t i = 0; i < kSensorCount; ++i) {
    gpio_set_level(sensor_power_pins[i], 1);
    vTaskDelay(pdMS_TO_TICKS(100));
    sensor_readings[i] = read_sensor(state, state.sensor_channels[i], state.cali_handles[i]);
    gpio_set_level(sensor_power_pins[i], 0);
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
  for (size_t i = 0; i < kSensorCount; ++i) {
    state.last_sensor_values[i] = sensor_readings[i].raw;
    state.last_sensor_voltages_mv[i] = sensor_readings[i].voltage_mv;
  }
  state.reading_token = esp_random();

  ESP_LOGI(TAG, "Sensor 1: raw=%d voltage=%umV", sensor_readings[0].raw, sensor_readings[0].voltage_mv);
  ESP_LOGI(TAG, "Sensor 2: raw=%d voltage=%umV", sensor_readings[1].raw, sensor_readings[1].voltage_mv);
  ESP_LOGI(TAG, "Sensor token: %u", state.reading_token);
}

// Turn off LEDs and sensor power pins, then enter deep sleep.
static void sleep_now() {
  gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 0);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_1), 0);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_2), 0);
  uint32_t sleep_s = static_cast<uint32_t>((SLEEP_TIME_US + 500000ULL) / 1000000ULL);
  g_uptime_s += sleep_s;
  esp_deep_sleep(SLEEP_TIME_US);
}

extern "C" void app_main(void) {
  // TODO: split up global state
  static FirmwareState state = {};
  state.boot_time_us = esp_timer_get_time();

  init_nvs_storage();
  disable_unused_radios();
  init_power_pins();
  init_adc(state);
  if (RED_LED_FLASH_MS > 0) {
    gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 1);
    vTaskDelay(pdMS_TO_TICKS(RED_LED_FLASH_MS));
    gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 0);
  }
  take_sensor_readings(state);

  if (!init_ble_stack(state)) {
    return;
  }

  // Give the supply rail a recovery window before BLE starts.
  vTaskDelay(pdMS_TO_TICKS(1000));

  // Start a new FreeRTOS task that runs in parallel with app_main.
  // We need this because the BLE stack requires its own event loop to function properly.
  // Going into deep sleep immediately after starting the BLE stack would prevent it from operating.
  xTaskCreate(sleep_task, "sleep_task", 2048, &state, 5, nullptr);
}

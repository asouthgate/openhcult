#include <stdint.h>

// OS and system includes
#include "freertos/FreeRTOS.h" //FreeRTOS is the OS for the ESP32, needed for tasks and delays.
#include "freertos/task.h"
#include "driver/gpio.h" // For pin controls
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_adc/adc_oneshot.h" // For ADC readings
#include "esp_bt.h"
#include "esp_pm.h"
#include "esp_wifi.h"

// Bluetooth includes
#include "ble.h"
#include "ble_config.h"
#include "pins.h"
#include "sensor.h"
#include "sleep.h"
#include "state.h"

static const char *TAG = "hcultfw"; // Tag for logging

static adc_channel_t adc_channel_for_gpio(gpio_num_t gpio) {
  switch (gpio) {
    case GPIO_NUM_36:
      return ADC_CHANNEL_0;
    case GPIO_NUM_37:
      return ADC_CHANNEL_1;
    case GPIO_NUM_38:
      return ADC_CHANNEL_2;
    case GPIO_NUM_39:
      return ADC_CHANNEL_3;
    case GPIO_NUM_32:
      return ADC_CHANNEL_4;
    case GPIO_NUM_33:
      return ADC_CHANNEL_5;
    case GPIO_NUM_34:
      return ADC_CHANNEL_6;
    case GPIO_NUM_35:
      return ADC_CHANNEL_7;
    default:
      ESP_LOGE(TAG, "Unsupported ADC GPIO: %d", static_cast<int>(gpio));
      return ADC_CHANNEL_0;
  }
}

// Configure LED and sensor power GPIOs as outputs and set safe defaults.
static void init_power_pins() {
  gpio_config_t io_conf = {};
  io_conf.intr_type = GPIO_INTR_DISABLE; // No interrupts
  io_conf.mode = GPIO_MODE_OUTPUT; // Set as output pins
  // pin_bit_mask specifies which pins this configuration struct applies to.
  io_conf.pin_bit_mask = (1ULL << LED_PIN) | (1ULL << SENSOR_POWER_PIN_1) |
                         (1ULL << SENSOR_POWER_PIN_2) | (1ULL << RED_LED_PIN);
  io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
  io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
  ESP_ERROR_CHECK(gpio_config(&io_conf));

  gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 0);
  gpio_set_level(static_cast<gpio_num_t>(RED_LED_PIN), 0);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_1), 0);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_2), 0);
}

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

static void reduce_cpu_peak_draw() {
  esp_pm_config_esp32_t cfg = {};
  cfg.max_freq_mhz = 80;
  cfg.min_freq_mhz = 40;
  cfg.light_sleep_enable = true;
  esp_err_t err = esp_pm_configure(&cfg);
  if (err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to configure power management: %d", err);
  }
}

// Power on or off a sensor connected to the given GPIO pin.
static void power_sensor(gpio_num_t pin, bool on) {
  gpio_set_level(pin, on ? 1 : 0);
}

// Read a sensor value from the specified ADC channel.
static float read_sensor_with_power(
  FirmwareState &state,
  gpio_num_t power_pin,
  adc_channel_t channel
) {
  // We power on the sensor, wait briefly for it to stabilize, read the value, then power it off.
  // This reduces artifacts in the readings.
  power_sensor(power_pin, true);
  vTaskDelay(pdMS_TO_TICKS(100));
  int value = read_sensor(state, channel);
  power_sensor(power_pin, false);
  return static_cast<float>(value);
}

// Take readings from all sensors and log the values.
static void take_sensor_readings(FirmwareState &state) {
  adc_oneshot_unit_init_cfg_t unit_cfg = {};
  unit_cfg.unit_id = ADC_UNIT_1;
  ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_cfg, &state.adc_handle));

  adc_oneshot_chan_cfg_t chan_cfg = {};
  chan_cfg.atten = ADC_ATTEN_DB_12;
  chan_cfg.bitwidth = ADC_BITWIDTH_12;
  const adc_channel_t sensor_channels[kSensorCount] = {
      adc_channel_for_gpio(SENSOR_PIN_1),
      adc_channel_for_gpio(SENSOR_PIN_2),
  };
  ESP_ERROR_CHECK(
      adc_oneshot_config_channel(state.adc_handle, sensor_channels[0], &chan_cfg));
  ESP_ERROR_CHECK(
      adc_oneshot_config_channel(state.adc_handle, sensor_channels[1], &chan_cfg));

  // Enable the red pin for debugging purposes and to force power draw to prevent battery from
  // going to sleep. Power banks sometimes cut power to the output if the draw is too low.
  gpio_set_level(static_cast<gpio_num_t>(RED_LED_PIN), 1);
  vTaskDelay(pdMS_TO_TICKS(RED_LED_FLASH_MS));
  gpio_set_level(static_cast<gpio_num_t>(RED_LED_PIN), 0);
  const gpio_num_t sensor_power_pins[kSensorCount] = {
      static_cast<gpio_num_t>(SENSOR_POWER_PIN_1),
      static_cast<gpio_num_t>(SENSOR_POWER_PIN_2),
  };
  float sensor_values[kSensorCount];
  for (size_t i = 0; i < kSensorCount; ++i) {
    sensor_values[i] = read_sensor_with_power(
      state,
      sensor_power_pins[i],
      sensor_channels[i]
    );
    vTaskDelay(pdMS_TO_TICKS(100));
  }
  for (size_t i = 0; i < kSensorCount; ++i) {
    state.last_sensor_values[i] = sensor_values[i];
  }

  ESP_LOGI(TAG, "Sensor value 1: %.2f", sensor_values[0]);
  ESP_LOGI(TAG, "Sensor value 2: %.2f", sensor_values[1]);
}

// Turn off LEDs and sensor power pins, then enter deep sleep.
static void sleep_now() {
  gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 0);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_1), 0);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_2), 0);
  esp_deep_sleep(SLEEP_TIME_US);
}

extern "C" void app_main(void) {
  // TODO: remove global
  static FirmwareState state = {};
  state.boot_time_us = esp_timer_get_time();
  
  disable_unused_radios();
  reduce_cpu_peak_draw();
  init_power_pins();
  gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 1);
  vTaskDelay(pdMS_TO_TICKS(50));
  gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 0);
  take_sensor_readings(state);

  if (!init_ble_stack(state)) {
    return;
  }

  // Give the supply rail a short recovery window before BLE starts.
  vTaskDelay(pdMS_TO_TICKS(250));

  // Start a new FreeRTOS task that runs in parallel with app_main.
  // We need this because the BLE stack requires its own event loop to function properly.
  // Going into deep sleep immediately after starting the BLE stack would prevent it from operating.
  xTaskCreate(sleep_task, "sleep_task", 2048, &state, 5, nullptr);
}

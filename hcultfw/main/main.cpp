#include <stdint.h>

// OS and system includes
#include "freertos/FreeRTOS.h" //FreeRTOS is the OS for the ESP32, needed for tasks and delays.
#include "freertos/task.h"
#include "driver/gpio.h" // For pin controls
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "esp_rtc_time.h"
#include "nvs_flash.h" // NVS (Non-Volatile Storage) is a small key-value store in flash.
#include "esp_adc/adc_oneshot.h" // For ADC readings

// Bluetooth includes
#include "esp_nimble_hci.h" // NimBLE is BLE host stack, this include is required for initialization.
#include "nimble/nimble_port.h" 
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h" // Core BLE host types/config
#include "services/gap/ble_svc_gap.h" // Helpers for GAP
#include "services/gatt/ble_svc_gatt.h" // Helpers for GATT

#include "ble.h"
#include "ble_config.h"
#include "pins.h"
#include "sensor.h"
#include "sleep.h"
#include "state.h"

static const char *TAG = "hcultfw"; // Tag for logging

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

  gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 1);
  gpio_set_level(static_cast<gpio_num_t>(RED_LED_PIN), 0);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_1), 0);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_2), 0);
}

// Power on or off a sensor connected to the given GPIO pin.
static void power_sensor(gpio_num_t pin, bool on) {
  gpio_set_level(pin, on ? 1 : 0);
}

// Read a sensor value from the specified ADC channel.
static int read_sensor_with_power(
  FirmwareState &state,
  gpio_num_t power_pin,
  adc_channel_t channel
) {
  // We power on the sensor, wait briefly for it to stabilize, read the value, then power it off.
  // This reduces artifacts in the readings.
  power_sensor(power_pin, true);
  vTaskDelay(pdMS_TO_TICKS(50));
  int value = read_sensor(state, channel);
  power_sensor(power_pin, false);
  return value;
}

// Take readings from all sensors and log the values.
static void take_sensor_readings(FirmwareState &state) {
  adc_oneshot_unit_init_cfg_t unit_cfg = {};
  unit_cfg.unit_id = ADC_UNIT_1;
  ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_cfg, &state.adc_handle));

  adc_oneshot_chan_cfg_t chan_cfg = {};
  chan_cfg.atten = ADC_ATTEN_DB_12;
  chan_cfg.bitwidth = ADC_BITWIDTH_12;
  ESP_ERROR_CHECK(adc_oneshot_config_channel(state.adc_handle, ADC_CHANNEL_6, &chan_cfg));
  ESP_ERROR_CHECK(adc_oneshot_config_channel(state.adc_handle, ADC_CHANNEL_7, &chan_cfg));

  // Enable the red pin for debugging purposes and to force power draw to prevent battery from
  // going to sleep. Power banks sometimes cut power to the output if the draw is too low.
  gpio_set_level(static_cast<gpio_num_t>(RED_LED_PIN), 1);
  vTaskDelay(pdMS_TO_TICKS(RED_LED_FLASH_MS));
  gpio_set_level(static_cast<gpio_num_t>(RED_LED_PIN), 0);
  const adc_channel_t sensor_channels[kSensorCount] = {
      ADC_CHANNEL_6,
      ADC_CHANNEL_7,
  };
  const gpio_num_t sensor_power_pins[kSensorCount] = {
      static_cast<gpio_num_t>(SENSOR_POWER_PIN_1),
      static_cast<gpio_num_t>(SENSOR_POWER_PIN_2),
  };
  int sensor_values[kSensorCount];
  for (size_t i = 0; i < kSensorCount; ++i) {
    int64_t sample_time_us = static_cast<int64_t>(esp_rtc_get_time_us());
    sensor_values[i] = read_sensor_with_power(
      state,
      sensor_power_pins[i],
      sensor_channels[i]
    );
    push_sensor_measurement(sensor_values[i], sample_time_us);
  }

  ESP_LOGI(TAG, "Sensor value 1: %d", sensor_values[0]);
  ESP_LOGI(TAG, "Sensor value 2: %d", sensor_values[1]);
}

// Init NVS, which is used to store persistent data across reboots, such as BLE bonding info.
static void init_nvs_storage() {
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ESP_ERROR_CHECK(nvs_flash_init());
  }
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
  
  init_nvs_storage();
  init_power_pins();
  take_sensor_readings(state);

  ++g_sleep_cycle_count;
  if (g_sleep_cycle_count < N_SLEEPS_PER_TRANSMISSION) {
    ESP_LOGI(TAG, "Skipping BLE (%u/%u)",
             g_sleep_cycle_count,
             N_SLEEPS_PER_TRANSMISSION);
    sleep_now();
    return;
  }

  if (!init_ble_stack(state)) {
    return;
  }

  // Start a new FreeRTOS task that runs in parallel with app_main.
  // We need this because the BLE stack requires its own event loop to function properly.
  // Going into deep sleep immediately after starting the BLE stack would prevent it from operating.
  xTaskCreate(sleep_task, "sleep_task", 2048, &state, 5, nullptr);
}

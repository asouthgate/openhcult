/* Standard integer types used across ESP-IDF and NimBLE APIs. */
#include <stdint.h>

/*
 * FreeRTOS is the real-time OS that ESP-IDF runs on; it provides tasks,
 * scheduling, and timing primitives used throughout this file.
 */
// Q: so this is the main OS library?
// A: This is the core FreeRTOS header; it gives you the kernel types and APIs.
// A: ESP-IDF uses FreeRTOS as its runtime, so most system services depend on it.
#include "freertos/FreeRTOS.h"
/* Task creation and scheduling APIs built on top of FreeRTOS core types. */
// Q: tasks? so multithreading?
// A: FreeRTOS tasks are lightweight threads scheduled by the RTOS.
// A: On ESP32 they can run across the two cores, but they are still RTOS tasks.
#include "freertos/task.h"

/* GPIO driver for pin direction and level control. */
// Q: Why do we need a GPIO driver? 
// A: GPIO pins are hardware peripherals; this driver configures pin modes and
// A: safely reads/writes levels so the chip behaves as intended.
#include "driver/gpio.h"
/* ESP-IDF logging macros (ESP_LOGI/W/E). */
#include "esp_log.h"
/* Deep sleep APIs for low-power operation. */
// Q: is this proper deep sleep?
// A: Yes. ESP-IDF's deep sleep powers down most of the chip; only RTC keeps time
// A: and wake sources. A wake resets the CPU and restarts app_main.
#include "esp_sleep.h"
/* High-resolution timer for microsecond timestamps. */
#include "esp_timer.h"
/*
 * NVS (Non-Volatile Storage) is a small key-value store in flash.
 * NimBLE uses it for BLE state (e.g., bonding keys), so we initialize it.
 */
// Q: is this in-memory?
// A: No. NVS is stored in flash (non-volatile). It may cache in RAM while running,
// A: but the authoritative data is persisted across reboots.
#include "nvs_flash.h"
/* ADC oneshot driver for single-sample reads (ESP-IDF v6). */
// Q: one does oneshot mean? 
// A: "One-shot" ADC means you request a conversion each time you need a sample,
// A: instead of running the ADC continuously in the background.
#include "esp_adc/adc_oneshot.h"

/*
 * NimBLE is the BLE host stack (GAP/GATT, security, etc.).
 * It talks to the ESP32 BLE controller via the HCI layer.
 */
// Q: What is GAP/GATT?
// A: GAP manages advertising, connections, and device roles.
// A: GATT defines services/characteristics and how clients read/write them.
// Q: What is HCI?
// A: HCI is the command/event interface between the BLE host stack and controller.
#include "esp_nimble_hci.h"
/* NimBLE host initialization and core runtime. */
#include "nimble/nimble_port.h"
/* FreeRTOS integration for running the NimBLE host. */
// Q: what is FreeRTOS?
// A: It's the real-time OS used by ESP-IDF; it provides tasks, queues, timers,
// A: and synchronization primitives used throughout this code.
#include "nimble/nimble_port_freertos.h"
/* Core BLE host definitions (GATT, GAP types). */
#include "host/ble_hs.h"
/* GAP service helper (device name, appearance). */
#include "services/gap/ble_svc_gap.h"
/* GATT service helper (standard GATT service). */
#include "services/gatt/ble_svc_gatt.h"
/* Simple NVS-backed storage helpers for NimBLE. */
// Q: NVS?
// A: Non-Volatile Storage in flash; NimBLE uses it to persist BLE state.
#include "host/ble_store.h"

#include "ble.h"
#include "globals.h"
#include "pins.h"
#include "sensor.h"
#include "sleep.h"

extern "C" void app_main(void) {
  // Capture a fixed reference time so the device sleeps after a consistent
  // window even if advertising restarts or a client disconnects.
  boot_time_us = esp_timer_get_time();
  // Q: What do we mean by flash here?
  // A: NVS lives in on-chip flash memory (persistent storage), not RAM.
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ESP_ERROR_CHECK(nvs_flash_init());
  }

  gpio_config_t io_conf = {};
  // Q: what is this mode?
  // A: GPIO_MODE_OUTPUT sets these pins as digital outputs.
  io_conf.intr_type = GPIO_INTR_DISABLE;
  io_conf.mode = GPIO_MODE_OUTPUT;
  // Q: What is this? Does io_config apply to all these pins?
  // A: Yes. pin_bit_mask specifies which pins this configuration struct applies to.
  io_conf.pin_bit_mask = (1ULL << LED_PIN) | (1ULL << SENSOR_POWER_PIN_1) |
                         (1ULL << SENSOR_POWER_PIN_2);
  io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
  io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
  ESP_ERROR_CHECK(gpio_config(&io_conf));

  // Power the sensors just long enough to take a single batch of readings.
  // Q: what's the static_cast for again?
  // A: It converts an integer macro to the gpio_num_t enum expected by the API.
  gpio_set_level(static_cast<gpio_num_t>(LED_PIN), 1);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_1), 1);
  gpio_set_level(static_cast<gpio_num_t>(SENSOR_POWER_PIN_2), 1);
  vTaskDelay(pdMS_TO_TICKS(200));

  // Q: what is this?
  // A: It configures the ADC unit (ADC1) for oneshot sampling.
  adc_oneshot_unit_init_cfg_t unit_cfg = {};
  unit_cfg.unit_id = ADC_UNIT_1;
  ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_cfg, &adc_handle));

  // Q: what is this?
  // A: It configures per-channel settings like attenuation and bit width.
  adc_oneshot_chan_cfg_t chan_cfg = {};
  chan_cfg.atten = ADC_ATTEN_DB_12;
  chan_cfg.bitwidth = ADC_BITWIDTH_12;
  ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, ADC_CHANNEL_6, &chan_cfg));
  ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, ADC_CHANNEL_7, &chan_cfg));

  // Read sensors once per boot to keep runtime and power usage predictable.
  // Q: does this actually keep anything predictable?
  // A: It keeps runtime and power usage predictable (single read per boot).
  // A: It does not make the sensor values themselves predictable.
  g_sensor_value_1 = read_sensor(ADC_CHANNEL_6);
  g_sensor_value_2 = read_sensor(ADC_CHANNEL_7);

  ESP_LOGI(TAG, "Sensor value 1: %d", g_sensor_value_1);
  ESP_LOGI(TAG, "Sensor value 2: %d", g_sensor_value_2);

  load_ble_uuids();
  if (!g_ble_uuid_ok) {
    ESP_LOGE(TAG, "BLE UUIDs not configured; aborting");
    return;
  }

  nimble_port_init();

  ble_svc_gap_init();
  ble_svc_gatt_init();

  // Q: what is gatts_count?
  // A: It counts how many GATT attributes are needed so NimBLE can allocate them
  // A: before we register the services.
  int rc = ble_gatts_count_cfg(gatt_svcs);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gatts_count_cfg failed: %d", rc);
    return;
  }
  rc = ble_gatts_add_svcs(gatt_svcs);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gatts_add_svcs failed: %d", rc);
    return;
  }

  ble_hs_cfg.sync_cb = ble_on_sync;
  ble_hs_cfg.reset_cb = ble_on_reset;
  ble_hs_cfg.store_status_cb = ble_store_util_status_rr;

  nimble_port_freertos_init(ble_host_task);

  // Q: what is this function call, how does it work relative to control flow? Sleep_task puts into deep_sleep, but how do we end up back at the start of this function?
  // A: xTaskCreate starts a new FreeRTOS task that runs in parallel with app_main.
  // A: Deep sleep resets the CPU, so on wake the firmware starts at app_main again.
  xTaskCreate(sleep_task, "sleep_task", 2048, nullptr, 5, nullptr);
}

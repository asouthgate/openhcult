/* Standard integer types used across ESP-IDF and NimBLE APIs. */
#include <stdint.h>
/* C string utilities for zeroing structs and basic buffer helpers. */
#include <string.h>

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
/* Address inference helpers for advertising. */
#include "host/ble_hs_id.h"
/* GAP service helper (device name, appearance). */
#include "services/gap/ble_svc_gap.h"
/* GATT service helper (standard GATT service). */
#include "services/gatt/ble_svc_gatt.h"
/* UUID parsing helpers for loading config at runtime. */
#include "host/ble_uuid.h"
/* Simple NVS-backed storage helpers for NimBLE. */
// Q: NVS?
// A: Non-Volatile Storage in flash; NimBLE uses it to persist BLE state.
#include "host/ble_store.h"

/* Build-time generated config with BLE UUID strings. */
#include "ble_config.h"

// Q: should these be #defines?
// A: They can be macros or constexprs. In C++ I'd prefer constexpr, but ESP-IDF
// A: examples often use macros for pin/constant configuration.
#define LED_PIN 4
#define SENSOR_POWER_PIN_1 5
#define SENSOR_PIN_1 GPIO_NUM_34

#define SENSOR_POWER_PIN_2 18
#define SENSOR_PIN_2 GPIO_NUM_35

static const char *TAG = "hcultfw";


// Q: What are these?
// A: These are the 128-bit BLE UUIDs loaded from the repo config at boot.
static ble_uuid128_t g_service_uuid;
static ble_uuid128_t g_characteristic_uuid;
static bool g_ble_uuid_ok;

// Q: what are each of these?
// A: gatt_chr_handle: runtime handle to the characteristic value.
// A: g_sensor_value_1/2: cached ADC readings sent over BLE.
// A: boot_time_us: timestamp used to enforce the 45s window.
// A: adc_handle: ADC driver instance for oneshot reads.
// A: g_ble_addr_type: public vs random address used for advertising.
static uint16_t gatt_chr_handle;
static int g_sensor_value_1;
static int g_sensor_value_2;
static int64_t boot_time_us;
static adc_oneshot_unit_handle_t adc_handle;
static uint8_t g_ble_addr_type;
// Q: what is volatile?
// A: It tells the compiler the variable can change outside the current context.
// A: It prevents some optimizations, but it is not a full thread-safety mechanism.
static volatile bool g_request_sleep;
static volatile bool g_sent_payload;

/* Forward declaration: used before definition by the advertising function. */
// Q: what is a gap event?
// A: A GAP event is a BLE connection/advertising state change (connect, disconnect,
// A: advertising complete, etc.) delivered to the host callback.
static int gap_event_cb(struct ble_gap_event *event, void *arg);

/*
 * GATT access callback for our characteristic.
 * Read: returns the two sensor values as a 2x uint16 payload.
 * Write: treats any write as a "send now" request, notifies the same payload,
 * then signals the sleep task to power down early.
 */
static int gatt_svr_chr_access(uint16_t conn_handle, uint16_t attr_handle,
                               struct ble_gatt_access_ctxt *ctxt, void *arg) {
  uint16_t payload[2];
  payload[0] = static_cast<uint16_t>(g_sensor_value_1);
  payload[1] = static_cast<uint16_t>(g_sensor_value_2);

  // Q: what is this condition for?
  // A: It checks whether the client is doing a GATT read on the characteristic.
  if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
    return os_mbuf_append(ctxt->om, payload, sizeof(payload));
  }

  // Q: what is this condition for?
  // A: It checks whether the client wrote to the characteristic, which we treat
  // A: as a request to send the data back via a notification.
  if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
    // Q: what is the os_mbuf struct?
    // A: It's NimBLE's packet buffer type (a chainable buffer used for BLE data).
    // Q: why do we define a struct in this way? we define the os_mbfu struct? What about fields?
    // A: We don't define the struct here; we just allocate one using a helper.
    // A: The helper fills internal fields that NimBLE uses to manage the buffer.
    struct os_mbuf *om = ble_hs_mbuf_from_flat(payload, sizeof(payload));
    if (om == nullptr) {
      // Q: what does this mean?
      // A: It returns a GATT error code telling the client we ran out of memory.
      return BLE_ATT_ERR_INSUFFICIENT_RES;
    }
    // Q: what is this?
    // A: It sends a GATT notification to the client with our payload.
    int rc = ble_gatts_notify_custom(conn_handle, gatt_chr_handle, om);
    if (rc != 0) {
      ESP_LOGW(TAG, "Notify failed: %d", rc);
    } else {
      ESP_LOGI(TAG, "Client wrote request; notified payload");
      g_sent_payload = true;
      int dc = ble_gap_terminate(conn_handle, BLE_ERR_REM_USER_CONN_TERM);
      if (dc != 0) {
        ESP_LOGW(TAG, "Disconnect request failed: %d", dc);
      }
    }
    return 0;
  }

  return BLE_ATT_ERR_UNLIKELY;
}


// Q: what is this struct? Something like characteristic definition? What's that?
// A: Yes. It declares the characteristic (UUID, access callback, flags) so the
// A: GATT server knows what attributes to expose.
static struct ble_gatt_chr_def gatt_chr_defs[] = {
    {
        &g_characteristic_uuid.u,
        gatt_svr_chr_access,
        nullptr,
        nullptr,
        static_cast<ble_gatt_chr_flags>(BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_WRITE |
                                        BLE_GATT_CHR_F_WRITE_NO_RSP |
                                        BLE_GATT_CHR_F_NOTIFY),
        0,
        &gatt_chr_handle,
        nullptr,
    },
    {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        nullptr,
        nullptr,
    },
};


// Q: what is this struct? Something like svc definition? What's that?
// A: It declares the service that groups characteristics under one UUID.
static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        BLE_GATT_SVC_TYPE_PRIMARY,
        &g_service_uuid.u,
        nullptr,
        gatt_chr_defs,
    },
    {
        0,
        nullptr,
        nullptr,
        nullptr,
    },
};

/*
 * Starts BLE advertising with our custom service UUID.
 * This keeps the device discoverable for a short window so a client can
 * connect and request the sensor readings.
 */
static void start_advertising(void) {
  // Q: what are each of these structs for?
  // A: ble_hs_adv_fields describes the advertising payload (what we broadcast).
  // A: ble_gap_adv_params controls advertising behavior (connectable, discoverable).
  struct ble_gap_adv_params adv_params;
  struct ble_hs_adv_fields fields;

  // Q: why we set the fields to 0?
  // A: Zero-initialization avoids uninitialized garbage and clearly marks fields
  // A: we are not using.
  memset(&fields, 0, sizeof(fields));
  // Advertise the custom service UUID so clients can discover it quickly.
  fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
  fields.tx_pwr_lvl_is_present = 1;
  fields.tx_pwr_lvl = BLE_HS_ADV_TX_PWR_LVL_AUTO;
  fields.uuids128 = &g_service_uuid;
  fields.num_uuids128 = 1;
  fields.uuids128_is_complete = 1;

  // Q: where is ble_gap_adv_set_fields from?
  // A: It's part of NimBLE's GAP API (ble_gap.h); it builds the advertising data.
  int rc = ble_gap_adv_set_fields(&fields);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gap_adv_set_fields failed: %d", rc);
    return;
  }

  memset(&adv_params, 0, sizeof(adv_params));
  adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
  adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;

  rc = ble_gap_adv_start(g_ble_addr_type, nullptr, BLE_HS_FOREVER,
                         &adv_params, gap_event_cb, nullptr);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gap_adv_start failed: %d", rc);
    return;
  }

  ESP_LOGI(TAG, "BLE advertising started");
}

/*
 * GAP event handler invoked by NimBLE for connection state changes.
 * We restart advertising on failures or disconnects to remain discoverable.
 */
// Q: what is GAP?
// A: GAP is the BLE layer that manages advertising, discovery, and connections.
static int gap_event_cb(struct ble_gap_event *event, void *arg) {
  switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
      if (event->connect.status == 0) {
        ESP_LOGI(TAG, "Client connected");
      } else {
        // Keep advertising when a connection attempt fails.
        ESP_LOGW(TAG, "Connection failed; restarting advertising");
        start_advertising();
      }
      return 0;
    case BLE_GAP_EVENT_DISCONNECT:
      ESP_LOGI(TAG, "Client disconnected");
      if (g_sent_payload) {
        ESP_LOGI(TAG, "Payload delivered; sleeping");
        g_request_sleep = true;
      } else {
        // Restart advertising so the next client can fetch the one-shot data.
        start_advertising();
      }
      return 0;
    default:
      return 0;
  }
}

/*
 * NimBLE reset callback.
 * This is mostly informational; it helps diagnose unexpected controller resets.
 */
static void ble_on_reset(int reason) {
  ESP_LOGE(TAG, "Resetting NimBLE; reason=%d", reason);
}

/*
 * NimBLE sync callback.
 * Runs after the host stack is ready, so it's the right time to set the
 * device name and start advertising.
 */
// Q: what is a NimBLE sync?
// A: It's the point where the host stack has synchronized with the controller and
// A: is ready to start using GAP/GATT (e.g., start advertising).
static void ble_on_sync(void) {
  // Use the controller-provided address type (public or random).
  int rc = ble_hs_id_infer_auto(0, &g_ble_addr_type);
  if (rc != 0) {
    ESP_LOGE(TAG, "Address ensure failed: %d", rc);
    return;
  }

  ble_svc_gap_device_name_set("ESP32_Sensor");
  start_advertising();
}

/*
 * Loads BLE UUIDs from build-time config and validates their format.
 * This keeps firmware and monitor UUIDs in sync via the shared repo config.
 */
static void load_ble_uuids(void) {
  ble_uuid_any_t uuid_any;
  int rc = ble_uuid_from_str(&uuid_any, BLE_SERVICE_UUID_STR);
  if (rc != 0 || uuid_any.u.type != BLE_UUID_TYPE_128) {
    ESP_LOGE(TAG, "Invalid BLE service UUID: %s", BLE_SERVICE_UUID_STR);
    return;
  }
  g_service_uuid = *BLE_UUID128(&uuid_any.u);

  rc = ble_uuid_from_str(&uuid_any, BLE_CHARACTERISTIC_UUID_STR);
  if (rc != 0 || uuid_any.u.type != BLE_UUID_TYPE_128) {
    ESP_LOGE(TAG, "Invalid BLE characteristic UUID: %s", BLE_CHARACTERISTIC_UUID_STR);
    return;
  }
  g_characteristic_uuid = *BLE_UUID128(&uuid_any.u);
  g_ble_uuid_ok = true;
}

/*
 * FreeRTOS task entry point for the NimBLE host.
 * NimBLE runs an internal event loop here for the lifetime of BLE activity.
 */
// Q: How does this relate to the control flow from our main function? This event loop is the main loop or what?
// A: It's the BLE stack's own event loop running in a separate FreeRTOS task.
// A: app_main continues after starting the task; this does not replace app_main.
static void ble_host_task(void *param) {
  // NimBLE runs an internal event loop; this task blocks until it exits.
  nimble_port_run();
  nimble_port_freertos_deinit();
}

/*
 * Reads a single ADC channel using multiple samples.
 * This favors stable readings over speed because we only sample once per boot.
 */
static int read_sensor(adc_channel_t channel) {
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

/*
 * Background task that decides when to enter deep sleep.
 * It sleeps either when the fixed advertising window expires or once data has
 * been successfully notified to a client.
 */
static void sleep_task(void *param) {
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

/*
 * ESP-IDF entry point.
 * Initializes power, reads sensors once, brings up NimBLE, and then waits for
 * either a client request or the 45s timeout before sleeping.
 */
// Why do we have to extern "C" in this context, isn't it compatible with C++?
//
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

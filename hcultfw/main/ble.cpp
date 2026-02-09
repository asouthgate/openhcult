// This file is basically a bunch of callbacks and boilerplate. The callbacks are important.
#include <string.h>
#include <stdio.h>

#include "ble.h"
#include "state.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "host/ble_hs.h"
#include "host/ble_hs_id.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "esp_bt.h"
#include "services/gap/ble_svc_gap.h"

static const char *TAG = "hcultfw";
// TODO: this global shouldn't be a global and if it has to be shouldn't be defined here.
//   or the BLE globals should be split from the others
static FirmwareState *s_state;
static char s_device_name[32];

static int gap_event_cb(struct ble_gap_event *event, void *arg);
static void start_advertising_beacon(void);

namespace {
constexpr uint16_t kAdvCompanyId = 0xFFFF;
constexpr uint8_t kAdvPayloadVersion = 1;
// Future: rotate sensor types (e.g., moisture/temp/light) across 5s windows and
// encode the type in a payload flag to keep the advertisement compact.
constexpr uint8_t kAdvPayloadMagic0 = 'H';
constexpr uint8_t kAdvPayloadMagic1 = 'C';
constexpr size_t kAdvPayloadSize = 12; // Magic(2) + version + count + values(4) + timestamp(4).
constexpr size_t kAdvMfgDataSize = 2 + kAdvPayloadSize; // Company ID + payload.
} // namespace

static bool build_adv_mfg_data(
  uint8_t *out,
  size_t out_capacity,
  size_t *out_len
) {
  if (out_capacity < kAdvMfgDataSize) {
    return false;
  }
  float values[kSensorCount] = {};
  for (size_t i = 0; i < kSensorCount; ++i) {
    values[i] = s_state->last_sensor_values[i];
  }

  out[0] = static_cast<uint8_t>(kAdvCompanyId & 0xFF);
  out[1] = static_cast<uint8_t>((kAdvCompanyId >> 8) & 0xFF);
  out[2] = kAdvPayloadMagic0;
  out[3] = kAdvPayloadMagic1;
  out[4] = kAdvPayloadVersion;
  out[5] = static_cast<uint8_t>(kSensorCount & 0xFF);

  uint16_t sensor0 = static_cast<uint16_t>(values[0]);
  uint16_t sensor1 = static_cast<uint16_t>(values[1]);
  out[6] = static_cast<uint8_t>(sensor0 & 0xFF);
  out[7] = static_cast<uint8_t>((sensor0 >> 8) & 0xFF);
  out[8] = static_cast<uint8_t>(sensor1 & 0xFF);
  out[9] = static_cast<uint8_t>((sensor1 >> 8) & 0xFF);

  uint32_t timestamp_s = s_state->last_timestamp_s;
  out[10] = static_cast<uint8_t>(timestamp_s & 0xFF);
  out[11] = static_cast<uint8_t>((timestamp_s >> 8) & 0xFF);
  out[12] = static_cast<uint8_t>((timestamp_s >> 16) & 0xFF);
  out[13] = static_cast<uint8_t>((timestamp_s >> 24) & 0xFF);

  *out_len = kAdvMfgDataSize;
  return true;
}

void ble_init(FirmwareState &state) {
  s_state = &state;
}

bool init_ble_stack(FirmwareState &state) {
  nimble_port_init();

  ble_svc_gap_init();
  ble_init(state);

  ble_hs_cfg.sync_cb = ble_on_sync;
  ble_hs_cfg.reset_cb = ble_on_reset;
  nimble_port_freertos_init(ble_host_task);
  return true;
}

static void start_advertising_beacon(void) {
  struct ble_gap_adv_params adv_params;
  struct ble_hs_adv_fields fields;
  uint8_t mfg_data[kAdvMfgDataSize] = {};
  size_t mfg_len = 0;

  if (!build_adv_mfg_data(mfg_data, sizeof(mfg_data), &mfg_len)) {
    ESP_LOGW(TAG, "No sensor data available for beacon");
    return;
  }

  memset(&fields, 0, sizeof(fields));
  fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
  fields.tx_pwr_lvl_is_present = 1;
  fields.tx_pwr_lvl = BLE_HS_ADV_TX_PWR_LVL_AUTO;
  fields.mfg_data = mfg_data;
  fields.mfg_data_len = mfg_len;

  int rc = ble_gap_adv_set_fields(&fields);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gap_adv_set_fields failed: %d", rc);
    return;
  }

  memset(&adv_params, 0, sizeof(adv_params));
  adv_params.conn_mode = BLE_GAP_CONN_MODE_NON;
  adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
  adv_params.itvl_min = 0x0320; // 500 ms in 0.625 ms units.
  adv_params.itvl_max = 0x0320; // 500 ms in 0.625 ms units.
  // Reduce peak draw by limiting to two advertising channels (lower discovery reliability).
  adv_params.channel_map = 0x03; // Channels 37 and 38.

  rc = ble_gap_adv_start(s_state->ble_addr_type, nullptr, BLE_HS_FOREVER,
                         &adv_params, gap_event_cb, nullptr);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gap_adv_start failed: %d", rc);
    return;
  }

  ESP_LOGI(TAG, "BLE beacon advertising started");
}

// GAP event handler invoked by NimBLE for connection state changes.
static int gap_event_cb(struct ble_gap_event *event, void *arg) {
  switch (event->type) {
    case BLE_GAP_EVENT_ADV_COMPLETE:
      ESP_LOGW(TAG, "Advertising stopped; restarting beacon");
      start_advertising_beacon();
      return 0;
    default:
      return 0;
  }
}

// NimBLE reset callback.
void ble_on_reset(int reason) {
  ESP_LOGE(TAG, "Resetting NimBLE; reason=%d", reason);
}

// NimBLE sync callback, runs after the host stack is ready
void ble_on_sync(void) {
  // Use the controller-provided address type (public or random).
  int rc = ble_hs_id_infer_auto(0, &s_state->ble_addr_type);
  if (rc != 0) {
    ESP_LOGE(TAG, "Address ensure failed: %d", rc);
    return;
  }

  uint8_t mac[6] = {};
  esp_read_mac(mac, ESP_MAC_BT);
  snprintf(
    s_device_name,
    sizeof(s_device_name),
    "ESP32_Sensor_%02X%02X%02X",
    mac[3],
    mac[4],
    mac[5]
  );
  rc = ble_svc_gap_device_name_set(s_device_name);
  if (rc != 0) {
    ESP_LOGE(TAG, "Device name set failed: %d", rc);
    return;
  }
  ESP_LOGI(TAG, "BLE device name: %s", s_device_name);
  esp_err_t err = esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, ESP_PWR_LVL_N12);
  if (err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to set BLE TX power: %d", err);
  }
  start_advertising_beacon();
}

// Entrypoint for the NimBLE host task.
void ble_host_task(void *param) {
  // NimBLE runs an internal event loop; this task blocks until it exits.
  nimble_port_run();
  nimble_port_freertos_deinit();
}

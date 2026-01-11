// This file is basically a bunch of callbacks and boilerplate. The callbacks are important.
#include <string.h>
#include <stdio.h>

#include "ble.h"
#include "state.h"
#include "ble_config.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "host/ble_hs.h"
#include "host/ble_store.h"
#include "host/ble_hs_id.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "payload.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

static const char *TAG = "hcultfw";
// TODO: this global shouldn't be a global and if it has to be shouldn't be defined here.
//   or the BLE globals should be split from the others
static FirmwareState *s_state;
static char s_device_name[32];

static int gap_event_cb(struct ble_gap_event *event, void *arg);

// This is the GATT characteristic access callback, in which we handle response to read.
static int gatt_svr_chr_access(
  uint16_t conn_handle,
  uint16_t attr_handle,
  struct ble_gatt_access_ctxt *ctxt,
  void *arg
) {
  // Main logic for read operation
  if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
    uint8_t payload[kSensorCount * kSensorPayloadStride];
    size_t payload_count = g_sensor_buffer_count / kSensorCount;
    uint8_t header[kPayloadHeaderSize];
    size_t header_size = build_payload_header(
      static_cast<uint16_t>(payload_count),
      header,
      sizeof(header)
    );
    if (header_size == 0) {
      return BLE_ATT_ERR_UNLIKELY;
    }
    int rc = os_mbuf_append(ctxt->om, header, header_size);
    if (rc != 0) {
      return rc;
    }
    for (size_t i = 0; i < payload_count; ++i) {
      size_t payload_size = get_payload_i(i, payload, sizeof(payload));
      if (payload_size == 0) {
        break;
      }
      rc = os_mbuf_append(ctxt->om, payload, payload_size);
      if (rc != 0) {
        return rc;
      }
    }

    s_state->sent_payload = true;
    return 0;
  }
  return BLE_ATT_ERR_UNLIKELY;
}

// Definition for the characteristics in our GATT service.
// See where we specify gatt_svr_chr_access, read/write handler.
// NimBLE expects the array to be terminated by an entry with all zeroes.
static struct ble_gatt_chr_def gatt_chr_defs[] = {
    {
        nullptr,
        gatt_svr_chr_access,
        nullptr,
        nullptr,
        static_cast<ble_gatt_chr_flags>(BLE_GATT_CHR_F_READ),
        0,
        nullptr,
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

// Service is basically just a container for characteristics.
struct ble_gatt_svc_def gatt_svcs[] = {
    {
        BLE_GATT_SVC_TYPE_PRIMARY,
        nullptr,
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

void ble_init(FirmwareState &state) {
  s_state = &state;
  gatt_chr_defs[0].uuid = &state.characteristic_uuid.u;
  gatt_chr_defs[0].val_handle = &state.gatt_chr_handle;
  gatt_svcs[0].uuid = &state.service_uuid.u;
}

bool init_ble_stack(FirmwareState &state) {
  load_ble_uuids(state);
  if (!state.ble_uuid_ok) {
    ESP_LOGE(TAG, "BLE UUIDs not configured; aborting");
    return false;
  }

  nimble_port_init();

  ble_svc_gap_init();
  ble_svc_gatt_init();
  ble_init(state);

  int rc = ble_gatts_count_cfg(gatt_svcs);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gatts_count_cfg failed: %d", rc);
    return false;
  }
  rc = ble_gatts_add_svcs(gatt_svcs);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gatts_add_svcs failed: %d", rc);
    return false;
  }

  ble_hs_cfg.sync_cb = ble_on_sync;
  ble_hs_cfg.reset_cb = ble_on_reset;
  ble_hs_cfg.store_status_cb = ble_store_util_status_rr;

  nimble_port_freertos_init(ble_host_task);
  return true;
}

static void start_advertising(void) {
  struct ble_gap_adv_params adv_params;
  struct ble_hs_adv_fields fields;
  memset(&fields, 0, sizeof(fields));
  fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
  fields.tx_pwr_lvl_is_present = 1;
  fields.tx_pwr_lvl = BLE_HS_ADV_TX_PWR_LVL_AUTO;
  fields.uuids128 = &s_state->service_uuid;
  fields.num_uuids128 = 1;
  fields.uuids128_is_complete = 1;

  int rc = ble_gap_adv_set_fields(&fields);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gap_adv_set_fields failed: %d", rc);
    return;
  }

  memset(&fields, 0, sizeof(fields));
  const char *adv_name = ble_svc_gap_device_name();
  if (adv_name) {
    fields.name = reinterpret_cast<const uint8_t *>(adv_name);
    fields.name_len = strlen(adv_name);
    fields.name_is_complete = 1;
    rc = ble_gap_adv_rsp_set_fields(&fields);
    if (rc != 0) {
      ESP_LOGE(TAG, "ble_gap_adv_rsp_set_fields failed: %d", rc);
      return;
    }
  }

  memset(&adv_params, 0, sizeof(adv_params));
  adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
  adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;

  rc = ble_gap_adv_start(s_state->ble_addr_type, nullptr, BLE_HS_FOREVER,
                         &adv_params, gap_event_cb, nullptr);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gap_adv_start failed: %d", rc);
    return;
  }

  ESP_LOGI(TAG, "BLE advertising started");
}

// GAP event handler invoked by NimBLE for connection state changes.
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
      if (s_state->sent_payload) {
        ESP_LOGI(TAG, "Payload delivered; sleeping");
        clear_sensor_buffer();
        g_sleep_cycle_count = 0;
        s_state->request_sleep = true;
      } else {
        // Restart advertising so the next client can fetch the one-shot data.
        start_advertising();
      }
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
  start_advertising();
}

void load_ble_uuids(FirmwareState &state) {
  ble_uuid_any_t uuid_any;
  int rc = ble_uuid_from_str(&uuid_any, BLE_SERVICE_UUID_STR);
  if (rc != 0 || uuid_any.u.type != BLE_UUID_TYPE_128) {
    ESP_LOGE(TAG, "Invalid BLE service UUID: %s", BLE_SERVICE_UUID_STR);
    return;
  }
  state.service_uuid = *BLE_UUID128(&uuid_any.u);

  rc = ble_uuid_from_str(&uuid_any, BLE_CHARACTERISTIC_UUID_STR);
  if (rc != 0 || uuid_any.u.type != BLE_UUID_TYPE_128) {
    ESP_LOGE(TAG, "Invalid BLE characteristic UUID: %s", BLE_CHARACTERISTIC_UUID_STR);
    return;
  }
  state.characteristic_uuid = *BLE_UUID128(&uuid_any.u);
  state.ble_uuid_ok = true;
}

// Entrypoint for the NimBLE host task.
void ble_host_task(void *param) {
  // NimBLE runs an internal event loop; this task blocks until it exits.
  nimble_port_run();
  nimble_port_freertos_deinit();
}

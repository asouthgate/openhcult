/* C string utilities for zeroing structs and basic buffer helpers. */
#include <string.h>

#include "ble.h"
#include "state.h"
#include "ble_config.h"
#include "esp_log.h"
#include "host/ble_hs.h"
#include "host/ble_store.h"
#include "host/ble_hs_id.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "payload.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

static const char *TAG = "hcultfw";
static FirmwareState *s_state;

/* Forward declaration: used before definition by the advertising function. */
// Q: what is a gap event?
// A: A GAP event is a BLE connection/advertising state change (connect, disconnect,
// A: advertising complete, etc.) delivered to the host callback.
static int gap_event_cb(struct ble_gap_event *event, void *arg);

/*
 * GATT access callback for our characteristic.
 * Read: returns the latest buffered values with timestamps.
 */
static int gatt_svr_chr_access(uint16_t conn_handle, uint16_t attr_handle,
                               struct ble_gatt_access_ctxt *ctxt, void *arg) {
  if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
    uint8_t payload[kSensorCount * kSensorPayloadStride];
    size_t payload_count = g_sensor_buffer_count / kSensorCount;
    uint8_t header[kPayloadHeaderSize];
    size_t header_size =
        build_payload_header(static_cast<uint16_t>(payload_count),
                             header, sizeof(header));
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


// Q: what is this struct? Something like characteristic definition? What's that?
// A: Yes. It declares the characteristic (UUID, access callback, flags) so the
// A: GATT server knows what attributes to expose.
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


// Q: what is this struct? Something like svc definition? What's that?
// A: It declares the service that groups characteristics under one UUID.
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

  // Q: what is gatts_count?
  // A: It counts how many GATT attributes are needed so NimBLE can allocate them
  // A: before we register the services.
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
  fields.uuids128 = &s_state->service_uuid;
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

  rc = ble_gap_adv_start(s_state->ble_addr_type, nullptr, BLE_HS_FOREVER,
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
      if (s_state->sent_payload) {
        ESP_LOGI(TAG, "Payload delivered; sleeping");
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

/*
 * NimBLE reset callback.
 * This is mostly informational; it helps diagnose unexpected controller resets.
 */
void ble_on_reset(int reason) {
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
void ble_on_sync(void) {
  // Use the controller-provided address type (public or random).
  int rc = ble_hs_id_infer_auto(0, &s_state->ble_addr_type);
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

/*
 * FreeRTOS task entry point for the NimBLE host.
 * NimBLE runs an internal event loop here for the lifetime of BLE activity.
 */
// Q: How does this relate to the control flow from our main function? This event loop is the main loop or what?
// A: It's the BLE stack's own event loop running in a separate FreeRTOS task.
// A: app_main continues after starting the task; this does not replace app_main.
void ble_host_task(void *param) {
  // NimBLE runs an internal event loop; this task blocks until it exits.
  nimble_port_run();
  nimble_port_freertos_deinit();
}

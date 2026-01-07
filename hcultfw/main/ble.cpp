/* C string utilities for zeroing structs and basic buffer helpers. */
#include <string.h>

#include "ble.h"
#include "state.h"
#include "ble_config.h"
#include "esp_log.h"
#include "host/ble_hs.h"
#include "host/ble_hs_id.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "payload.h"
#include "services/gap/ble_svc_gap.h"

static const char *TAG = "hcultfw";
static FirmwareState *s_state;

/* Forward declaration: used before definition by the advertising function. */
// Q: what is a gap event?
// A: A GAP event is a BLE connection/advertising state change (connect, disconnect,
// A: advertising complete, etc.) delivered to the host callback.
static int gap_event_cb(struct ble_gap_event *event, void *arg);

/*
 * GATT access callback for our characteristic.
 * Read: returns the latest buffered values as a 2x uint16 payload.
 * Write: treats any write as a "send now" request, notifies the same payload,
 * then signals the sleep task to power down early.
 */
static int gatt_svr_chr_access(uint16_t conn_handle, uint16_t attr_handle,
                               struct ble_gatt_access_ctxt *ctxt, void *arg) {
  int latest_values[kSensorCount];
  size_t latest_count = copy_latest_measurements(
      latest_values, sizeof(latest_values) / sizeof(latest_values[0]));
  uint16_t payload[kSensorCount];
  size_t payload_count = build_sensor_payload(
      latest_values, latest_count,
      payload, sizeof(payload) / sizeof(payload[0]));

  // Q: what is this condition for?
  // A: It checks whether the client is doing a GATT read on the characteristic.
  if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
    return os_mbuf_append(ctxt->om, payload,
                          payload_count * sizeof(payload[0]));
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
    struct os_mbuf *om =
        ble_hs_mbuf_from_flat(payload, payload_count * sizeof(payload[0]));
    if (om == nullptr) {
      // Q: what does this mean?
      // A: It returns a GATT error code telling the client we ran out of memory.
      return BLE_ATT_ERR_INSUFFICIENT_RES;
    }
    // Q: what is this?
    // A: It sends a GATT notification to the client with our payload.
    int rc = ble_gatts_notify_custom(conn_handle, s_state->gatt_chr_handle, om);
    if (rc != 0) {
      ESP_LOGW(TAG, "Notify failed: %d", rc);
    } else {
      ESP_LOGI(TAG, "Client wrote request; notified payload");
      s_state->sent_payload = true;
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
        nullptr,
        gatt_svr_chr_access,
        nullptr,
        nullptr,
        static_cast<ble_gatt_chr_flags>(BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_WRITE |
                                        BLE_GATT_CHR_F_WRITE_NO_RSP |
                                        BLE_GATT_CHR_F_NOTIFY),
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

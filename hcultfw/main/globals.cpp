/* Standard integer types used across ESP-IDF and NimBLE APIs. */
#include <stdint.h>

#include "globals.h"

static const char *kTag = "hcultfw";
const char *TAG = kTag;

// Q: What are these?
// A: These are the 128-bit BLE UUIDs loaded from the repo config at boot.
ble_uuid128_t g_service_uuid;
ble_uuid128_t g_characteristic_uuid;
bool g_ble_uuid_ok;

// Q: what are each of these?
// A: gatt_chr_handle: runtime handle to the characteristic value.
// A: g_sensor_value_1/2: cached ADC readings sent over BLE.
// A: boot_time_us: timestamp used to enforce the 45s window.
// A: adc_handle: ADC driver instance for oneshot reads.
// A: g_ble_addr_type: public vs random address used for advertising.
uint16_t gatt_chr_handle;
int g_sensor_value_1;
int g_sensor_value_2;
int64_t boot_time_us;
adc_oneshot_unit_handle_t adc_handle;
uint8_t g_ble_addr_type;
// Q: what is volatile?
// A: It tells the compiler the variable can change outside the current context.
// A: It prevents some optimizations, but it is not a full thread-safety mechanism.
volatile bool g_request_sleep;
volatile bool g_sent_payload;

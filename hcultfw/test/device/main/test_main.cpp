#include "unity.h"
#include "nvs_flash.h"

#include "ble_packet.h"
#include "sensor.h"
#include "state.h"

void setUp() {}
void tearDown() {}

static void test_packet_layout() {
  uint8_t buf[32] = {};
  size_t len = 0;
  const int raw[2] = {100, 200};
  const uint16_t mv[2] = {3300, 1650};

  TEST_ASSERT_TRUE(build_adv_mfg_data(buf, sizeof(buf), &len, raw, mv, 2, 0xDEADBEEF));
  TEST_ASSERT_EQUAL(18, len);
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[0]);
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[1]);
  TEST_ASSERT_EQUAL_HEX8('H',  buf[2]);
  TEST_ASSERT_EQUAL_HEX8('C',  buf[3]);
  TEST_ASSERT_EQUAL_HEX8(2,    buf[4]);
  TEST_ASSERT_EQUAL_HEX8(2,    buf[5]);
  TEST_ASSERT_EQUAL_HEX8(0xEF, buf[14]);
  TEST_ASSERT_EQUAL_HEX8(0xBE, buf[15]);
  TEST_ASSERT_EQUAL_HEX8(0xAD, buf[16]);
  TEST_ASSERT_EQUAL_HEX8(0xDE, buf[17]);
}

static void test_adc_raw_in_range() {
  FirmwareState state = {};
  init_adc(state);
  SensorReading r = read_sensor(state, state.sensor_channels[0], state.cali_handles[0]);
  TEST_ASSERT_GREATER_OR_EQUAL_INT(0, r.raw);
  TEST_ASSERT_LESS_OR_EQUAL_INT(4095, r.raw);
  TEST_ASSERT_LESS_OR_EQUAL_UINT16(3900, r.voltage_mv);
}

extern "C" void app_main() {
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ESP_ERROR_CHECK(nvs_flash_init());
  }

  UNITY_BEGIN();
  RUN_TEST(test_packet_layout);
  RUN_TEST(test_adc_raw_in_range);
  UNITY_END();
}

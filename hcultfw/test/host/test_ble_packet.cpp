#include "unity.h"
#include "ble_packet.h"

void setUp() {}
void tearDown() {}

static void test_packet_layout() {
  uint8_t buf[32] = {};
  size_t len = 0;
  const int raw[2] = {100, 200};
  const uint16_t mv[2] = {3300, 1650};

  TEST_ASSERT_TRUE(build_adv_mfg_data(buf, sizeof(buf), &len, raw, mv, 2, 0xDEADBEEF));
  TEST_ASSERT_EQUAL(18, len);

  // Company ID 0xFFFF little-endian
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[0]);
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[1]);
  // Magic
  TEST_ASSERT_EQUAL_HEX8('H', buf[2]);
  TEST_ASSERT_EQUAL_HEX8('C', buf[3]);
  // Version
  TEST_ASSERT_EQUAL_HEX8(2, buf[4]);
  // Sensor count
  TEST_ASSERT_EQUAL_HEX8(2, buf[5]);
  // Sensor 0 raw=100 little-endian
  TEST_ASSERT_EQUAL_HEX8(0x64, buf[6]);
  TEST_ASSERT_EQUAL_HEX8(0x00, buf[7]);
  // Sensor 0 mv=3300=0x0CE4 little-endian
  TEST_ASSERT_EQUAL_HEX8(0xE4, buf[8]);
  TEST_ASSERT_EQUAL_HEX8(0x0C, buf[9]);
  // Sensor 1 raw=200 little-endian
  TEST_ASSERT_EQUAL_HEX8(0xC8, buf[10]);
  TEST_ASSERT_EQUAL_HEX8(0x00, buf[11]);
  // Sensor 1 mv=1650=0x0672 little-endian
  TEST_ASSERT_EQUAL_HEX8(0x72, buf[12]);
  TEST_ASSERT_EQUAL_HEX8(0x06, buf[13]);
  // Token 0xDEADBEEF little-endian
  TEST_ASSERT_EQUAL_HEX8(0xEF, buf[14]);
  TEST_ASSERT_EQUAL_HEX8(0xBE, buf[15]);
  TEST_ASSERT_EQUAL_HEX8(0xAD, buf[16]);
  TEST_ASSERT_EQUAL_HEX8(0xDE, buf[17]);
}

static void test_buffer_too_small() {
  uint8_t buf[10];
  size_t len = 0;
  const int raw[2] = {0, 0};
  const uint16_t mv[2] = {0, 0};
  TEST_ASSERT_FALSE(build_adv_mfg_data(buf, sizeof(buf), &len, raw, mv, 2, 0));
}

static void test_max_values() {
  uint8_t buf[32] = {};
  size_t len = 0;
  const int raw[2] = {4095, 4095};
  const uint16_t mv[2] = {0xFFFF, 0xFFFF};

  TEST_ASSERT_TRUE(build_adv_mfg_data(buf, sizeof(buf), &len, raw, mv, 2, 0xFFFFFFFF));
  TEST_ASSERT_EQUAL(18, len);
  // raw=4095=0x0FFF
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[6]);
  TEST_ASSERT_EQUAL_HEX8(0x0F, buf[7]);
  // mv=0xFFFF
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[8]);
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[9]);
  // token=0xFFFFFFFF
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[14]);
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[15]);
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[16]);
  TEST_ASSERT_EQUAL_HEX8(0xFF, buf[17]);
}

static void test_zero_sensors() {
  uint8_t buf[32] = {};
  size_t len = 0;
  TEST_ASSERT_TRUE(build_adv_mfg_data(buf, sizeof(buf), &len, nullptr, nullptr, 0, 0));
  TEST_ASSERT_EQUAL(10, len); // 2 + 4 + 0 + 4
  TEST_ASSERT_EQUAL_HEX8(0, buf[5]); // sensor count = 0
}

int main() {
  UNITY_BEGIN();
  RUN_TEST(test_packet_layout);
  RUN_TEST(test_buffer_too_small);
  RUN_TEST(test_max_values);
  RUN_TEST(test_zero_sensors);
  return UNITY_END();
}

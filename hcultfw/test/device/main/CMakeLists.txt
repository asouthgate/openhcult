idf_component_register(
  SRCS
    "test_main.cpp"
    "../../../main/ble_packet.cpp"
    "../../../main/sensor.cpp"
  INCLUDE_DIRS "." "../../../main"
  REQUIRES unity esp_adc driver nvs_flash
)

if(DEFINED BOARD)
  target_compile_definitions(${COMPONENT_LIB} PUBLIC BOARD_${BOARD})
else()
  message(FATAL_ERROR "No BOARD defined. Use -DBOARD=FIREBEETLE_ESP32E or -DBOARD=FIREBEETLE2_ESP32C5")
endif()

set(BLE_CONFIG_PATH "${CMAKE_CURRENT_LIST_DIR}/../../openhcult.conf")
if(NOT EXISTS "${BLE_CONFIG_PATH}")
    message(FATAL_ERROR "Missing BLE config at ${BLE_CONFIG_PATH}")
endif()

# Parse the repo-level config to keep firmware UUIDs in sync with hcultmon.
file(STRINGS "${BLE_CONFIG_PATH}" BLE_CONFIG_LINES REGEX "^[a-z_]+=")
foreach(line IN LISTS BLE_CONFIG_LINES)
    if(line MATCHES "^ble_advertising_time_ms=([0-9]+)$")
        set(BLE_ADVERTISING_TIME_MS "${CMAKE_MATCH_1}")
    elseif(line MATCHES "^sleep_time_us=([0-9]+)$")
        set(SLEEP_TIME_US "${CMAKE_MATCH_1}")
    elseif(line MATCHES "^red_led_flash_ms=([0-9]+)$")
        set(RED_LED_FLASH_MS "${CMAKE_MATCH_1}")
    endif()
endforeach()

if(NOT DEFINED BLE_ADVERTISING_TIME_MS OR NOT DEFINED SLEEP_TIME_US OR NOT DEFINED RED_LED_FLASH_MS)
    message(FATAL_ERROR "BLE config must define ble_advertising_time_ms, sleep_time_us, and red_led_flash_ms")
endif()

# Generate a header with UUID strings for the firmware to parse at boot.
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/ble_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ble_config.h"
    @ONLY
)

idf_component_register(
    SRCS
        "main.cpp"
        "ble.cpp"
        "ble_packet.cpp"
        "sensor.cpp"
        "sleep.cpp"
        "state.cpp"
    INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}"
)

if(DEFINED BOARD)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC BOARD_${BOARD})
else()
    message(FATAL_ERROR "No BOARD defined. Use -DBOARD=FIREBEETLE_ESP32E or -DBOARD=FIREBEETLE2_ESP32C5")
endif()

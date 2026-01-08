set(BLE_CONFIG_PATH "${CMAKE_CURRENT_LIST_DIR}/../../openhcult.conf")
if(NOT EXISTS "${BLE_CONFIG_PATH}")
    message(FATAL_ERROR "Missing BLE config at ${BLE_CONFIG_PATH}")
endif()

# Parse the repo-level config to keep firmware UUIDs in sync with hcultmon.
file(STRINGS "${BLE_CONFIG_PATH}" BLE_CONFIG_LINES REGEX "^[a-z_]+=")
foreach(line IN LISTS BLE_CONFIG_LINES)
    if(line MATCHES "^service_uuid=(.+)$")
        set(BLE_SERVICE_UUID_STR "${CMAKE_MATCH_1}")
    elseif(line MATCHES "^characteristic_uuid=(.+)$")
        set(BLE_CHARACTERISTIC_UUID_STR "${CMAKE_MATCH_1}")
    elseif(line MATCHES "^ble_advertising_time_ms=([0-9]+)$")
        set(BLE_ADVERTISING_TIME_MS "${CMAKE_MATCH_1}")
    elseif(line MATCHES "^sleep_time_us=([0-9]+)$")
        set(SLEEP_TIME_US "${CMAKE_MATCH_1}")
    elseif(line MATCHES "^n_sleeps_per_transmission=([0-9]+)$")
        set(N_SLEEPS_PER_TRANSMISSION "${CMAKE_MATCH_1}")
    endif()
endforeach()

if(NOT BLE_SERVICE_UUID_STR OR NOT BLE_CHARACTERISTIC_UUID_STR)
    message(FATAL_ERROR "BLE config must define service_uuid and characteristic_uuid")
endif()

if(NOT DEFINED BLE_ADVERTISING_TIME_MS OR NOT DEFINED SLEEP_TIME_US OR NOT DEFINED N_SLEEPS_PER_TRANSMISSION)
    message(FATAL_ERROR "BLE config must define ble_advertising_time_ms, sleep_time_us, and n_sleeps_per_transmission")
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
        "payload.cpp"
        "sensor.cpp"
        "sleep.cpp"
        "state.cpp"
    INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}"
)

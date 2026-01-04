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
    endif()
endforeach()

if(NOT BLE_SERVICE_UUID_STR OR NOT BLE_CHARACTERISTIC_UUID_STR)
    message(FATAL_ERROR "BLE config must define service_uuid and characteristic_uuid")
endif()

# Generate a header with UUID strings for the firmware to parse at boot.
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/ble_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ble_config.h"
    @ONLY
)

idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}"
)

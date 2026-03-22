cmake_minimum_required(VERSION 3.16)
project(hcultfw_host_tests C CXX)

include(FetchContent)
FetchContent_Declare(
  unity
  GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
  GIT_TAG        v2.6.0
)
FetchContent_MakeAvailable(unity)

add_library(unity_lib STATIC ${unity_SOURCE_DIR}/src/unity.c)
target_include_directories(unity_lib PUBLIC ${unity_SOURCE_DIR}/src)

add_executable(test_ble_packet
  test_ble_packet.cpp
  ../../main/ble_packet.cpp
)
target_include_directories(test_ble_packet PRIVATE ../../main)
target_link_libraries(test_ble_packet PRIVATE unity_lib)

enable_testing()
add_test(NAME test_ble_packet COMMAND test_ble_packet)

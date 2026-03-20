#pragma once

#include "driver/gpio.h"

#if defined(BOARD_FIREBEETLE_ESP32E)

#define LED_PIN             21
#define SENSOR_POWER_PIN_1  23
#define SENSOR_PIN_1        GPIO_NUM_34
#define SENSOR_POWER_PIN_2  18
#define SENSOR_PIN_2        GPIO_NUM_35

#elif defined(BOARD_FIREBEETLE2_ESP32C5)

/* Sensor pins must be ADC-capable (ESP32-C5 ADC1: GPIO0-6) */
#define LED_PIN             15
#define SENSOR_POWER_PIN_1  7
#define SENSOR_PIN_1        GPIO_NUM_4
#define SENSOR_POWER_PIN_2  6
#define SENSOR_PIN_2        GPIO_NUM_5

#else
#error "No board defined. Use -DBOARD=FIREBEETLE_ESP32E or -DBOARD=FIREBEETLE2_ESP32C5"
#endif

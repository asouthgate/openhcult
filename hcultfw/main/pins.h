#pragma once

/* GPIO pin assignments for sensors and status LED. */
#include "driver/gpio.h"

// Q: should these be #defines?
// A: They can be macros or constexprs. In C++ I'd prefer constexpr, but ESP-IDF
// A: examples often use macros for pin/constant configuration.
#define LED_PIN 4
#define SENSOR_POWER_PIN_1 5
#define SENSOR_PIN_1 GPIO_NUM_34

#define SENSOR_POWER_PIN_2 18
#define SENSOR_PIN_2 GPIO_NUM_35

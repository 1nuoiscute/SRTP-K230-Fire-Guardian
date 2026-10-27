#ifndef SRTP_ADS1115_H
#define SRTP_ADS1115_H

#include "gpio_i2c.h"

#include <stdint.h>

/* The ADS1115 sits on the same bit-banged bus as SHT31/BMP280/SSD1306.
 * ADDR is tied to VDD on this board, so the address is 0x49 (not 0x48). */
#define ADS1115_ADDRESS 0x49

/* Single-ended input channels, encoded as the MUX field value.
 *
 * Config bits 14:12 must be 1xx for single-ended (AINx vs GND):
 *   100 = AIN0-GND, 101 = AIN1-GND, 110 = AIN2-GND, 111 = AIN3-GND
 * The 0xx patterns are DIFFERENTIAL pairs and will silently return the
 * difference of two pins instead of a voltage to ground.
 *
 * Board wiring (verified on the bench):
 *   AIN1 = flame sensor
 *   AIN2 = MQ-2 gas sensor
 */
typedef enum {
    ADS1115_AIN0 = 0x4000,  /* unconnected on this build  */
    ADS1115_AIN1 = 0x5000,  /* flame sensor               */
    ADS1115_AIN2 = 0x6000,  /* MQ-2 gas sensor            */
    ADS1115_AIN3 = 0x7000   /* unconnected on this build  */
} Ads1115Mux;

/* Read one single-ended channel.
 *
 * Returns 0 on success and stores the signed 16-bit conversion result in *raw.
 * At the +/-4.096 V PGA setting one LSB is 0.125 mV, so millivolts are
 * simply raw / 8.
 */
int ads1115_read_raw(GpioI2c *bus, Ads1115Mux channel, int16_t *raw);

/* Convenience wrapper: convert a raw reading to millivolts. */
static inline int ads1115_raw_to_mv(int16_t raw)
{
    return (int)raw / 8;
}

#endif

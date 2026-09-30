#ifndef SRTP_SHT31_H
#define SRTP_SHT31_H

#include "gpio_i2c.h"

typedef struct {
    double temperature_c;
    double humidity_percent;
} Sht31Sample;

/* One-shot, high repeatability, no clock stretching; checks both CRC bytes. */
int sht31_read(GpioI2c *bus, Sht31Sample *sample);

#endif

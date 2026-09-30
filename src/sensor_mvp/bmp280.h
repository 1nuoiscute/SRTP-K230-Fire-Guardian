#ifndef SRTP_BMP280_H
#define SRTP_BMP280_H

#include "gpio_i2c.h"

typedef struct {
    unsigned t1, p1;
    int t2, t3;
    int p2, p3, p4, p5, p6, p7, p8, p9;
    int initialized;
} Bmp280;

typedef struct {
    double temperature_c;
    double pressure_hpa;
} Bmp280Sample;

/* Checks chip ID 0x58, reads factory calibration, enables normal sampling. */
int bmp280_init(GpioI2c *bus, Bmp280 *sensor);
/* One six-byte burst keeps temperature and pressure from the same sample. */
int bmp280_read(GpioI2c *bus, const Bmp280 *sensor, Bmp280Sample *sample);

#endif

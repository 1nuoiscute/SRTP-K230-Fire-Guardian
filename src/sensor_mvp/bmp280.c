#include "bmp280.h"

#include <stdint.h>
#include <string.h>

#define BMP280_ADDRESS 0x76

static unsigned le16_unsigned(const uint8_t *p)
{
    return (unsigned)p[0] | ((unsigned)p[1] << 8);
}

static int le16_signed(const uint8_t *p)
{
    return (int16_t)le16_unsigned(p);
}

static int read_registers(GpioI2c *bus, uint8_t register_address,
                          uint8_t *data, size_t count)
{
    return gpio_i2c_write_read(bus, BMP280_ADDRESS, &register_address, 1,
                               data, count);
}

int bmp280_init(GpioI2c *bus, Bmp280 *sensor)
{
    uint8_t chip_id, calibration[24];
    const uint8_t control[2] = {0xF4, 0x27}; /* temp x1, pressure x1, normal */
    memset(sensor, 0, sizeof(*sensor));
    if (read_registers(bus, 0xD0, &chip_id, 1) || chip_id != 0x58 ||
        read_registers(bus, 0x88, calibration, sizeof(calibration)))
        return -1;

    sensor->t1 = le16_unsigned(calibration);
    sensor->t2 = le16_signed(calibration + 2);
    sensor->t3 = le16_signed(calibration + 4);
    sensor->p1 = le16_unsigned(calibration + 6);
    sensor->p2 = le16_signed(calibration + 8);
    sensor->p3 = le16_signed(calibration + 10);
    sensor->p4 = le16_signed(calibration + 12);
    sensor->p5 = le16_signed(calibration + 14);
    sensor->p6 = le16_signed(calibration + 16);
    sensor->p7 = le16_signed(calibration + 18);
    sensor->p8 = le16_signed(calibration + 20);
    sensor->p9 = le16_signed(calibration + 22);
    if (!sensor->p1 || gpio_i2c_write(bus, BMP280_ADDRESS,
                                      control, sizeof(control)))
        return -1;
    sensor->initialized = 1;
    gpio_i2c_delay_ms(20);
    return 0;
}

int bmp280_read(GpioI2c *bus, const Bmp280 *sensor, Bmp280Sample *sample)
{
    uint8_t raw[6];
    if (!sensor->initialized || read_registers(bus, 0xF7, raw, sizeof(raw)))
        return -1;
    unsigned adc_pressure = ((unsigned)raw[0] << 12) |
                            ((unsigned)raw[1] << 4) | (raw[2] >> 4);
    unsigned adc_temperature = ((unsigned)raw[3] << 12) |
                               ((unsigned)raw[4] << 4) | (raw[5] >> 4);

    /* Bosch BMP280 data sheet, appendix 8.1: double compensation. */
    double var1 = ((double)adc_temperature / 16384.0 -
                   (double)sensor->t1 / 1024.0) * sensor->t2;
    double delta = (double)adc_temperature / 131072.0 -
                   (double)sensor->t1 / 8192.0;
    double var2 = delta * delta * sensor->t3;
    int32_t t_fine = (int32_t)(var1 + var2);
    sample->temperature_c = (var1 + var2) / 5120.0;

    var1 = (double)t_fine / 2.0 - 64000.0;
    var2 = var1 * var1 * sensor->p6 / 32768.0;
    var2 += var1 * sensor->p5 * 2.0;
    var2 = var2 / 4.0 + sensor->p4 * 65536.0;
    var1 = (sensor->p3 * var1 * var1 / 524288.0 + sensor->p2 * var1) /
           524288.0;
    var1 = (1.0 + var1 / 32768.0) * sensor->p1;
    if (var1 == 0.0) return -1;
    double pressure_pa = 1048576.0 - adc_pressure;
    pressure_pa = (pressure_pa - var2 / 4096.0) * 6250.0 / var1;
    var1 = sensor->p9 * pressure_pa * pressure_pa / 2147483648.0;
    var2 = pressure_pa * sensor->p8 / 32768.0;
    pressure_pa += (var1 + var2 + sensor->p7) / 16.0;
    sample->pressure_hpa = pressure_pa / 100.0;
    return sample->pressure_hpa >= 300.0 && sample->pressure_hpa <= 1200.0
               ? 0 : -1;
}

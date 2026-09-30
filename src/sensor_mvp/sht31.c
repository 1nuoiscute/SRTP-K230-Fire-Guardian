#include "sht31.h"

#include <stdint.h>

#define SHT31_ADDRESS 0x44

static uint8_t crc8(const uint8_t *bytes)
{
    uint8_t crc = 0xff;
    for (int i = 0; i < 2; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc & 0x80U) ? (uint8_t)((crc << 1) ^ 0x31U)
                                : (uint8_t)(crc << 1);
    }
    return crc;
}

int sht31_read(GpioI2c *bus, Sht31Sample *sample)
{
    const uint8_t command[2] = {0x24, 0x00};
    uint8_t raw[6];
    if (gpio_i2c_write(bus, SHT31_ADDRESS, command, sizeof(command))) return -1;
    gpio_i2c_delay_ms(30);
    if (gpio_i2c_read(bus, SHT31_ADDRESS, raw, sizeof(raw))) return -1;
    if (crc8(raw) != raw[2] || crc8(raw + 3) != raw[5]) return -1;

    unsigned temp = ((unsigned)raw[0] << 8) | raw[1];
    unsigned humidity = ((unsigned)raw[3] << 8) | raw[4];
    sample->temperature_c = -45.0 + 175.0 * temp / 65535.0;
    sample->humidity_percent = 100.0 * humidity / 65535.0;
    return 0;
}

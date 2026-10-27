#include "ads1115.h"

/* Config register layout (datasheet section 9.6.3):
 *   bit15    OS   = 1   start a single conversion
 *   bit14-12 MUX  = single-ended channel select
 *   bit11-9  PGA  = 001  +/-4.096 V full scale
 *   bit8     MODE = 1    single-shot
 *   bit7-5   DR   = 100  128 samples per second
 *   bit4     COMP_MODE, bit3-2 COMP_POL/LAT, bit1-0 COMP_QUE = 11 comparator off
 *
 * Base word without the MUX: 0x8383.
 */
#define ADS1115_REG_CONVERSION 0x00
#define ADS1115_REG_CONFIG     0x01

static uint16_t config_for(Ads1115Mux channel)
{
    return (uint16_t)(0x8000u | (uint16_t)channel | 0x0200u | 0x0100u | 0x0080u | 0x0003u);
}

int ads1115_read_raw(GpioI2c *bus, Ads1115Mux channel, int16_t *raw)
{
    const uint16_t config = config_for(channel);
    const uint8_t command[3] = {
        ADS1115_REG_CONFIG,
        (uint8_t)(config >> 8),
        (uint8_t)(config & 0xFFu),
    };
    if (gpio_i2c_write(bus, ADS1115_ADDRESS, command, sizeof(command))) return -1;

    /* One conversion at 128 SPS takes ~7.8 ms; the bit-banged bus is far
     * slower than that anyway, but a short wait keeps the intent explicit. */
    gpio_i2c_delay_ms(10);

    const uint8_t pointer = ADS1115_REG_CONVERSION;
    uint8_t data[2];
    if (gpio_i2c_write_read(bus, ADS1115_ADDRESS, &pointer, 1, data, sizeof(data))) return -1;

    const uint16_t value = (uint16_t)(((uint16_t)data[0] << 8) | data[1]);
    /* Inputs are single-ended (0..+4.096 V) so the code is never negative in
     * practice, but keep the sign interpretation honest. */
    *raw = (int16_t)value;
    return 0;
}

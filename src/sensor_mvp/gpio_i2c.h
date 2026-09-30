#ifndef SRTP_GPIO_I2C_H
#define SRTP_GPIO_I2C_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    volatile uint32_t *iomux;
    volatile uint32_t *gpio;
    uint32_t saved_mux44, saved_mux45;
    uint32_t saved_data, saved_direction;
    unsigned half_period_us;
    int active;
} GpioI2c;

/* GPIO44=SCL and GPIO45=SDA. The bus is released on close. */
int gpio_i2c_open(GpioI2c *bus, unsigned half_period_us);
void gpio_i2c_close(GpioI2c *bus);

int gpio_i2c_write(GpioI2c *bus, uint8_t address,
                   const uint8_t *data, size_t length);
int gpio_i2c_read(GpioI2c *bus, uint8_t address,
                  uint8_t *data, size_t length);
int gpio_i2c_write_read(GpioI2c *bus, uint8_t address,
                        const uint8_t *command, size_t command_length,
                        uint8_t *data, size_t length);
void gpio_i2c_delay_ms(unsigned milliseconds);

#endif

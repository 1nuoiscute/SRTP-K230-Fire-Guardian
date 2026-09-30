#include "gpio_i2c.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define IOMUX_BASE 0x91105000UL
#define GPIO1_BASE 0x9140C000UL
#define GPIO_SCL (1U << 12)
#define GPIO_SDA (1U << 13)
#define PAD_INPUT (1U << 31)

static void delay_us(unsigned microseconds)
{
    if (microseconds >= 1000) {
        struct timespec pause = {
            microseconds / 1000000U,
            (microseconds % 1000000U) * 1000U
        };
        nanosleep(&pause, NULL);
        return;
    }
    struct timespec begin, now;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    do {
        clock_gettime(CLOCK_MONOTONIC, &now);
    } while ((now.tv_sec - begin.tv_sec) * 1000000L +
             (now.tv_nsec - begin.tv_nsec) / 1000L < microseconds);
}

void gpio_i2c_delay_ms(unsigned milliseconds)
{
    delay_us(milliseconds * 1000U);
}

static int level(GpioI2c *bus, int pad)
{
    return !!(bus->iomux[pad] & PAD_INPUT);
}

/* GPIO output data stays zero; direction=0 releases each open-drain wire. */
static void pull(GpioI2c *bus, uint32_t pin, int low)
{
    uint32_t direction = bus->gpio[1];
    bus->gpio[1] = low ? direction | pin : direction & ~pin;
    (void)bus->gpio[1];
    delay_us(bus->half_period_us);
}

static int clock_high(GpioI2c *bus)
{
    pull(bus, GPIO_SCL, 0);
    return level(bus, 44) ? 0 : -1;
}

static int start_condition(GpioI2c *bus)
{
    pull(bus, GPIO_SDA, 0);
    if (clock_high(bus) || !level(bus, 45)) return -1;
    pull(bus, GPIO_SDA, 1);
    pull(bus, GPIO_SCL, 1);
    return 0;
}

static void stop_condition(GpioI2c *bus)
{
    pull(bus, GPIO_SDA, 1);
    (void)clock_high(bus);
    pull(bus, GPIO_SDA, 0);
}

/* Returns 1=ACK, 0=NACK, -1=electrical line error. */
static int write_byte(GpioI2c *bus, uint8_t value)
{
    for (int bit = 7; bit >= 0; --bit) {
        int one = !!(value & (1U << bit));
        pull(bus, GPIO_SDA, !one);
        if (clock_high(bus) || (one && !level(bus, 45))) return -1;
        pull(bus, GPIO_SCL, 1);
    }
    pull(bus, GPIO_SDA, 0);
    if (clock_high(bus)) return -1;
    int ack = !level(bus, 45);
    pull(bus, GPIO_SCL, 1);
    return ack;
}

static int read_byte(GpioI2c *bus, int acknowledge)
{
    int value = 0;
    pull(bus, GPIO_SDA, 0);
    for (int bit = 7; bit >= 0; --bit) {
        if (clock_high(bus)) return -1;
        value |= level(bus, 45) << bit;
        pull(bus, GPIO_SCL, 1);
    }
    pull(bus, GPIO_SDA, acknowledge);
    if (clock_high(bus)) return -1;
    pull(bus, GPIO_SCL, 1);
    pull(bus, GPIO_SDA, 0);
    return value;
}

int gpio_i2c_open(GpioI2c *bus, unsigned half_period_us)
{
    memset(bus, 0, sizeof(*bus));
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return -1; }
    void *iomux = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, IOMUX_BASE);
    void *gpio = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, GPIO1_BASE);
    close(fd);
    if (iomux == MAP_FAILED || gpio == MAP_FAILED) {
        perror("mmap GPIO/IOMUX");
        if (iomux != MAP_FAILED) munmap(iomux, 4096);
        if (gpio != MAP_FAILED) munmap(gpio, 4096);
        return -1;
    }
    bus->iomux = iomux;
    bus->gpio = gpio;
    bus->saved_mux44 = bus->iomux[44];
    bus->saved_mux45 = bus->iomux[45];
    bus->saved_data = bus->gpio[0];
    bus->saved_direction = bus->gpio[1];
    if (((bus->saved_mux44 >> 11) & 7U) ||
        ((bus->saved_mux45 >> 11) & 7U) ||
        (bus->saved_direction & (GPIO_SCL | GPIO_SDA))) {
        fprintf(stderr, "GPIO44/45 must be idle GPIO inputs\n");
        munmap(iomux, 4096);
        munmap(gpio, 4096);
        return -1;
    }
    bus->half_period_us = half_period_us ? half_period_us : 30;
    bus->active = 1;
    bus->gpio[0] = bus->saved_data & ~(GPIO_SCL | GPIO_SDA);
    bus->gpio[1] = bus->saved_direction & ~(GPIO_SCL | GPIO_SDA);
    bus->iomux[44] = bus->saved_mux44 & ~PAD_INPUT;
    bus->iomux[45] = bus->saved_mux45 & ~PAD_INPUT;
    if (!level(bus, 44) || !level(bus, 45)) {
        fprintf(stderr, "I2C bus is not idle high\n");
        gpio_i2c_close(bus);
        return -1;
    }
    return 0;
}

void gpio_i2c_close(GpioI2c *bus)
{
    if (!bus->active) return;
    bus->gpio[1] = (bus->gpio[1] & ~(GPIO_SCL | GPIO_SDA)) |
                    (bus->saved_direction & (GPIO_SCL | GPIO_SDA));
    bus->gpio[0] = (bus->gpio[0] & ~(GPIO_SCL | GPIO_SDA)) |
                    (bus->saved_data & (GPIO_SCL | GPIO_SDA));
    bus->iomux[44] = bus->saved_mux44 & ~PAD_INPUT;
    bus->iomux[45] = bus->saved_mux45 & ~PAD_INPUT;
    munmap((void *)bus->iomux, 4096);
    munmap((void *)bus->gpio, 4096);
    bus->active = 0;
}

int gpio_i2c_write(GpioI2c *bus, uint8_t address,
                   const uint8_t *data, size_t length)
{
    if (start_condition(bus)) return -1;
    if (write_byte(bus, (uint8_t)(address << 1)) != 1) {
        stop_condition(bus);
        return -1;
    }
    for (size_t i = 0; i < length; ++i) {
        if (write_byte(bus, data[i]) != 1) {
            stop_condition(bus);
            return -1;
        }
    }
    stop_condition(bus);
    return 0;
}

int gpio_i2c_read(GpioI2c *bus, uint8_t address,
                  uint8_t *data, size_t length)
{
    if (!length || start_condition(bus)) return -1;
    if (write_byte(bus, (uint8_t)((address << 1) | 1U)) != 1) {
        stop_condition(bus);
        return -1;
    }
    for (size_t i = 0; i < length; ++i) {
        int byte = read_byte(bus, i + 1 < length);
        if (byte < 0) { stop_condition(bus); return -1; }
        data[i] = (uint8_t)byte;
    }
    stop_condition(bus);
    return 0;
}

int gpio_i2c_write_read(GpioI2c *bus, uint8_t address,
                        const uint8_t *command, size_t command_length,
                        uint8_t *data, size_t length)
{
    if (!command_length || !length || start_condition(bus)) return -1;
    if (write_byte(bus, (uint8_t)(address << 1)) != 1) {
        stop_condition(bus);
        return -1;
    }
    for (size_t i = 0; i < command_length; ++i) {
        if (write_byte(bus, command[i]) != 1) {
            stop_condition(bus);
            return -1;
        }
    }
    if (start_condition(bus) ||
        write_byte(bus, (uint8_t)((address << 1) | 1U)) != 1) {
        stop_condition(bus);
        return -1;
    }
    for (size_t i = 0; i < length; ++i) {
        int byte = read_byte(bus, i + 1 < length);
        if (byte < 0) { stop_condition(bus); return -1; }
        data[i] = (uint8_t)byte;
    }
    stop_condition(bus);
    return 0;
}

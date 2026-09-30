#ifndef SRTP_SSD1306_H
#define SRTP_SSD1306_H

#include "gpio_i2c.h"

#define OLED_WIDTH 128
#define OLED_HEIGHT 64
#define OLED_BYTES (OLED_WIDTH * OLED_HEIGHT / 8)

typedef struct {
    GpioI2c *bus;
    uint8_t address;
    uint8_t pixels[OLED_BYTES];
    uint8_t displayed[OLED_BYTES];
} Ssd1306;

int ssd1306_init(Ssd1306 *screen, GpioI2c *bus);
void ssd1306_clear(Ssd1306 *screen);
void ssd1306_text(Ssd1306 *screen, int x, int y, const char *text);
/* Sends only pages whose pixels changed since the last successful refresh. */
int ssd1306_refresh(Ssd1306 *screen);

#endif

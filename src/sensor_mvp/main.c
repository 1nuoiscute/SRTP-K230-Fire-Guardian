#include "bmp280.h"
#include "gpio_i2c.h"
#include "sht31.h"
#include "ssd1306.h"

#include <signal.h>
#include <stdio.h>

static volatile sig_atomic_t running = 1;

static void request_stop(int signal_number)
{
    (void)signal_number;
    running = 0;
}

static void draw_readings(Ssd1306 *screen,
                          int sht_ok, const Sht31Sample *sht,
                          int bmp_ok, const Bmp280Sample *bmp)
{
    char line[32];
    ssd1306_clear(screen);

    if (sht_ok) {
        snprintf(line, sizeof(line), "SHT T %.1fC", sht->temperature_c);
        ssd1306_text(screen, 2, 1, line);
        snprintf(line, sizeof(line), "SHT H %.1f%%", sht->humidity_percent);
        ssd1306_text(screen, 2, 13, line);
    } else {
        ssd1306_text(screen, 2, 1, "SHT31 ERR");
    }

    if (bmp_ok) {
        snprintf(line, sizeof(line), "BMP T %.1fC", bmp->temperature_c);
        ssd1306_text(screen, 2, 25, line);
        snprintf(line, sizeof(line), "BMP P %.1fHPA", bmp->pressure_hpa);
        ssd1306_text(screen, 2, 37, line);
    } else {
        ssd1306_text(screen, 2, 25, "BMP280 ERR");
    }

    /* Reserved for RW007 connectivity or RSSI after the Wi-Fi test. */
    ssd1306_text(screen, 2, 49, "WIFI --");
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, request_stop);
    signal(SIGTERM, request_stop);

    GpioI2c bus;
    if (gpio_i2c_open(&bus, 30)) return 1;
    Ssd1306 screen;
    if (ssd1306_init(&screen, &bus)) {
        fprintf(stderr, "SSD1306 not found at 0x3C/0x3D\n");
        gpio_i2c_close(&bus);
        return 1;
    }
    printf("SSD1306 address: 0x%02X\n", screen.address);

    Bmp280 bmp;
    int bmp_ready = bmp280_init(&bus, &bmp) == 0;
    for (; running; gpio_i2c_delay_ms(1000)) {
        Sht31Sample sht_sample = {0};
        Bmp280Sample bmp_sample = {0};
        int sht_ok = sht31_read(&bus, &sht_sample) == 0;
        if (!bmp_ready) bmp_ready = bmp280_init(&bus, &bmp) == 0;
        int bmp_ok = bmp_ready && bmp280_read(&bus, &bmp, &bmp_sample) == 0;
        if (!bmp_ok) bmp_ready = 0;

        if (sht_ok)
            printf("SHT31: %.2f C, %.2f %%RH (CRC OK)\n",
                   sht_sample.temperature_c, sht_sample.humidity_percent);
        else fprintf(stderr, "SHT31 read/CRC failed\n");
        if (bmp_ok)
            printf("BMP280: %.2f C, %.2f hPa\n",
                   bmp_sample.temperature_c, bmp_sample.pressure_hpa);
        else fprintf(stderr, "BMP280 read failed\n");

        draw_readings(&screen, sht_ok, &sht_sample, bmp_ok, &bmp_sample);
        if (ssd1306_refresh(&screen)) {
            fprintf(stderr, "SSD1306 refresh failed\n");
            break;
        }
    }
    ssd1306_clear(&screen);
    (void)ssd1306_refresh(&screen);
    gpio_i2c_close(&bus);
    return 0;
}

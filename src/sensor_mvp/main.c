#include "ads1115.h"
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

/* Alarm thresholds, in millivolts at the ADC pin.
 *
 * Placeholders derived from the clean-air readings measured on the bench
 * Bench data: MQ-2 idles around 215 mV and reached ~1220 mV with lighter gas,
 * so 500 mV separates clean air from a leak with margin. The flame sensor on
 * AIN1 sits at ~816 mV and did not respond to a flame during bring-up, so its
 * threshold is parked high to avoid a permanent false alarm; re-tune it once
 * the module is fixed. */
#define MQ2_ALARM_MV   500
#define FLAME_ALARM_MV 1200

static void draw_readings(Ssd1306 *screen,
                          int sht_ok, const Sht31Sample *sht,
                          int bmp_ok, const Bmp280Sample *bmp,
                          int mq_ok, int mq_mv,
                          int flame_ok, int flame_mv)
{
    char line[32];
    ssd1306_clear(screen);

    if (sht_ok) {
        snprintf(line, sizeof(line), "SHT T %.1fC", sht->temperature_c);
        ssd1306_text(screen, 2, 0, line);
        snprintf(line, sizeof(line), "SHT H %.1f%%", sht->humidity_percent);
        ssd1306_text(screen, 2, 10, line);
    } else {
        ssd1306_text(screen, 2, 0, "SHT31 ERR");
    }

    if (bmp_ok) {
        snprintf(line, sizeof(line), "BMP %.1fhPa", bmp->pressure_hpa);
        ssd1306_text(screen, 2, 20, line);
    } else {
        ssd1306_text(screen, 2, 20, "BMP280 ERR");
    }

    if (mq_ok) {
        snprintf(line, sizeof(line), "MQ2   %dmV", mq_mv);
        ssd1306_text(screen, 2, 30, line);
    } else {
        ssd1306_text(screen, 2, 30, "MQ2   ERR");
    }

    if (flame_ok) {
        snprintf(line, sizeof(line), "FLAME %dmV", flame_mv);
        ssd1306_text(screen, 2, 40, line);
    } else {
        ssd1306_text(screen, 2, 40, "FLAME ERR");
    }

    const int gas_alarm = mq_ok && mq_mv >= MQ2_ALARM_MV;
    const int fire_alarm = flame_ok && flame_mv >= FLAME_ALARM_MV;
    if (gas_alarm && fire_alarm)      snprintf(line, sizeof(line), "ALARM GAS+FIRE");
    else if (gas_alarm)               snprintf(line, sizeof(line), "ALARM GAS");
    else if (fire_alarm)              snprintf(line, sizeof(line), "ALARM FIRE");
    else                              snprintf(line, sizeof(line), "STATUS OK");
    ssd1306_text(screen, 2, 50, line);
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
    int adc_ok = 1;
    /* Consecutive OLED refresh failures. A display glitch must never stop the
     * sampling loop: the gas channel is the safety-critical part and the
     * fire_guard treats a missing sample as "close the valve", so dying here
     * would masquerade as a gas alarm. */
    int oled_fail = 0;

    for (; running; gpio_i2c_delay_ms(1000)) {
        Sht31Sample sht_sample = {0};
        Bmp280Sample bmp_sample = {0};

        const int sht_ok = sht31_read(&bus, &sht_sample) == 0;
        if (!bmp_ready) bmp_ready = bmp280_init(&bus, &bmp) == 0;
        int bmp_ok = bmp_ready && bmp280_read(&bus, &bmp, &bmp_sample) == 0;
        if (!bmp_ok) bmp_ready = 0;

        int16_t mq_raw = 0, flame_raw = 0;
        /* Verified wiring: MQ-2 sits on AIN2, the flame sensor on AIN1. */
        const int mq_ok = adc_ok && ads1115_read_raw(&bus, ADS1115_AIN2, &mq_raw) == 0;
        const int flame_ok = adc_ok && ads1115_read_raw(&bus, ADS1115_AIN1, &flame_raw) == 0;
        if (!mq_ok && !flame_ok) adc_ok = 0;   /* stop hammering a dead bus */
        const int mq_mv = ads1115_raw_to_mv(mq_raw);
        const int flame_mv = ads1115_raw_to_mv(flame_raw);

        /* These two lines keep the existing status_bridge contract intact. */
        if (sht_ok)
            printf("SHT31: %.2f C, %.2f %%RH (CRC OK)\n",
                   sht_sample.temperature_c, sht_sample.humidity_percent);
        else fprintf(stderr, "SHT31 read/CRC failed\n");
        if (bmp_ok)
            printf("BMP280: %.2f C, %.2f hPa\n",
                   bmp_sample.temperature_c, bmp_sample.pressure_hpa);
        else fprintf(stderr, "BMP280 read failed\n");

        /* Extra lines for the gas/flame channels. status_bridge ignores any
         * line it does not recognise, so this stays backward compatible. */
        if (mq_ok)    printf("MQ2: %d mV\n", mq_mv);
        else          fprintf(stderr, "MQ2 read failed\n");
        if (flame_ok) printf("FLAME: %d mV\n", flame_mv);
        else          fprintf(stderr, "FLAME read failed\n");

        draw_readings(&screen, sht_ok, &sht_sample, bmp_ok, &bmp_sample,
                      mq_ok, mq_mv, flame_ok, flame_mv);
        if (ssd1306_refresh(&screen)) {
            /* Tolerate transient I2C glitches (the relay switching on the bench
             * cause them). Re-init the panel once after three in a row, but
             * keep sampling no matter what. */
            ++oled_fail;
            if (oled_fail == 3) {
                fprintf(stderr, "SSD1306 refresh failed 3x, re-initialising\n");
                (void)ssd1306_init(&screen, &bus);
            } else if (oled_fail < 3) {
                fprintf(stderr, "SSD1306 refresh failed (%d)\n", oled_fail);
            }
        } else {
            oled_fail = 0;
        }
    }
    ssd1306_clear(&screen);
    (void)ssd1306_refresh(&screen);
    gpio_i2c_close(&bus);
    return 0;
}


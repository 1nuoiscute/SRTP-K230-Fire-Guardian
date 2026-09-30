/* K230 SensorFusion Shield: live SHT31 temperature and humidity on SSD1306.
 * Runs on the Linux small core; GPIO44/45 are bit-banged as open-drain I2C.
 * Based on the verified one-shot SHT31 probe in this directory.
 */
#define main sht31_probe_original_main
#include "sht31_gpio_probe.c"
#undef main
#include <string.h>

#define WIDTH 128
#define HEIGHT 64
static uint8_t frame[WIDTH * HEIGHT / 8];
static uint8_t sent[WIDTH * HEIGHT / 8];
static unsigned oled_addr;
static unsigned bmp_t1;
static int bmp_t2, bmp_t3;

static int packet(int is_data, const uint8_t *data, size_t n)
{
    if (start()) return -1;
    if (write_byte((uint8_t)(oled_addr << 1)) != 1 ||
        write_byte(is_data ? 0x40 : 0x00) != 1) {
        stop();
        return -1;
    }
    for (size_t i = 0; i < n; ++i) {
        if (write_byte(data[i]) != 1) {
            stop();
            return -1;
        }
    }
    stop();
    return 0;
}

static int oled_detect(void)
{
    for (unsigned addr = 0x3c; addr <= 0x3d; ++addr) {
        if (start()) return -1;
        int ack = write_byte((uint8_t)(addr << 1));
        stop();
        printf("OLED 0x%02X: %s\n", addr,
               ack < 0 ? "line error" : ack ? "ACK" : "NACK");
        if (ack == 1) { oled_addr = addr; return 0; }
    }
    return -1;
}

static int oled_init(void)
{
    static const uint8_t init[] = {
        0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40,
        0x8D, 0x14, 0x20, 0x02, 0xA1, 0xC8, 0xDA, 0x12,
        0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0xAF
    };
    return packet(0, init, sizeof(init));
}

static int bmp_write_reg(uint8_t reg, uint8_t value)
{
    if (start()) return -1;
    int ok = write_byte(0x76 << 1) == 1 &&
             write_byte(reg) == 1 && write_byte(value) == 1;
    stop();
    return ok ? 0 : -1;
}

static int bmp_read_burst(uint8_t reg, uint8_t *out, int count)
{
    if (start()) return -1;
    if (write_byte(0x76 << 1) != 1 || write_byte(reg) != 1 || start() ||
        write_byte((0x76 << 1) | 1) != 1) {
        stop();
        return -1;
    }
    for (int i = 0; i < count; ++i) {
        int b = read_byte(i + 1 < count);
        if (b < 0) { stop(); return -1; }
        out[i] = (uint8_t)b;
    }
    stop();
    return 0;
}

static int bmp_init(void)
{
    uint8_t id, cal[6];
    if (bmp_read_burst(0xD0, &id, 1) || id != 0x58 ||
        bmp_read_burst(0x88, cal, 6) || bmp_write_reg(0xF4, 0x27))
        return -1;
    bmp_t1 = (unsigned)cal[0] | ((unsigned)cal[1] << 8);
    bmp_t2 = (int16_t)((unsigned)cal[2] | ((unsigned)cal[3] << 8));
    bmp_t3 = (int16_t)((unsigned)cal[4] | ((unsigned)cal[5] << 8));
    printf("BMP280 ID=0x58, calibration T1=%u T2=%d T3=%d\n",
           bmp_t1, bmp_t2, bmp_t3);
    return 0;
}

static int bmp_temperature(double *temp_c)
{
    uint8_t raw[3];
    if (bmp_read_burst(0xFA, raw, 3)) return -1;
    unsigned adc = ((unsigned)raw[0] << 12) |
                   ((unsigned)raw[1] << 4) | (raw[2] >> 4);
    double v1 = ((double)adc / 16384.0 - (double)bmp_t1 / 1024.0) * bmp_t2;
    double d = (double)adc / 131072.0 - (double)bmp_t1 / 8192.0;
    double v2 = d * d * bmp_t3;
    *temp_c = (v1 + v2) / 5120.0;
    return 0;
}

static const uint8_t *glyph(char ch)
{
    static const uint8_t digits[10][5] = {
        {0x3E,0x51,0x49,0x45,0x3E}, {0x00,0x42,0x7F,0x40,0x00},
        {0x42,0x61,0x51,0x49,0x46}, {0x21,0x41,0x45,0x4B,0x31},
        {0x18,0x14,0x12,0x7F,0x10}, {0x27,0x45,0x45,0x45,0x39},
        {0x3C,0x4A,0x49,0x49,0x30}, {0x01,0x71,0x09,0x05,0x03},
        {0x36,0x49,0x49,0x49,0x36}, {0x06,0x49,0x49,0x29,0x1E}
    };
    static const uint8_t blank[5] = {0,0,0,0,0};
    static const uint8_t dot[5] = {0,0x60,0x60,0,0};
    static const uint8_t colon[5] = {0,0x36,0x36,0,0};
    static const uint8_t percent[5] = {0x63,0x13,0x08,0x64,0x63};
    static const uint8_t minus[5] = {0x08,0x08,0x08,0x08,0x08};
    static const uint8_t letters[26][5] = {
        {0x7E,0x11,0x11,0x11,0x7E}, {0x7F,0x49,0x49,0x49,0x36},
        {0x3E,0x41,0x41,0x41,0x22}, {0x7F,0x41,0x41,0x22,0x1C},
        {0x7F,0x49,0x49,0x49,0x41}, {0x7F,0x09,0x09,0x09,0x01},
        {0x3E,0x41,0x49,0x49,0x7A}, {0x7F,0x08,0x08,0x08,0x7F},
        {0x00,0x41,0x7F,0x41,0x00}, {0x20,0x40,0x41,0x3F,0x01},
        {0x7F,0x08,0x14,0x22,0x41}, {0x7F,0x40,0x40,0x40,0x40},
        {0x7F,0x02,0x0C,0x02,0x7F}, {0x7F,0x04,0x08,0x10,0x7F},
        {0x3E,0x41,0x41,0x41,0x3E}, {0x7F,0x09,0x09,0x09,0x06},
        {0x3E,0x41,0x51,0x21,0x5E}, {0x7F,0x09,0x19,0x29,0x46},
        {0x46,0x49,0x49,0x49,0x31}, {0x01,0x01,0x7F,0x01,0x01},
        {0x3F,0x40,0x40,0x40,0x3F}, {0x1F,0x20,0x40,0x20,0x1F},
        {0x7F,0x20,0x18,0x20,0x7F}, {0x63,0x14,0x08,0x14,0x63},
        {0x03,0x04,0x78,0x04,0x03}, {0x61,0x51,0x49,0x45,0x43}
    };
    if (ch >= '0' && ch <= '9') return digits[ch - '0'];
    if (ch >= 'A' && ch <= 'Z') return letters[ch - 'A'];
    if (ch == '.') return dot;
    if (ch == ':') return colon;
    if (ch == '%') return percent;
    if (ch == '-') return minus;
    return blank;
}

static void pixel(int x, int y)
{
    if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT)
        frame[(y / 8) * WIDTH + x] |= (uint8_t)(1U << (y % 8));
}

static void text_line(int x, int y, const char *s, int scale)
{
    for (; *s; ++s, x += 6 * scale) {
        const uint8_t *g = glyph(*s);
        for (int col = 0; col < 5; ++col)
            for (int row = 0; row < 7; ++row)
                if (g[col] & (1U << row))
                    for (int dx = 0; dx < scale; ++dx)
                        for (int dy = 0; dy < scale; ++dy)
                            pixel(x + col * scale + dx, y + row * scale + dy);
    }
}

static int oled_update(int force)
{
    for (int page = 0; page < 8; ++page) {
        const uint8_t *row = frame + page * WIDTH;
        uint8_t *previous = sent + page * WIDTH;
        if (!force && memcmp(row, previous, WIDTH) == 0) continue;
        uint8_t pos[] = {(uint8_t)(0xB0 | page), 0x00, 0x10};
        if (packet(0, pos, sizeof(pos)) || packet(1, row, WIDTH)) {
            fprintf(stderr, "OLED page %d write failed\n", page);
            return -1;
        }
        memcpy(previous, row, WIDTH);
    }
    return 0;
}

static void compose(int sht_ok, int bmp_ok, double bmp_c)
{
    char line[32];
    memset(frame, 0, sizeof(frame));
    if (sht_ok) {
        snprintf(line, sizeof(line), "SHT31 T %.1fC", last_temp_c);
        text_line(3, 2, line, 1);
        snprintf(line, sizeof(line), "SHT31 H %.1f%%", last_humidity);
        text_line(3, 23, line, 1);
    } else {
        text_line(3, 2, "SHT31 ERR", 1);
    }
    if (bmp_ok) snprintf(line, sizeof(line), "BMP280 T %.1fC", bmp_c);
    else snprintf(line, sizeof(line), "BMP280 ERR");
    text_line(3, 44, line, 1);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem"); return 1; }
    void *a = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IOMUX_BASE);
    void *b = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, GPIO1_BASE);
    close(fd);
    if (a == MAP_FAILED || b == MAP_FAILED) { perror("mmap"); return 1; }
    mux = a; gpio = b;
    old44 = mux[44]; old45 = mux[45];
    old_dr = gpio[0]; old_ddr = gpio[1];
    if (((old44 >> 11) & 7) || ((old45 >> 11) & 7) ||
        (old_ddr & (SCL | SDA))) {
        fprintf(stderr, "GPIO44/45 not idle GPIO inputs; abort\n");
        return 2;
    }
    ready = 1;
    atexit(restore);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    gpio[0] = old_dr & ~(SCL | SDA);
    gpio[1] = old_ddr & ~(SCL | SDA);
    mux[44] = old44 & ~DI;
    mux[45] = old45 & ~DI;
    bit_delay_us = 30;
    quiet = 1;
    printf("I2C idle: SCL=%d SDA=%d\n", level(44), level(45));
    if (oled_detect() || oled_init()) {
        fprintf(stderr, "OLED init failed\n");
        return 1;
    }
    memset(sent, 0xff, sizeof(sent));
    int bmp_ready = bmp_init() == 0;
    for (;;) {
        int sht_ok = probe(0x44) == 0;
        if (sht_ok) printf("SHT31 %.2f C, %.2f %%RH (CRC OK)\n",
                           last_temp_c, last_humidity);
        else fprintf(stderr, "SHT31 read failed\n");
        if (!bmp_ready) bmp_ready = bmp_init() == 0;
        double bmp_c = 0.0;
        int bmp_ok = bmp_ready && bmp_temperature(&bmp_c) == 0;
        if (bmp_ok) printf("BMP280 %.2f C\n", bmp_c);
        else fprintf(stderr, "BMP280 read failed\n");
        compose(sht_ok, bmp_ok, bmp_c);
        if (oled_update(0)) return 1;
        delay_us(1000000);
    }
}

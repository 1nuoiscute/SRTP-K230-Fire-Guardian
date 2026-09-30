/* Temporary, low-speed BMP280 I2C probe for the K230 SensorFusion Shield.
 * Runs on the Linux small core, uses GPIO44=SCL and GPIO45=SDA, and restores
 * the original GPIO/IOMUX registers before exit. It does not modify firmware.
 */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define IOMUX_BASE 0x91105000UL
#define GPIO1_BASE 0x9140c000UL
#define GPIO_SCL (1U << 12)
#define GPIO_SDA (1U << 13)
#define INPUT_BIT (1U << 31)

static volatile uint32_t *iomux;
static volatile uint32_t *gpio;
static uint32_t saved_mux44, saved_mux45, saved_dr, saved_ddr;
static int mapped;

static void delay_us(long microseconds)
{
    struct timespec ts = {0, microseconds * 1000};
    nanosleep(&ts, NULL);
}

static void restore(void)
{
    if (!mapped) return;
    gpio[1] = (gpio[1] & ~(GPIO_SCL | GPIO_SDA)) |
              (saved_ddr & (GPIO_SCL | GPIO_SDA));
    gpio[0] = (gpio[0] & ~(GPIO_SCL | GPIO_SDA)) |
              (saved_dr & (GPIO_SCL | GPIO_SDA));
    iomux[44] = saved_mux44 & ~INPUT_BIT;
    iomux[45] = saved_mux45 & ~INPUT_BIT;
}

static void on_signal(int signo)
{
    restore();
    _exit(128 + signo);
}

static void drive(uint32_t mask, int low)
{
    uint32_t ddr = gpio[1];
    gpio[1] = low ? (ddr | mask) : (ddr & ~mask);
    (void)gpio[1];
    delay_us(1000);
}

static int level(int pin)
{
    return !!(iomux[pin] & INPUT_BIT);
}

static int clock_high(void)
{
    drive(GPIO_SCL, 0);
    if (!level(44)) {
        fprintf(stderr, "SCL stayed low after release: DDR=0x%08X PAD44=0x%08X GPIO_EXT=0x%08X\n",
                gpio[1], iomux[44], gpio[0x50 / 4]);
        return -1;
    }
    return 0;
}

static int start_condition(void)
{
    drive(GPIO_SDA, 0);
    if (clock_high() || !level(45)) {
        fprintf(stderr, "bus is not idle high\n");
        return -1;
    }
    drive(GPIO_SDA, 1);
    drive(GPIO_SCL, 1);
    return 0;
}

static int repeated_start(void)
{
    drive(GPIO_SDA, 0);
    if (clock_high()) return -1;
    drive(GPIO_SDA, 1);
    drive(GPIO_SCL, 1);
    return 0;
}

static void stop_condition(void)
{
    drive(GPIO_SDA, 1);
    drive(GPIO_SCL, 0);
    drive(GPIO_SDA, 0);
}

/* Returns 1 for ACK, 0 for NACK, -1 for a line error. */
static int write_byte(uint8_t value)
{
    for (int bit = 7; bit >= 0; --bit) {
        int one = !!(value & (1U << bit));
        drive(GPIO_SDA, !one);
        if (clock_high()) return -1;
        if (one && !level(45)) {
            fprintf(stderr, "SDA low while master sent 1 (bit %d)\n", bit);
            return -1;
        }
        drive(GPIO_SCL, 1);
    }
    drive(GPIO_SDA, 0);
    if (clock_high()) return -1;
    int ack = !level(45);
    drive(GPIO_SCL, 1);
    return ack;
}

static int read_byte(void)
{
    unsigned value = 0;
    drive(GPIO_SDA, 0);
    for (int bit = 7; bit >= 0; --bit) {
        if (clock_high()) return -1;
        value |= (unsigned)level(45) << bit;
        drive(GPIO_SCL, 1);
    }
    /* Master NACKs the only byte, then sends STOP. */
    if (clock_high()) return -1;
    drive(GPIO_SCL, 1);
    return (int)value;
}

static void probe(unsigned addr)
{
    if (start_condition()) return;
    int ack = write_byte((uint8_t)(addr << 1));
    printf("0x%02X write-address %s\n", addr,
           ack < 0 ? "LINE ERROR" : ack ? "ACK" : "NACK");
    if (ack != 1) { stop_condition(); return; }
    ack = write_byte(0xD0);
    printf("0x%02X register 0xD0 %s\n", addr,
           ack < 0 ? "LINE ERROR" : ack ? "ACK" : "NACK");
    if (ack != 1 || repeated_start()) { stop_condition(); return; }
    ack = write_byte((uint8_t)((addr << 1) | 1));
    printf("0x%02X read-address %s\n", addr,
           ack < 0 ? "LINE ERROR" : ack ? "ACK" : "NACK");
    if (ack != 1) { stop_condition(); return; }
    int id = read_byte();
    if (id >= 0) printf("0x%02X chip ID: 0x%02X%s\n", addr, id,
                        id == 0x58 ? " (BMP280)" : " (unexpected)");
    stop_condition();
}

int main(void)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return 1; }
    void *im = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IOMUX_BASE);
    void *gm = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, GPIO1_BASE);
    close(fd);
    if (im == MAP_FAILED || gm == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    iomux = im;
    gpio = gm;
    saved_mux44 = iomux[44];
    saved_mux45 = iomux[45];
    saved_dr = gpio[0];
    saved_ddr = gpio[1];
    if (((saved_mux44 >> 11) & 7) != 0 || ((saved_mux45 >> 11) & 7) != 0 ||
        (saved_ddr & (GPIO_SCL | GPIO_SDA))) {
        fprintf(stderr, "GPIO44/45 not in expected idle input state; aborting\n");
        return 2;
    }
    mapped = 1;
    atexit(restore);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGALRM, on_signal);
    alarm(8);
    /* Output data is low. Direction=input releases each open-drain line. */
    gpio[0] = saved_dr & ~(GPIO_SCL | GPIO_SDA);
    gpio[1] = saved_ddr & ~(GPIO_SCL | GPIO_SDA);
    iomux[44] = (saved_mux44 & ~INPUT_BIT) | 0x100;
    iomux[45] = (saved_mux45 & ~INPUT_BIT) | 0x100;
    printf("idle SCL=%d SDA=%d\n", level(44), level(45));
    probe(0x76);
    probe(0x77);
    return 0;
}

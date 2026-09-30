/* One-shot SHT31 read on K230 GPIO44=SCL, GPIO45=SDA.
 * Linux small core /dev/mem tool. Restores GPIO and IOMUX registers on exit.
 * Run only while the RT-Smart I2C3 pins are muxed away from GPIO44/45.
 */
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
#define SCL (1U << 12)
#define SDA (1U << 13)
#define DI (1U << 31)

static volatile uint32_t *mux;
static volatile uint32_t *gpio;
static uint32_t old44, old45, old_dr, old_ddr;
static int ready;
static long bit_delay_us = 150;
static int quiet;
static double last_temp_c, last_humidity;

static void delay_us(long us)
{
    if (us < 1000) {
        struct timespec begin, now;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        do {
            clock_gettime(CLOCK_MONOTONIC, &now);
        } while ((now.tv_sec - begin.tv_sec) * 1000000L +
                 (now.tv_nsec - begin.tv_nsec) / 1000L < us);
        return;
    }
    struct timespec t = { us / 1000000, (us % 1000000) * 1000 };
    nanosleep(&t, NULL);
}

static void restore(void)
{
    if (!ready) return;
    gpio[1] = (gpio[1] & ~(SCL | SDA)) | (old_ddr & (SCL | SDA));
    gpio[0] = (gpio[0] & ~(SCL | SDA)) | (old_dr & (SCL | SDA));
    mux[44] = old44 & ~DI;
    mux[45] = old45 & ~DI;
}

static void on_signal(int sig)
{
    restore();
    _exit(128 + sig);
}

static int level(int pin) { return !!(mux[pin] & DI); }

/* Output data is always zero; DDR=0 releases the open-drain signal. */
static void pull(uint32_t pin, int low)
{
    uint32_t ddr = gpio[1];
    gpio[1] = low ? (ddr | pin) : (ddr & ~pin);
    (void)gpio[1];
    delay_us(bit_delay_us);
}

static int scl_high(void)
{
    pull(SCL, 0);
    if (!level(44)) {
        fprintf(stderr, "SCL stuck low; PAD44=0x%08X PAD45=0x%08X\n",
                mux[44], mux[45]);
        return -1;
    }
    return 0;
}

static int start(void)
{
    pull(SDA, 0);
    if (scl_high() || !level(45)) {
        fprintf(stderr, "bus not idle high\n");
        return -1;
    }
    pull(SDA, 1);
    pull(SCL, 1);
    return 0;
}

static void stop(void)
{
    pull(SDA, 1);
    (void)scl_high();
    pull(SDA, 0);
}

/* 1=ACK, 0=NACK, -1=line error. */
static int write_byte(uint8_t b)
{
    for (int bit = 7; bit >= 0; --bit) {
        int one = !!(b & (1U << bit));
        pull(SDA, !one);
        if (scl_high()) return -1;
        if (one && !level(45)) {
            fprintf(stderr, "SDA low while sending 1\n");
            return -1;
        }
        pull(SCL, 1);
    }
    pull(SDA, 0);
    if (scl_high()) return -1;
    int ack = !level(45);
    pull(SCL, 1);
    return ack;
}

static int read_byte(int send_ack)
{
    int b = 0;
    pull(SDA, 0);
    for (int bit = 7; bit >= 0; --bit) {
        if (scl_high()) return -1;
        b |= level(45) << bit;
        pull(SCL, 1);
    }
    pull(SDA, send_ack);
    if (scl_high()) return -1;
    pull(SCL, 1);
    pull(SDA, 0);
    return b;
}

static uint8_t crc8(const uint8_t *p)
{
    uint8_t crc = 0xff;
    for (int j = 0; j < 2; ++j) {
        crc ^= p[j];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x31) : (uint8_t)(crc << 1);
    }
    return crc;
}

static int probe(uint8_t addr)
{
    uint8_t bytes[6];
    if (start()) return -1;
    int ack = write_byte((uint8_t)(addr << 1));
    if (!quiet) printf("0x%02X write address: %s\n", addr,
                       ack < 0 ? "line error" : ack ? "ACK" : "NACK");
    if (ack != 1) { stop(); return ack < 0 ? -1 : 1; }
    if (write_byte(0x24) != 1 || write_byte(0x00) != 1) {
        fprintf(stderr, "measurement command NACK/line error\n");
        stop();
        return -1;
    }
    stop();
    delay_us(30000); /* high repeatability, no clock stretching */

    if (start()) return -1;
    ack = write_byte((uint8_t)((addr << 1) | 1));
    if (!quiet) printf("0x%02X read address: %s\n", addr,
                       ack < 0 ? "line error" : ack ? "ACK" : "NACK");
    if (ack != 1) { stop(); return -1; }
    for (int i = 0; i < 6; ++i) {
        int b = read_byte(i < 5);
        if (b < 0) { stop(); return -1; }
        bytes[i] = (uint8_t)b;
    }
    stop();
    if (!quiet) printf("raw: %02X %02X %02X %02X %02X %02X\n",
                       bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]);
    if (crc8(bytes) != bytes[2] || crc8(bytes + 3) != bytes[5]) {
        fprintf(stderr, "CRC failed: expected %02X and %02X\n",
                crc8(bytes), crc8(bytes + 3));
        return -1;
    }
    unsigned raw_t = ((unsigned)bytes[0] << 8) | bytes[1];
    unsigned raw_h = ((unsigned)bytes[3] << 8) | bytes[4];
    last_temp_c = -45.0 + 175.0 * raw_t / 65535.0;
    last_humidity = 100.0 * raw_h / 65535.0;
    if (!quiet) printf("SHT31 temperature=%.2f C humidity=%.2f %%RH (CRC OK)\n",
                       last_temp_c, last_humidity);
    return 0;
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
    signal(SIGALRM, on_signal);
    alarm(10);
    gpio[0] = old_dr & ~(SCL | SDA);
    gpio[1] = old_ddr & ~(SCL | SDA);
    mux[44] = old44 & ~DI;
    mux[45] = old45 & ~DI;
    printf("idle SCL=%d SDA=%d\n", level(44), level(45));
    int ret = probe(0x44);
    if (ret == 1) ret = probe(0x45);
    return ret == 0 ? 0 : 1;
}

/* Linux small-core bridge: I2C sensor log -> one atomic /sharefs status record.
 * No network transport and no second process touches the sensor I2C bus.
 */
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <errno.h>
#include <net/if.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t running = 1;
static void stop_bridge(int signal_number) { (void)signal_number; running = 0; }

static double monotonic_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static void sleep_ms(int ms)
{
    struct timespec t = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&t, NULL);
}

typedef struct {
    off_t offset;
    ino_t inode;
    double sht_time;
    double bmp_time;
    double temp, humidity, pressure;
    int sht_ok, bmp_ok;
} SensorLog;

static void consume_new_log(const char *path, SensorLog *s)
{
    struct stat st;
    if (stat(path, &st) != 0) {
        s->offset = 0;
        s->inode = 0;
        s->sht_ok = s->bmp_ok = 0;
        return;
    }
    if (st.st_ino != s->inode || st.st_size < s->offset) {
        s->inode = st.st_ino;
        s->offset = 0;
        s->sht_ok = s->bmp_ok = 0;
    }
    if (st.st_size == s->offset) return;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (s->offset == 0 && st.st_size > 8192) {
        fseeko(f, st.st_size - 8192, SEEK_SET);
        char discard[256];
        (void)fgets(discard, sizeof(discard), f);
    } else {
        fseeko(f, s->offset, SEEK_SET);
    }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        double a, b;
        if (sscanf(line, "SHT31: %lf C, %lf %%RH (CRC OK)", &a, &b) == 2) {
            if (a >= -40 && a <= 125 && b >= 0 && b <= 100) {
                s->temp = a; s->humidity = b; s->sht_ok = 1;
                s->sht_time = monotonic_seconds();
            }
        } else if (strstr(line, "SHT31 read/CRC failed")) {
            s->sht_ok = 0;
        } else if (sscanf(line, "BMP280: %lf C, %lf hPa", &a, &b) == 2) {
            if (b >= 300 && b <= 1200) {
                s->pressure = b; s->bmp_ok = 1;
                s->bmp_time = monotonic_seconds();
            }
        } else if (strstr(line, "BMP280 read failed")) {
            s->bmp_ok = 0;
        }
    }
    off_t end = ftello(f);
    if (end >= 0) s->offset = end;
    fclose(f);
}

static int wlan0_has_ipv4(void)
{
    FILE *f = fopen("/sys/class/net/wlan0/operstate", "r");
    if (!f) return 0;
    char state[16] = {0};
    int up = fgets(state, sizeof(state), f) && strncmp(state, "up", 2) == 0;
    fclose(f);
    if (!up) return 0;
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return 0;
    struct ifreq req;
    memset(&req, 0, sizeof(req));
    strncpy(req.ifr_name, "wlan0", IFNAMSIZ - 1);
    int has_address = ioctl(fd, SIOCGIFADDR, &req) == 0;
    close(fd);
    return has_address;
}

static int write_status(const char *path, unsigned long sequence,
                        const SensorLog *s, int wifi)
{
    char next[512];
    if (snprintf(next, sizeof(next), "%s.next", path) >= (int)sizeof(next)) return -1;
    FILE *f = fopen(next, "w");
    if (!f) return -1;
    double now = monotonic_seconds();
    int sht = s->sht_ok && now - s->sht_time <= 5.0;
    int bmp = s->bmp_ok && now - s->bmp_time <= 5.0;
    int result = fprintf(f, "v1 %lu %d %.1f %.1f %d %.1f %d\n", sequence,
                         sht, s->temp, s->humidity, bmp, s->pressure, wifi);
    if (fclose(f) != 0 || result <= 0) return -1;
    return rename(next, path);
}

/* Last 20 minutes at 2-second cadence, kept only in the mounted tmpfs. */
#define HISTORY_CAP 600
typedef struct { unsigned long seq; int valid; double temp, humidity; } HistoryPoint;
static HistoryPoint history[HISTORY_CAP];
static size_t history_count;
static size_t history_next;

static int write_history(const char *status_path, unsigned long sequence,
                         const SensorLog *s)
{
    double now = monotonic_seconds();
    history[history_next] = (HistoryPoint){sequence,
        s->sht_ok && now - s->sht_time <= 5.0, s->temp, s->humidity};
    history_next = (history_next + 1) % HISTORY_CAP;
    if (history_count < HISTORY_CAP) ++history_count;
    const char *slash = strrchr(status_path, '/');
    if (!slash) return -1;
    char path[512], next[520];
    int prefix = (int)(slash - status_path + 1);
    if (snprintf(path, sizeof(path), "%.*shistory.txt", prefix, status_path) >= (int)sizeof(path) ||
        snprintf(next, sizeof(next), "%s.next", path) >= (int)sizeof(next)) return -1;
    FILE *f = fopen(next, "w");
    if (!f) return -1;
    int bad = fprintf(f, "v1 %zu\n", history_count) <= 0;
    for (size_t i = 0; i < history_count && !bad; ++i) {
        const HistoryPoint *p = &history[(history_next + HISTORY_CAP - history_count + i) % HISTORY_CAP];
        bad = fprintf(f, "%lu %d %.1f %.1f\n", p->seq, p->valid, p->temp, p->humidity) <= 0;
    }
    if (fclose(f) != 0 || bad) return -1;
    return rename(next, path);
}

int main(int argc, char **argv)
{
    const char *log_path = argc > 1 ? argv[1] : "/tmp/sensor_mvp.log";
    const char *status_path = argc > 2 ? argv[2] : "/sharefs/srtp_clean/ui_overlay/status_ram/status.txt";
    int max_cycles = argc > 3 ? atoi(argv[3]) : 0;
    signal(SIGINT, stop_bridge);
    signal(SIGTERM, stop_bridge);
    SensorLog sensor = {0};
    unsigned long sequence = 0;
    int cycles = 0;
    while (running && (max_cycles <= 0 || cycles < max_cycles)) {
        consume_new_log(log_path, &sensor);
        if (write_status(status_path, ++sequence, &sensor, wlan0_has_ipv4()) != 0)
            fprintf(stderr, "status write failed: %s (%d)\n", status_path, errno);
        if (write_history(status_path, sequence, &sensor) != 0)
            fprintf(stderr, "history write failed: %s (%d)\n", status_path, errno);
        ++cycles;
        if (running && (max_cycles <= 0 || cycles < max_cycles)) sleep_ms(2000);
    }
    return 0;
}

/* Standalone 536x960 data display. The original camera/box process is stopped
 * before this program starts; this process owns VB, connector and OSD0.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <time.h>
#include <unistd.h>

#include "k_module.h"
#include "k_vo_comm.h"
#include "k_video_comm.h"
#include "k_connector_comm.h"
#include "mpi_connector_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vo_api.h"
#include "board_connector_profile.h"

#define W 536
#define H 960
#define BYTES ((size_t)W * H * 4)
#define HISTORY_CAP 600
#define GLYPHS "0123456789.-"

typedef struct { unsigned long seq; int sht, bmp, wifi; double temp, humidity, pressure; } Status;
typedef struct { unsigned long seq; int valid; double temp, humidity; } Point;
static const int large_widths[12] = {16,16,16,16,16,16,16,16,16,16,8,11};
static const int small_widths[12] = {7,7,7,7,7,7,7,7,7,7,4,5};

static unsigned char *load_exact(const char *dir, const char *name, size_t bytes)
{
    char path[320];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "data asset missing: %s\n", path); return NULL; }
    unsigned char *p = malloc(bytes);
    if (!p || fread(p, 1, bytes, f) != bytes || fgetc(f) != EOF) {
        fprintf(stderr, "data asset invalid: %s\n", path);
        free(p); p = NULL;
    }
    fclose(f);
    return p;
}

static int read_status(Status *out)
{
    FILE *f = fopen("/sharefs/srtp_clean/ui_overlay/status_ram/status.txt", "r");
    if (!f) return -1;
    char line[160];
    int got = fgets(line, sizeof(line), f) != NULL;
    fclose(f);
    Status s = {0};
    if (!got || sscanf(line, "v1 %lu %d %lf %lf %d %lf %d", &s.seq, &s.sht,
                       &s.temp, &s.humidity, &s.bmp, &s.pressure, &s.wifi) != 7 ||
        !isfinite(s.temp) || !isfinite(s.humidity) || !isfinite(s.pressure) ||
        s.sht < 0 || s.sht > 1 || s.bmp < 0 || s.bmp > 1 ||
        s.wifi < 0 || s.wifi > 1) return -1;
    *out = s;
    return 0;
}

static int read_history(Point points[HISTORY_CAP], int *count)
{
    FILE *f = fopen("/sharefs/srtp_clean/ui_overlay/status_ram/history.txt", "r");
    if (!f) return -1;
    int n;
    char version[8];
    if (fscanf(f, "%7s %d", version, &n) != 2 || strcmp(version, "v1") ||
        n < 0 || n > HISTORY_CAP) { fclose(f); return -1; }
    for (int i = 0; i < n; ++i) {
        Point p;
        if (fscanf(f, "%lu %d %lf %lf", &p.seq, &p.valid,
                   &p.temp, &p.humidity) != 4 ||
            (p.valid != 0 && p.valid != 1) ||
            !isfinite(p.temp) || !isfinite(p.humidity)) {
            fclose(f); return -1;
        }
        points[i] = p;
    }
    fclose(f);
    *count = n;
    return 0;
}

static void blit(unsigned char *dst, int x, int y, const unsigned char *src,
                 int src_w, int src_x, int copy_w, int copy_h)
{
    if (x < 0 || y < 0 || x + copy_w > W || y + copy_h > H) return;
    for (int row = 0; row < copy_h; ++row) {
        const unsigned char *from = src + (((size_t)row * src_w + src_x) * 4);
        unsigned char *to = dst + (((size_t)(y + row) * W + x) * 4);
        for (int col = 0; col < copy_w; ++col) {
            if (from[col * 4 + 3]) memcpy(to + col * 4, from + col * 4, 4);
        }
    }
}

static int draw_num(unsigned char *dst, int x, int y, const char *value,
                    const unsigned char *atlas, int cell_w, int cell_h,
                    const int widths[12])
{
    for (const char *p = value; *p; ++p) {
        const char *found = strchr(GLYPHS, *p);
        if (!found) continue;
        int index = (int)(found - GLYPHS);
        blit(dst, x, y, atlas, cell_w * 12, index * cell_w, widths[index], cell_h);
        x += widths[index];
    }
    return x;
}

static void pixel(unsigned char *dst, int x, int y, unsigned char b,
                  unsigned char g, unsigned char r)
{
    if (x < 46 || x > 499 || y < 359 || y > 736) return;
    size_t at = ((size_t)y * W + x) * 4;
    dst[at] = b; dst[at + 1] = g; dst[at + 2] = r; dst[at + 3] = 255;
}

static void line(unsigned char *dst, int x0, int y0, int x1, int y1,
                 unsigned char b, unsigned char g, unsigned char r)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        pixel(dst, x0, y0, b, g, r);
        pixel(dst, x0, y0 + 1, b, g, r);
        if (x0 == x1 && y0 == y1) break;
        int e = 2 * err;
        if (e >= dy) { err += dy; x0 += sx; }
        if (e <= dx) { err += dx; y0 += sy; }
    }
}

static void plot(unsigned char *dst, const Point *points, int count, int humidity,
                 const unsigned char *small)
{
    if (count < 1) return;
    double lo = 1e9, hi = -1e9;
    for (int i = 0; i < count; ++i) if (points[i].valid) {
        double v = humidity ? points[i].humidity : points[i].temp;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    if (hi < lo) return;
    double span = hi - lo;
    double min_span = humidity ? 10.0 : 5.0;
    if (span < min_span) { double mid = (hi + lo) / 2; lo = mid - min_span / 2; hi = mid + min_span / 2; }
    if (humidity) { if (lo < 0) lo = 0; if (hi > 100) hi = 100; }
    int top = humidity ? 592 : 359;
    int bottom = humidity ? 736 : 503;
    char label[16];
    snprintf(label, sizeof(label), "%.1f", hi);
    draw_num(dst, 51, top + 3, label, small, 14, 20, small_widths);
    snprintf(label, sizeof(label), "%.1f", lo);
    draw_num(dst, 51, bottom - 20, label, small, 14, 20, small_widths);
    unsigned long newest = points[count - 1].seq;
    int prev = -1, px = 0, py = 0;
    for (int i = 0; i < count; ++i) {
        if (!points[i].valid || newest < points[i].seq || newest - points[i].seq >= HISTORY_CAP) {
            prev = -1; continue;
        }
        double v = humidity ? points[i].humidity : points[i].temp;
        int x = 499 - (int)((newest - points[i].seq) * 453 / (HISTORY_CAP - 1));
        int y = bottom - (int)((v - lo) * (bottom - top) / (hi - lo));
        if (y < top) y = top;
        if (y > bottom) y = bottom;
        if (prev >= 0) {
            if (humidity) line(dst, px, py, x, y, 112, 107, 66);
            else line(dst, px, py, x, y, 58, 105, 157);
        }
        prev = i; px = x; py = y;
    }
}

static int quit_requested(int ms)
{
    fd_set fds;
    FD_ZERO(&fds); FD_SET(STDIN_FILENO, &fds);
    struct timeval timeout = {ms / 1000, (ms % 1000) * 1000};
    if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &timeout) > 0) {
        char c;
        if (read(STDIN_FILENO, &c, 1) == 1 && c == 'q') return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s data_asset_dir live_asset_dir [max_seconds; default30, 0=until q]\n", argv[0]);
        return 2;
    }
    long max_seconds = 30;
    if (argc == 4) {
        char *end = NULL;
        errno = 0;
        max_seconds = strtol(argv[3], &end, 10);
        if (errno || !*argv[3] || *end || max_seconds < 0 || max_seconds > 3600) return 2;
    }
    struct timespec started;
    if (clock_gettime(CLOCK_MONOTONIC, &started) != 0) return 2;
    unsigned char *base = load_exact(argv[1], "base.bgra", BYTES);
    unsigned char *wifi[3] = {
        load_exact(argv[1], "wifi_off.bgra", 110 * 29 * 4),
        load_exact(argv[1], "wifi_on.bgra", 110 * 29 * 4),
        load_exact(argv[1], "wifi_unknown.bgra", 110 * 29 * 4),
    };
    unsigned char *large = load_exact(argv[2], "glyphs_large.bgra", 24 * 12 * 40 * 4);
    unsigned char *small = load_exact(argv[2], "glyphs_small.bgra", 14 * 12 * 20 * 4);
    unsigned char *unit_temp = load_exact(argv[2], "unit_temp.bgra", 34 * 25 * 4);
    unsigned char *unit_humidity = load_exact(argv[2], "unit_humidity.bgra", 18 * 25 * 4);
    unsigned char *unit_pressure = load_exact(argv[2], "unit_pressure.bgra", 28 * 20 * 4);
    int rc = 3, vb_ready = 0;
    if (!base || !wifi[0] || !wifi[1] || !wifi[2] || !large || !small ||
        !unit_temp || !unit_humidity || !unit_pressure) goto done;

    k_vb_config config = {0};
    config.max_pool_cnt = 8;
    if (kd_mpi_vb_set_config(&config) != 0 || kd_mpi_vb_init() != 0) {
        fprintf(stderr, "data VB init failed\n"); goto done;
    }
    vb_ready = 1;
    k_connector_info connector;
    make_board_connector(&connector);
    printf("DATA: verified-ELF profile display540x960 canvas536x960 pclk39600 type-offset104\n");
    int connector_fd = kd_mpi_connector_open(connector.connector_name);
    if (connector_fd < 0 || kd_mpi_connector_power_set(connector_fd, K_TRUE) != 0 ||
        kd_mpi_connector_init(connector_fd, connector) != 0) {
        fprintf(stderr, "data connector init failed\n"); goto done;
    }

    k_vb_pool_config pool_config = {0};
    pool_config.blk_size = (BYTES + 1023) & ~(size_t)1023;
    pool_config.blk_cnt = 1;
    pool_config.mode = VB_REMAP_MODE_NOCACHE;
    k_u32 pool = kd_mpi_vb_create_pool(&pool_config);
    if (pool == VB_INVALID_POOLID) { fprintf(stderr, "data pool failed\n"); goto done; }
    k_vb_blk_handle block = kd_mpi_vb_get_block(pool, BYTES, NULL);
    if (block == VB_INVALID_HANDLE) { fprintf(stderr, "data block failed\n"); kd_mpi_vb_destory_pool(pool); goto done; }
    k_u64 phys = kd_mpi_vb_handle_to_phyaddr(block);
    unsigned char *virt = kd_mpi_sys_mmap(phys, BYTES);
    if (!phys || !virt) { fprintf(stderr, "data mmap failed\n"); kd_mpi_vb_release_block(block); kd_mpi_vb_destory_pool(pool); goto done; }

    k_vo_video_osd_attr attr = {0};
    attr.display_rect.x = 0; attr.display_rect.y = 0;
    attr.img_size.width = W; attr.img_size.height = H;
    attr.pixel_format = PIXEL_FORMAT_ARGB_8888;
    attr.stride = W * 4 / 8; attr.global_alptha = 255;
    kd_mpi_vo_osd_disable(K_VO_OSD0);
    if (kd_mpi_vo_set_video_osd_attr(K_VO_OSD0, &attr) != 0 ||
        kd_mpi_vo_osd_enable(K_VO_OSD0) != 0) {
        fprintf(stderr, "data OSD setup failed\n"); rc = 5; goto cleanup;
    }
    if (connector_fd >= 0) kd_mpi_connector_close(connector_fd);
    int inserted = 0;
    unsigned long last_seq = 0;
    int stale = 3, ticks = 0;
    Status status = {0};
    Point points[HISTORY_CAP];
    int count = 0;
    rc = 0;
    for (;;) {
        if (ticks % 50 == 0) {
            Status next;
            if (read_status(&next) == 0 && next.seq != last_seq) {
                status = next; last_seq = next.seq; stale = 0;
            } else ++stale;
            if (read_history(points, &count) != 0) count = 0;
            memcpy(virt, base, BYTES);
            int fresh = stale < 3;
            char value[20];
            snprintf(value, sizeof(value), fresh && status.sht ? "%.1f" : "--", status.temp);
            int end = draw_num(virt, 27, 171, value, large, 24, 40, large_widths);
            if (fresh && status.sht) blit(virt, end + 3, 180, unit_temp, 34, 0, 34, 25);
            snprintf(value, sizeof(value), fresh && status.sht ? "%.1f" : "--", status.humidity);
            end = draw_num(virt, 278, 171, value, large, 24, 40, large_widths);
            if (fresh && status.sht) blit(virt, end + 3, 180, unit_humidity, 18, 0, 18, 25);
            snprintf(value, sizeof(value), fresh && status.bmp ? "%.1f" : "--", status.pressure);
            end = draw_num(virt, 27, 253, value, small, 14, 20, small_widths);
            if (fresh && status.bmp) blit(virt, end + 3, 253, unit_pressure, 28, 0, 28, 20);
            blit(virt, 278, 252, wifi[fresh ? status.wifi : 2], 110, 0, 110, 29);
            plot(virt, points, count, 0, small);
            plot(virt, points, count, 1, small);
            if (!inserted) {
                k_video_frame_info frame = {0};
                frame.mod_id = K_ID_VO; frame.pool_id = pool;
                frame.v_frame.width = W; frame.v_frame.height = H;
                frame.v_frame.stride[0] = W; frame.v_frame.phys_addr[0] = phys;
                frame.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
                frame.v_frame.priv_data = K_VO_ONLY_CHANGE_PHYADDR;
                if (kd_mpi_vo_chn_insert_frame(K_VO_OSD0 + 3, &frame) != 0) {
                    fprintf(stderr, "data frame insert failed\n"); rc = 6; break;
                }
                inserted = 1;
            }
            printf("DATA: seq=%lu temp=%.1f humidity=%.1f history=%d fresh=%d\n",
                   last_seq, status.temp, status.humidity, count, fresh);
        }
        if (quit_requested(40)) break;
        ++ticks;
        struct timespec now;
        if (max_seconds > 0 && clock_gettime(CLOCK_MONOTONIC, &now) == 0 &&
            (now.tv_sec - started.tv_sec > max_seconds ||
             (now.tv_sec - started.tv_sec == max_seconds && now.tv_nsec >= started.tv_nsec))) {
            puts("DATA: bounded candidate timeout");
            break;
        }
    }
cleanup:
    kd_mpi_vo_osd_disable(K_VO_OSD0);
    /* With no camera layer, leaving VO scanning a freed OSD buffer produces
     * garbage after exit. Shut down the standalone display before VB cleanup. */
    kd_mpi_vo_disable();
    struct timespec drain = {0, 250000000L};
    nanosleep(&drain, NULL);
    kd_mpi_sys_munmap(virt, BYTES);
    kd_mpi_vb_release_block(block);
    kd_mpi_vb_destory_pool(pool);
done:
    if (vb_ready && kd_mpi_vb_exit() != 0) { fprintf(stderr, "data VB exit failed\n"); rc = 7; }
    free(base); for (int i = 0; i < 3; ++i) free(wifi[i]);
    free(large); free(small); free(unit_temp); free(unit_humidity); free(unit_pressure);
    return rc;
}

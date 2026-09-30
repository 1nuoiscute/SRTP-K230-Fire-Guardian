/* RT-Smart large-core live UI. Reads status from /sharefs, never touches I2C/Wi-Fi.
 * Reuses the verified OSD0 plane while the original visual ELF owns VO/camera.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <time.h>
#include <unistd.h>

#include "k_module.h"
#include "k_vo_comm.h"
#include "k_video_comm.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vo_api.h"

#define W 536
#define H 348
#define PANEL_Y 612
#define PANEL_BYTES ((size_t)W * H * 4)
#define GLYPHS "0123456789.-"
#define GPIO_INPUT _IOW('G', 1, int)
#define GPIO_OUTPUT _IOW('G', 0, int)
#define GPIO_LOW _IOW('G', 4, int)
#define GPIO_READ _IOW('G', 12, int)

typedef struct { int sht, bmp, wifi; double temp, humidity, pressure; unsigned long seq; } Status;
typedef struct { char temp[16], humidity[16], pressure[16]; int wifi; } Display;
typedef struct { k_vb_blk_handle handle; k_u64 phys; unsigned char *virt; } Buffer;
typedef struct { uint16_t pin, value; } GpioRequest;
typedef struct { int fd, stable, candidate, count; } MatrixKey;

static int gpio_action(int fd, unsigned long cmd, uint16_t pin, uint16_t *value)
{
    GpioRequest arg = {pin, 0};
    if (ioctl(fd, cmd, &arg) != 0) return -1;
    if (value) *value = arg.value;
    return 0;
}

static void matrix_key_close(MatrixKey *key)
{
    if (key->fd >= 0) {
        (void)gpio_action(key->fd, GPIO_INPUT, 28, NULL);
        close(key->fd);
        key->fd = -1;
    }
}

static int matrix_key_open(MatrixKey *key)
{
    memset(key, 0, sizeof(*key));
    key->fd = open("/dev/gpio", O_RDWR);
    if (key->fd < 0) return -1;
    if (gpio_action(key->fd, GPIO_INPUT, 18, NULL) ||
        gpio_action(key->fd, GPIO_OUTPUT, 28, NULL) ||
        gpio_action(key->fd, GPIO_LOW, 28, NULL)) {
        matrix_key_close(key);
        return -1;
    }
    return 0;
}

static int matrix_key_poll_press(MatrixKey *key)
{
    if (key->fd < 0) return 0;
    uint16_t level = 1;
    if (gpio_action(key->fd, GPIO_READ, 18, &level) != 0) {
        fprintf(stderr, "UI matrix KEY1 read failed\n");
        matrix_key_close(key);
        return 0;
    }
    int pressed = level == 0;
    if (pressed != key->candidate) {
        key->candidate = pressed;
        key->count = 1;
        return 0;
    }
    if (key->count < 3) ++key->count;
    if (key->count == 3 && pressed != key->stable) {
        key->stable = pressed;
        return pressed;
    }
    return 0;
}

static const int large_widths[12] = {16,16,16,16,16,16,16,16,16,16,8,11};
static const int small_widths[12] = {7,7,7,7,7,7,7,7,7,7,4,5};

static void *load_exact(const char *dir, const char *name, size_t bytes)
{
    char path[256];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "UI asset missing: %s\n", path); return NULL; }
    void *data = malloc(bytes);
    if (!data || fread(data, 1, bytes, f) != bytes || fgetc(f) != EOF) {
        fprintf(stderr, "UI asset size invalid: %s\n", path);
        free(data); data = NULL;
    }
    fclose(f);
    return data;
}

static int read_status(const char *path, Status *out)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[160];
    int got = fgets(line, sizeof(line), f) != NULL;
    fclose(f);
    if (!got) return -1;
    Status s = {0};
    if (sscanf(line, "v1 %lu %d %lf %lf %d %lf %d", &s.seq, &s.sht,
               &s.temp, &s.humidity, &s.bmp, &s.pressure, &s.wifi) != 7)
        return -1;
    if ((s.sht != 0 && s.sht != 1) || (s.bmp != 0 && s.bmp != 1) ||
        (s.wifi != 0 && s.wifi != 1) || !isfinite(s.temp) ||
        !isfinite(s.humidity) || !isfinite(s.pressure)) return -1;
    *out = s;
    return 0;
}

static void make_display(const Status *s, int fresh, Display *d)
{
    memset(d, 0, sizeof(*d));
    d->wifi = fresh ? s->wifi : -1;
    if (fresh && s->sht) {
        snprintf(d->temp, sizeof(d->temp), "%.1f", s->temp);
        snprintf(d->humidity, sizeof(d->humidity), "%.1f", s->humidity);
    } else {
        strcpy(d->temp, "--"); strcpy(d->humidity, "--");
    }
    if (fresh && s->bmp) snprintf(d->pressure, sizeof(d->pressure), "%.1f", s->pressure);
    else strcpy(d->pressure, "--");
}

static void blit(unsigned char *dst, int x, int y, const unsigned char *src,
                 int src_w, int src_x, int copy_w, int copy_h)
{
    if (x < 0 || y < 0 || x + copy_w > W || y + copy_h > H) return;
    for (int row = 0; row < copy_h; ++row)
        memcpy(dst + (((size_t)(y + row) * W + x) * 4),
               src + (((size_t)row * src_w + src_x) * 4), (size_t)copy_w * 4);
}

static int draw_number(unsigned char *dst, int x, int y, const char *value,
                       const unsigned char *atlas, int cell_w, int cell_h,
                       const int *widths)
{
    for (const char *p = value; *p; ++p) {
        const char *found = strchr(GLYPHS, *p);
        if (!found) continue;
        int index = (int)(found - GLYPHS);
        int width = widths[index];
        blit(dst, x, y, atlas, cell_w * 12, index * cell_w, width, cell_h);
        x += width;
    }
    return x;
}

static void paint(unsigned char *dst, const Display *d,
                  const unsigned char *bases[3], const unsigned char *large,
                  const unsigned char *small, const unsigned char *unit_temp,
                  const unsigned char *unit_humidity, const unsigned char *unit_pressure)
{
    int base_index = d->wifi < 0 ? 2 : d->wifi ? 1 : 0;
    memcpy(dst, bases[base_index], PANEL_BYTES);
    int end = draw_number(dst, 26, 201, d->temp, large, 24, 40, large_widths);
    if (strcmp(d->temp, "--") != 0) blit(dst, end + 3, 209, unit_temp, 34, 0, 34, 25);
    end = draw_number(dst, 280, 201, d->humidity, large, 24, 40, large_widths);
    if (strcmp(d->humidity, "--") != 0)
        blit(dst, end + 3, 209, unit_humidity, 18, 0, 18, 25);
    end = draw_number(dst, 60, 324, d->pressure, small, 14, 20, small_widths);
    if (strcmp(d->pressure, "--") != 0)
        blit(dst, end + 3, 324, unit_pressure, 28, 0, 28, 20);
}

static int wait_or_quit(int ms)
{
    fd_set readers;
    FD_ZERO(&readers);
    FD_SET(STDIN_FILENO, &readers);
    struct timeval timeout = {ms / 1000, (ms % 1000) * 1000};
    int ready = select(STDIN_FILENO + 1, &readers, NULL, NULL, &timeout);
    if (ready > 0) {
        char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1 && ch == 'q') return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s asset_dir [max_seconds; 0=until q]\n", argv[0]);
        return 2;
    }
    int max_seconds = argc == 3 ? atoi(argv[2]) : 0;
    if (max_seconds < 0 || max_seconds > 3600) return 2;
    const char *dir = argv[1];
    const unsigned char *bases[3] = {
        load_exact(dir, "base_off.bgra", PANEL_BYTES),
        load_exact(dir, "base_on.bgra", PANEL_BYTES),
        load_exact(dir, "base_unknown.bgra", PANEL_BYTES),
    };
    const unsigned char *large = load_exact(dir, "glyphs_large.bgra", 24 * 12 * 40 * 4);
    const unsigned char *small = load_exact(dir, "glyphs_small.bgra", 14 * 12 * 20 * 4);
    const unsigned char *unit_temp = load_exact(dir, "unit_temp.bgra", 34 * 25 * 4);
    const unsigned char *unit_humidity = load_exact(dir, "unit_humidity.bgra", 18 * 25 * 4);
    const unsigned char *unit_pressure = load_exact(dir, "unit_pressure.bgra", 28 * 20 * 4);
    int rc = 3;
    if (!bases[0] || !bases[1] || !bases[2] || !large || !small ||
        !unit_temp || !unit_humidity || !unit_pressure) goto free_assets;

    k_vb_pool_config config;
    memset(&config, 0, sizeof(config));
    config.blk_size = (PANEL_BYTES + 1023) & ~(size_t)1023;
    config.blk_cnt = 1;
    config.mode = VB_REMAP_MODE_NOCACHE;
    k_u32 pool = kd_mpi_vb_create_pool(&config);
    if (pool == VB_INVALID_POOLID) { fprintf(stderr, "UI VB pool failed\n"); goto free_assets; }
    Buffer buffers[1] = {{0}};
    for (int i = 0; i < 1; ++i) {
        buffers[i].handle = kd_mpi_vb_get_block(pool, PANEL_BYTES, NULL);
        if (buffers[i].handle == VB_INVALID_HANDLE) { fprintf(stderr, "UI VB block failed\n"); goto cleanup_vb; }
        buffers[i].phys = kd_mpi_vb_handle_to_phyaddr(buffers[i].handle);
        buffers[i].virt = kd_mpi_sys_mmap(buffers[i].phys, PANEL_BYTES);
        if (!buffers[i].phys || !buffers[i].virt) { fprintf(stderr, "UI VB map failed\n"); goto cleanup_vb; }
    }

    k_vo_video_osd_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.display_rect.x = 0; attr.display_rect.y = PANEL_Y;
    attr.img_size.width = W; attr.img_size.height = H;
    attr.pixel_format = PIXEL_FORMAT_ARGB_8888;
    attr.stride = W * 4 / 8;
    attr.global_alptha = 255;
    kd_mpi_vo_osd_disable(K_VO_OSD0);
    if (kd_mpi_vo_set_video_osd_attr(K_VO_OSD0, &attr) != 0 ||
        kd_mpi_vo_osd_enable(K_VO_OSD0) != 0) {
        fprintf(stderr, "UI OSD0 setup failed\n"); goto cleanup_vb;
    }
    MatrixKey key;
    if (matrix_key_open(&key) != 0)
        fprintf(stderr, "UI matrix KEY1 unavailable; panel remains visible\n");

    const char *status_path = "/sharefs/srtp_clean/ui_overlay/status_ram/status.txt";
    unsigned long previous_seq = 0;
    int stale_cycles = 3, shown = -1, ticks = 0, panel_visible = 1;
    Display displayed = {{0}};
    Status status = {0};
    rc = 0;
    for (;;) {
        if (matrix_key_poll_press(&key)) {
            panel_visible = !panel_visible;
            if (panel_visible) {
                if (kd_mpi_vo_osd_enable(K_VO_OSD0) != 0) {
                    fprintf(stderr, "UI panel enable failed\n"); rc = 7; break;
                }
                shown = -1;
            } else if (kd_mpi_vo_osd_disable(K_VO_OSD0) != 0) {
                fprintf(stderr, "UI panel disable failed\n"); rc = 8; break;
            }
            printf("UI: matrix KEY1 panel %s\n", panel_visible ? "shown" : "hidden");
        }
        if (ticks % 50 == 0) {
            if (read_status(status_path, &status) == 0 && status.seq != previous_seq) {
                previous_seq = status.seq;
                stale_cycles = 0;
            } else {
                ++stale_cycles;
            }
        }
        Display desired;
        make_display(&status, stale_cycles < 3, &desired);
        if (panel_visible && (shown < 0 || memcmp(&desired, &displayed, sizeof(desired)) != 0)) {
            paint(buffers[0].virt, &desired, bases, large, small,
                  unit_temp, unit_humidity, unit_pressure);
            if (shown < 0) {
                k_video_frame_info frame;
                memset(&frame, 0, sizeof(frame));
                frame.mod_id = K_ID_VO; frame.pool_id = pool;
                frame.v_frame.width = W; frame.v_frame.height = H;
                frame.v_frame.stride[0] = W;
                frame.v_frame.phys_addr[0] = buffers[0].phys;
                frame.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
                frame.v_frame.priv_data = K_VO_ONLY_CHANGE_PHYADDR;
                if (kd_mpi_vo_chn_insert_frame(K_VO_OSD0 + 3, &frame) != 0) {
                    fprintf(stderr, "UI frame insert failed\n"); rc = 5; break;
                }
            }
            shown = 0;
            displayed = desired;
            printf("UI: temp=%s humidity=%s pressure=%s wifi=%d\n",
                   desired.temp, desired.humidity, desired.pressure, desired.wifi);
        }
        if (wait_or_quit(40)) break;
        ++ticks;
        if (max_seconds > 0 && ticks >= max_seconds * 25) break;
    }
    matrix_key_close(&key);
    if (panel_visible && kd_mpi_vo_osd_disable(K_VO_OSD0) != 0) {
        fprintf(stderr, "UI OSD0 disable failed\n"); rc = 6;
    }
    struct timespec drain = {0, 250000000L};
    nanosleep(&drain, NULL);
cleanup_vb:
    for (int i = 0; i < 1; ++i) {
        if (buffers[i].virt) kd_mpi_sys_munmap(buffers[i].virt, PANEL_BYTES);
        if (buffers[i].phys && buffers[i].handle != VB_INVALID_HANDLE)
            kd_mpi_vb_release_block(buffers[i].handle);
    }
    kd_mpi_vb_destory_pool(pool);
free_assets:
    for (int i = 0; i < 3; ++i) free((void *)bases[i]);
    free((void *)large); free((void *)small);
    free((void *)unit_temp); free((void *)unit_humidity); free((void *)unit_pressure);
    return rc;
}

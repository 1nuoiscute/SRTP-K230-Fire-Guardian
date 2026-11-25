/* Pure CPU data-page renderer; camera, VB and connector belong to the main process. */
#define _POSIX_C_SOURCE 200809L
#include "status_cache.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <time.h>
#include <unistd.h>



#define W 536
#define H 960
#define BYTES ((size_t)W * H * 4)
#define HISTORY_CAP 600
#define GLYPHS "0123456789.-"

/* Relative gas concentration shown on screen.
 *
 * MQ-2 is a semiconductor sensor: its output is a resistance change, and a real
 * ppm / %LEL figure needs a per-sensor R0 from a 24-48 h clean-air burn-in plus
 * a known gas concentration for the curve. Until that calibration exists, the
 * page shows a relative figure anchored on bench measurements:
 *   0 %   = clean-air baseline
 *   100 % = the alarm threshold
 * It is labelled "相对空载" so it is never mistaken for an absolute reading.
 * Raw millivolts stay available in the log and on the OLED. */
#define MQ2_BASE_MV  250
#define MQ2_ALARM_MV 500
typedef struct { unsigned long seq; int sht, bmp, wifi; double temp, humidity, pressure;
                 int mq2, flame; } Status;
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
    char line[160];
    if (demo_status_snapshot(line, sizeof(line)) < 0) return -1;
    Status s = {0};
    s.mq2 = -1;
    s.flame = -1;
    /* v2 carries the gas channels; fall back to v1 so an older bridge still
     * drives the page (gas simply shows as "--"). */
    if (sscanf(line, "v2 %lu %d %lf %lf %d %lf %d %d %d", &s.seq, &s.sht,
               &s.temp, &s.humidity, &s.bmp, &s.pressure, &s.wifi,
               &s.mq2, &s.flame) != 9) {
        s.mq2 = -1;
        s.flame = -1;
        if (sscanf(line, "v1 %lu %d %lf %lf %d %lf %d", &s.seq, &s.sht,
                   &s.temp, &s.humidity, &s.bmp, &s.pressure, &s.wifi) != 7)
            return -1;
    }
    if (!isfinite(s.temp) || !isfinite(s.humidity) || !isfinite(s.pressure) ||
        s.sht < 0 || s.sht > 1 || s.bmp < 0 || s.bmp > 1 ||
        s.wifi < 0 || s.wifi > 1) return -1;
    *out = s;
    return 0;
}

static int read_history(Point points[HISTORY_CAP], int *count)
{
    static char text[DEMO_HISTORY_BYTES];
    int bytes = demo_history_snapshot(text, sizeof(text));
    if (bytes < 0) return -1;
    FILE *f = fmemopen(text, (size_t)bytes, "r");
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
            unsigned a=from[col*4+3];
            if(a==255)memcpy(to+col*4,from+col*4,4);
            else if(a){for(int c=0;c<3;++c)to[col*4+c]=(from[col*4+c]*a+to[col*4+c]*(255-a)+127)/255;to[col*4+3]=255;}
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


static unsigned char *base, *wifi[3], *large, *small, *unit_temp, *unit_humidity, *unit_pressure;
int datapage_init(const char *data, const char *live) {
 base=load_exact(data,"base.bgra",BYTES);
 wifi[0]=load_exact(data,"wifi_off.bgra",110*29*4); wifi[1]=load_exact(data,"wifi_on.bgra",110*29*4); wifi[2]=load_exact(data,"wifi_unknown.bgra",110*29*4);
 large=load_exact(live,"glyphs_large.bgra",24*12*40*4);small=load_exact(live,"glyphs_small.bgra",14*12*20*4);
 unit_temp=load_exact(live,"unit_temp.bgra",34*25*4);unit_humidity=load_exact(live,"unit_humidity.bgra",18*25*4);unit_pressure=load_exact(live,"unit_pressure.bgra",28*20*4);
 return base&&wifi[0]&&wifi[1]&&wifi[2]&&large&&small&&unit_temp&&unit_humidity&&unit_pressure?0:-1;
}
void datapage_close(void){free(base);for(int i=0;i<3;++i)free(wifi[i]);free(large);free(small);free(unit_temp);free(unit_humidity);free(unit_pressure);}
void datapage_paint(unsigned char *virt){
 static unsigned long last_seq;static int stale=3;static Status status;static Point points[HISTORY_CAP];int count=0;static double seen;struct timespec now;clock_gettime(CLOCK_MONOTONIC,&now);double t=now.tv_sec+now.tv_nsec/1e9;
            Status next;
            if (read_status(&next) == 0 && next.seq != last_seq) {
                status = next; last_seq = next.seq; stale = 0;seen=t;
            } else stale=seen&&t-seen<6?0:3;
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
            for(int y=252;y<281;++y)for(int x=278;x<388;++x)memcpy(virt+((size_t)y*W+x)*4,base,4);
            blit(virt, 278, 252, wifi[fresh ? status.wifi : 2], 110, 0, 110, 29);

            /* Gas channels, drawn into the bottom block.
             * MQ-2 shows a relative percentage (0 % = clean-air baseline,
             * 100 % = alarm threshold); the flame channel stays in mV.
             * -- is drawn when the value is stale or the bridge is still v1. */
            if (fresh && status.mq2 >= 0 && status.mq2 <= 5000) {
                int pct = (status.mq2 - MQ2_BASE_MV) * 100 / (MQ2_ALARM_MV - MQ2_BASE_MV);
                if (pct < 0) pct = 0;
                if (pct > 100) pct = 100;
                snprintf(value, sizeof(value), "%d", pct);
            } else {
                snprintf(value, sizeof(value), "--");
            }
            draw_num(virt, 100, 818, value, small, 14, 20, small_widths);

            if (fresh && status.flame >= 0 && status.flame <= 5000)
                snprintf(value, sizeof(value), "%d", status.flame);
            else
                snprintf(value, sizeof(value), "--");
            draw_num(virt, 100, 848, value, small, 14, 20, small_widths);

            plot(virt, points, count, 0, small);
            plot(virt, points, count, 1, small);

 printf("UNIFIED_DATA seq=%lu fresh=%d history=%d\n",last_seq,stale<3,count);
}

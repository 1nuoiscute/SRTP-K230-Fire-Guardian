/* Pure CPU panel renderer. Status snapshots never access the remote filesystem. */
#define _POSIX_C_SOURCE 200809L
#include "status_cache.h"
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








#define W 536
#define H 348
#define PANEL_Y 612
#define PANEL_BYTES ((size_t)W * H * 4)
#define GLYPHS "0123456789.-"

/* MQ-2 relative-scale anchors (see make_display). */
#define MQ2_BASE_MV  250
#define MQ2_ALARM_MV 500
#define GPIO_INPUT _IOW('G', 1, int)
#define GPIO_OUTPUT _IOW('G', 0, int)
#define GPIO_LOW _IOW('G', 4, int)
#define GPIO_READ _IOW('G', 12, int)

/* mq2 / flame are millivolts; -1 means "no fresh sample" */
typedef struct { int sht, bmp, wifi; double temp, humidity, pressure; unsigned long seq;
                 int mq2, flame; } Status;
typedef struct { char temp[16], humidity[16], pressure[16]; int wifi;
                 char mq2[16], flame[16]; } Display;

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
    (void)path;
    char line[160];
    if (demo_status_snapshot(line, sizeof(line)) < 0) return -1;
    Status s = {0};
    s.mq2 = -1;
    s.flame = -1;
    /* v2 appends the gas channels; fall back to v1 so an older bridge still
     * drives the panel (gas then shows as "--"). */
    if (sscanf(line, "v2 %lu %d %lf %lf %d %lf %d %d %d", &s.seq, &s.sht,
               &s.temp, &s.humidity, &s.bmp, &s.pressure, &s.wifi,
               &s.mq2, &s.flame) != 9) {
        s.mq2 = -1;
        s.flame = -1;
        if (sscanf(line, "v1 %lu %d %lf %lf %d %lf %d", &s.seq, &s.sht,
                   &s.temp, &s.humidity, &s.bmp, &s.pressure, &s.wifi) != 7)
            return -1;
    }
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

    /* MQ-2 is shown as a relative percentage: 0 % at the clean-air baseline,
     * 100 % at the alarm threshold. A true ppm/%LEL figure would need a per
     * sensor R0 from a 24-48 h burn-in plus a known gas concentration. */
    if (fresh && s->mq2 >= 0 && s->mq2 <= 5000) {
        int pct = (s->mq2 - MQ2_BASE_MV) * 100 / (MQ2_ALARM_MV - MQ2_BASE_MV);
        if (pct < 0) pct = 0;
        if (pct > 100) pct = 100;
        snprintf(d->mq2, sizeof(d->mq2), "%d", pct);
    } else {
        strcpy(d->mq2, "--");
    }
    if (fresh && s->flame >= 0 && s->flame <= 5000)
        snprintf(d->flame, sizeof(d->flame), "%d", s->flame);
    else
        strcpy(d->flame, "--");
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
    /* Row 2 of the metrics grid: gas and flame, drawn over the empty
     * .metric-value spans in the background asset. */
    draw_number(dst, 26, 271, d->mq2, small, 14, 20, small_widths);
    draw_number(dst, 280, 271, d->flame, small, 14, 20, small_widths);

    end = draw_number(dst, 60, 324, d->pressure, small, 14, 20, small_widths);
    if (strcmp(d->pressure, "--") != 0)
        blit(dst, end + 3, 324, unit_pressure, 28, 0, 28, 20);
}


static unsigned char *scratch;
static const unsigned char *assets[3],*plarge,*psmall,*ut,*uh,*up;
int panel_init(const char *dir){scratch=malloc(PANEL_BYTES);assets[0]=load_exact(dir,"base_off.bgra",PANEL_BYTES);assets[1]=load_exact(dir,"base_on.bgra",PANEL_BYTES);assets[2]=load_exact(dir,"base_unknown.bgra",PANEL_BYTES);plarge=load_exact(dir,"glyphs_large.bgra",24*12*40*4);psmall=load_exact(dir,"glyphs_small.bgra",14*12*20*4);ut=load_exact(dir,"unit_temp.bgra",34*25*4);uh=load_exact(dir,"unit_humidity.bgra",18*25*4);up=load_exact(dir,"unit_pressure.bgra",28*20*4);return scratch&&assets[0]&&assets[1]&&assets[2]&&plarge&&psmall&&ut&&uh&&up?0:-1;}
void panel_close(void){free(scratch);for(int i=0;i<3;++i)free((void*)assets[i]);free((void*)plarge);free((void*)psmall);free((void*)ut);free((void*)uh);free((void*)up);}
void panel_paint(unsigned char *dst){static Status s;static unsigned long seq;static int stale=3;static double seen;struct timespec now;clock_gettime(CLOCK_MONOTONIC,&now);double t=now.tv_sec+now.tv_nsec/1e9;if(read_status("/sharefs/srtp_clean/ui_overlay/status_ram/status.txt",&s)==0&&s.seq!=seq){seq=s.seq;seen=t;}stale=seen&&t-seen<6?0:3;Display d;make_display(&s,stale<3,&d);paint(scratch,&d,assets,plarge,psmall,ut,uh,up);for(size_t i=0;i<PANEL_BYTES;i+=4){unsigned a=scratch[i+3];if(a==255)memcpy(dst+i,scratch+i,4);else if(a){for(int c=0;c<3;++c)dst[i+c]=(scratch[i+c]*a+dst[i+c]*(255-a)+127)/255;dst[i+3]=255;}}}

/* Isolated K230 RT-Smart OSD experiment. Never reinitializes VO/VB/camera.
 * It occupies OSD0 for a bounded interval and leaves the existing OSD3 box
 * renderer and its ELF untouched. This is not yet a validated board UI.
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "k_module.h"
#include "k_vo_comm.h"
#include "k_video_comm.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vo_api.h"
#if STANDALONE
#include "k_connector_comm.h"
#include "mpi_connector_api.h"
#endif

#ifndef PANEL_W
#define PANEL_W 536
#endif
#ifndef PANEL_H
#define PANEL_H 348
#endif
#ifndef PANEL_Y
#define PANEL_Y 612
#endif
#ifndef HIDE_BOXES
#define HIDE_BOXES 0
#endif
#define PANEL_BYTES ((size_t)PANEL_W * PANEL_H * 4)

static int read_exact_panel(const char *path, void *destination)
{
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        fprintf(stderr, "panel open failed: %s (%d)\n", path, errno);
        return -1;
    }
    int bad = fseek(stream, 0, SEEK_END) != 0 || ftell(stream) != (long)PANEL_BYTES ||
              fseek(stream, 0, SEEK_SET) != 0 ||
              fread(destination, 1, PANEL_BYTES, stream) != PANEL_BYTES;
    fclose(stream);
    if (bad) {
        fprintf(stderr, "panel must be exactly %lu BGRA bytes\n", (unsigned long)PANEL_BYTES);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s panel.bgra seconds(1..30)\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    long duration = strtol(argv[2], &end, 10);
    if (!end || *end || duration < 1 || duration > 30) {
        fprintf(stderr, "duration must be 1..30 seconds\n");
        return 2;
    }
    void *scratch = malloc(PANEL_BYTES);
    if (!scratch) return 3;
    if (read_exact_panel(argv[1], scratch) != 0) {
        free(scratch);
        return 3;
    }

#if STANDALONE
    k_vb_config vb_config = {0};
    vb_config.max_pool_cnt = 8;
    if (kd_mpi_vb_set_config(&vb_config) != 0 || kd_mpi_vb_init() != 0) {
        fprintf(stderr, "standalone VB init failed\n");
        free(scratch);
        return 4;
    }
    k_connector_info connector = {0};
    if (kd_mpi_get_connector_info(NT35516_MIPI_2LAN_540X960_30FPS, &connector) != 0) {
        fprintf(stderr, "standalone connector info failed\n");
        kd_mpi_vb_exit();
        free(scratch);
        return 4;
    }
    k_s32 connector_fd = kd_mpi_connector_open(connector.connector_name);
    if (connector_fd < 0 || kd_mpi_connector_power_set(connector_fd, K_TRUE) != 0 ||
        kd_mpi_connector_init(connector_fd, connector) != 0) {
        fprintf(stderr, "standalone connector init failed\n");
        kd_mpi_vb_exit();
        free(scratch);
        return 4;
    }
#endif

    k_vb_pool_config pool_config;
    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_size = (PANEL_BYTES + 1023) & ~(size_t)1023;
    pool_config.blk_cnt = 1;
    pool_config.mode = VB_REMAP_MODE_NOCACHE;
    k_u32 pool = kd_mpi_vb_create_pool(&pool_config);
    if (pool == VB_INVALID_POOLID) {
        fprintf(stderr, "VB pool creation failed\n");
        free(scratch);
#if STANDALONE
        kd_mpi_vb_exit();
#endif
        return 4;
    }
    k_vb_blk_handle block = kd_mpi_vb_get_block(pool, PANEL_BYTES, NULL);
    if (block == VB_INVALID_HANDLE) {
        fprintf(stderr, "VB block allocation failed\n");
        kd_mpi_vb_destory_pool(pool);
        free(scratch);
#if STANDALONE
        kd_mpi_vb_exit();
#endif
        return 4;
    }
    k_u64 phys = kd_mpi_vb_handle_to_phyaddr(block);
    void *virt = kd_mpi_sys_mmap(phys, PANEL_BYTES);
    if (!phys || !virt) {
        fprintf(stderr, "VB block mapping failed\n");
        kd_mpi_vb_release_block(block);
        kd_mpi_vb_destory_pool(pool);
        free(scratch);
#if STANDALONE
        kd_mpi_vb_exit();
#endif
        return 4;
    }
    memcpy(virt, scratch, PANEL_BYTES);
    free(scratch);

    k_vo_video_osd_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.display_rect.x = 0;
    attr.display_rect.y = PANEL_Y;
    attr.img_size.width = PANEL_W;
    attr.img_size.height = PANEL_H;
    attr.pixel_format = PIXEL_FORMAT_ARGB_8888;
    attr.stride = PANEL_W * 4 / 8;
    attr.global_alptha = 255;

    /* A prior interrupted probe may have left this isolated OSD enabled. */
    kd_mpi_vo_osd_disable(K_VO_OSD0);
    k_s32 result = kd_mpi_vo_set_video_osd_attr(K_VO_OSD0, &attr);
    if (result != 0) {
        fprintf(stderr, "OSD0 attr failed: %d\n", result);
        kd_mpi_sys_munmap(virt, PANEL_BYTES);
        kd_mpi_vb_release_block(block);
        kd_mpi_vb_destory_pool(pool);
#if STANDALONE
        kd_mpi_vb_exit();
#endif
        return 5;
    }
    result = kd_mpi_vo_osd_enable(K_VO_OSD0);
    if (result != 0) {
        fprintf(stderr, "OSD0 enable failed: %d\n", result);
        kd_mpi_sys_munmap(virt, PANEL_BYTES);
        kd_mpi_vb_release_block(block);
        kd_mpi_vb_destory_pool(pool);
#if STANDALONE
        kd_mpi_vb_exit();
#endif
        return 6;
    }

    k_video_frame_info frame;
    memset(&frame, 0, sizeof(frame));
    frame.mod_id = K_ID_VO;
    frame.pool_id = pool;
    frame.v_frame.width = PANEL_W;
    frame.v_frame.height = PANEL_H;
    frame.v_frame.stride[0] = PANEL_W;
    frame.v_frame.phys_addr[0] = phys;
    frame.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    frame.v_frame.priv_data = K_VO_ONLY_CHANGE_PHYADDR;
    result = kd_mpi_vo_chn_insert_frame(K_VO_OSD0 + 3, &frame);
    int box_ok = 1;
    if (result != 0) {
        fprintf(stderr, "OSD0 insert failed: %d\n", result);
    } else {
#if HIDE_BOXES == 1
        k_s32 box_off = kd_mpi_vo_osd_disable(K_VO_OSD3);
        if (box_off != 0) {
            fprintf(stderr, "OSD3 box hide failed: %d\n", box_off);
            box_ok = 0;
        }
#elif HIDE_BOXES == 2
        k_vo_video_osd_attr boxes = {0};
        boxes.display_rect.x = 0;
        boxes.display_rect.y = 0;
        boxes.img_size.width = 540;
        boxes.img_size.height = 960;
        boxes.pixel_format = PIXEL_FORMAT_ARGB_8888;
        boxes.stride = 540 * 4 / 8;
        boxes.global_alptha = 0;
        k_s32 box_off = kd_mpi_vo_set_video_osd_attr(K_VO_OSD3, &boxes);
        if (box_off != 0) {
            fprintf(stderr, "OSD3 box alpha hide failed: %d\n", box_off);
            box_ok = 0;
        }
#endif
        printf("OSD0 panel visible for %ld seconds; original video process untouched\n", duration);
        sleep((unsigned int)duration);
#if HIDE_BOXES == 1
        if (box_off == 0) {
            k_s32 box_on = kd_mpi_vo_osd_enable(K_VO_OSD3);
            if (box_on != 0) {
                fprintf(stderr, "OSD3 box restore failed: %d\n", box_on);
                box_ok = 0;
            }
        }
#elif HIDE_BOXES == 2
        if (box_off == 0) {
            boxes.global_alptha = 255;
            k_s32 box_on = kd_mpi_vo_set_video_osd_attr(K_VO_OSD3, &boxes);
            if (box_on != 0) {
                fprintf(stderr, "OSD3 box alpha restore failed: %d\n", box_on);
                box_ok = 0;
            }
        }
#endif
    }

    k_s32 off = kd_mpi_vo_osd_disable(K_VO_OSD0);
    if (off != 0) fprintf(stderr, "OSD0 disable failed: %d\n", off);
    k_s32 unmapped = kd_mpi_sys_munmap(virt, PANEL_BYTES);
    k_s32 released = kd_mpi_vb_release_block(block);
    k_s32 destroyed = kd_mpi_vb_destory_pool(pool);
#if STANDALONE
    k_s32 vb_exited = kd_mpi_vb_exit();
    if (vb_exited != 0) fprintf(stderr, "standalone VB exit failed: %d\n", vb_exited);
#endif
    if (unmapped || released || destroyed)
        fprintf(stderr, "VB cleanup failed: %d/%d/%d\n", unmapped, released, destroyed);
    return result == 0 && box_ok && off == 0 && unmapped == 0 && released == 0 && destroyed == 0 ? 0 : 7;
}

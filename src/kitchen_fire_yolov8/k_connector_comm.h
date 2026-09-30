/**
 * @file k_sensor_comm.h
 * @author
 * @sxp
 * @version 1.0
 * @date 2023-03-20
 *
 * @copyright
 * Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef __K_CONNECTOR_COMM_H__
#define __K_CONNECTOR_COMM_H__

#include "k_type.h"
#include "k_errno.h"
#include "k_module.h"
#include "k_vo_comm.h"

#ifdef __cplusplus
extern "C" {
#endif /* End of #ifdef __cplusplus */

#define CONNECTOR_NUM_MAX                                  128

#define BACKGROUND_BLACK_COLOR                            (0x808000)
#define BACKGROUND_PINK_COLOR                             (0xffffff)

typedef enum {
    HX8377_V2_MIPI_4LAN_1080X1920_30FPS = 0, // real is HX8399, 8377 is a mistake.
    ILI9806_MIPI_2LAN_480X800_30FPS     = 4,
    ILI9881_MIPI_4LAN_800X1280_60FPS    = 2,
    NT35516_MIPI_2LAN_540X960_30FPS     = 5,
    NT35532_MIPI_2LAN_1080X1920_30FPS   = 6,
    GC9503_MIPI_2LAN_480X800_60FPS      = 7,
    ST7102_MIPI_2LAN_480X640_60FPS      = 8,
    AML020T_MIPI_2LAN_480X360_30FPS     = 9,

    ST7701_V1_MIPI_2LAN_480X800_30FPS = 20,
    ST7701_V1_MIPI_2LAN_480X854_30FPS = 21,
    ST7701_V1_MIPI_2LAN_480X640_30FPS = 22,
    ST7701_V1_MIPI_2LAN_368X544_60FPS = 23,

    LT9611_MIPI_ADAPT_RESOLUTION = 100,
    LT9611_MIPI_4LAN_1920X1080_30FPS,
    LT9611_MIPI_4LAN_1920X1080_60FPS,
    LT9611_MIPI_4LAN_1920X1080_50FPS,
    LT9611_MIPI_4LAN_1920X1080_25FPS,
    LT9611_MIPI_4LAN_1920X1080_24FPS,
    LT9611_MIPI_4LAN_1280X720_60FPS = 110,
    LT9611_MIPI_4LAN_1280X720_50FPS,
    LT9611_MIPI_4LAN_1280X720_30FPS,
    LT9611_MIPI_4LAN_640X480_60FPS = 120,

    VIRTUAL_DISPLAY_DEVICE = 200,
#if defined (CONFIG_MPP_ENABLE_DSI_DEBUGGER)
    DSI_DEBUGGER_DEVICE = 201,
#endif // CONFIG_MPP_ENABLE_DSI_DEBUGGER
} k_connector_type;

typedef struct
{
    k_u32 n;
    k_u32 m;
    k_u32 voc;
    k_u32 hs_freq;
} k_connectori_phy_attr;


typedef struct {
    const char *connector_name;
    k_u32 screen_test_mode;
    k_u32 dsi_test_mode;
    k_u32 bg_color;
    k_u32 intr_line;
    k_u32 pixclk_div;
    k_u32 buff_num;
    k_dsi_lan_num lan_num;
    k_dsi_work_mode work_mode;
    k_vo_dsi_cmd_mode cmd_mode;
    k_connectori_phy_attr phy_attr;
    k_vo_display_resolution resolution;
    k_connector_type type;
} k_connector_info;

typedef enum
{
    K_CONNECTOR_MIRROR_HOR = 1, 
    K_CONNECTOR_MIRROR_VER,
    K_CONNECTOR_MIRROR_BOTH,
}k_connector_mirror;


typedef struct
{
    k_u32 connection_status;
    k_u32 negotiated_count;
    k_connector_type negotiated_types[256];
} k_connector_negotiated_data;

enum k_connector_cmd_type {
    // 0x05  Command type: Single byte data (DCS Short Write, no parameters) 
    // 0x15  Command type: Two byte data (DCS Short Write, 1 parameter)
    // 0x39  Command type: Multi byte data (DCS Long Write, n parameters n > 2)

    CMD_TYPE_DCS_WRITE_05 = 0x05,
    CMD_TYPE_DCS_WRITE_15 = 0x15,
    CMD_TYPE_DCS_WRITE_39 = 0x39,

    // 0x03 Command type: Single byte data  (Generic Short Write, no parameters)
    // 0x13 Command type: Two byte data (Generic Short Write, 1 parameter)
    // 0x23 Command type: Three byte data  (Generic Short Write, 2 parameters)
    // 0x29 Command type: Multi byte data  (Generic Long Write, n parameters n > 2)

    CMD_TYPE_GENERIC_WRITE_03 = 0x03,
    CMD_TYPE_GENERIC_WRITE_13 = 0x13,
    CMD_TYPE_GENERIC_WRITE_23 = 0x23,
    CMD_TYPE_GENERIC_WRITE_29 = 0x29,
};

typedef struct {
    k_u8 cmd_type;
    k_u8 cmd_delay;
    k_u8 cmd_size;
    k_u8 cmd_data[0];
} k_connector_cmd_slice;

#define CONNECTOR_CMD_SEQUENCE(type, delay, ...)                                                                       \
    (type), (delay), sizeof((uint8_t[]) { __VA_ARGS__ }) / sizeof(uint8_t), __VA_ARGS__

#if defined (CONFIG_MPP_ENABLE_DSI_DEBUGGER)
typedef struct {
    k_u32 seq_size;
    k_u8 seq[0];
} k_connector_debugger_init_seq;

typedef struct {
    k_u32 pclk;
    k_u32 fps;
    k_dsi_lan_num lan_num;
    k_u32 intr_line;
    k_u32 buff_num;

    k_u32 hdisplay;
    k_u32 hsync_len;
    k_u32 hback_porch;
    k_u32 hfront_porch;

    k_u32 vdisplay;
    k_u32 vsync_len;
    k_u32 vback_porch;
    k_u32 vfront_porch;
} k_connector_debugger_config;

typedef struct {
    k_u32 setting_size;

    k_connector_info info;

    // must at last
    k_connector_debugger_init_seq init;
} k_connector_debugger_setting;
#endif // CONFIG_MPP_ENABLE_DSI_DEBUGGER

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

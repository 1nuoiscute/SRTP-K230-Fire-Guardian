SDK ?= /home/k230/k230_sdk_v1.6
MPP_SRC_DIR := $(SDK)/src/big/mpp
include $(MPP_SRC_DIR)/userapps/sample/mpp.mk
include $(MPP_SRC_DIR)/userapps/sample/rt-smart.mk

CC := $(SDK)/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/riscv64-unknown-linux-musl-gcc
TARGET := /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/board_ui_overlay/ui_osd_probe.elf
OBJECT := overlay.o
LIVE_TARGET := /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/board_ui_overlay/ui_overlay_live.elf
LIVE_OBJECT := overlay_live.o
KEY_PROBE_TARGET := /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/board_ui_overlay/matrix_key_probe.elf
KEY_PROBE_OBJECT := matrix_key_probe.o
DATA_PROBE_TARGET := /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/board_ui_overlay/data_page_probe.elf
DATA_PROBE_OBJECT := data_page_probe.o
ALPHA_PROBE_TARGET := /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/board_ui_overlay/data_page_probe_alpha.elf
ALPHA_PROBE_OBJECT := data_page_probe_alpha.o
STANDALONE_PROBE_TARGET := /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/board_ui_overlay/data_page_standalone_probe.elf
STANDALONE_PROBE_OBJECT := data_page_standalone_probe.o
CONNECTOR_COMPAT_OBJECT := connector_compat.o
DATA_PAGE_TARGET := /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/board_ui_overlay/data_page_live.elf
DATA_PAGE_OBJECT := data_page.o

all: $(TARGET) $(LIVE_TARGET) $(KEY_PROBE_TARGET) $(DATA_PROBE_TARGET) $(ALPHA_PROBE_TARGET) $(STANDALONE_PROBE_TARGET) $(DATA_PAGE_TARGET)

$(OBJECT): overlay.c
	$(CC) $(CC_CFLAGS) $(MPP_USER_CFLGAS) -std=c11 -c $< -o $@

$(TARGET): $(OBJECT)
	@mkdir -p $(dir $(TARGET))
	$(CC) -o $@ $(LINKFLAG) -n --static $^ -L$(MPP_LIB_PATH) -Wl,--start-group $(MPP_LIBS) -Wl,--end-group

$(LIVE_OBJECT): overlay_live.c
	$(CC) $(CC_CFLAGS) $(MPP_USER_CFLGAS) -std=c11 -c $< -o $@

$(LIVE_TARGET): $(LIVE_OBJECT)
	@mkdir -p $(dir $(LIVE_TARGET))
	$(CC) -o $@ $(LINKFLAG) -n --static $^ -L$(MPP_LIB_PATH) -Wl,--start-group $(MPP_LIBS) -Wl,--end-group

$(KEY_PROBE_OBJECT): matrix_key_probe.c
	$(CC) $(CC_CFLAGS) $(MPP_USER_CFLGAS) -std=c11 -c $< -o $@

$(KEY_PROBE_TARGET): $(KEY_PROBE_OBJECT)
	@mkdir -p $(dir $(KEY_PROBE_TARGET))
	$(CC) -o $@ $(LINKFLAG) -n --static $^ -L$(MPP_LIB_PATH) -Wl,--start-group $(MPP_LIBS) -Wl,--end-group

$(DATA_PROBE_OBJECT): overlay.c
	$(CC) $(CC_CFLAGS) $(MPP_USER_CFLGAS) -std=c11 -DPANEL_H=960 -DPANEL_Y=0 -DHIDE_BOXES=1 -c $< -o $@

$(DATA_PROBE_TARGET): $(DATA_PROBE_OBJECT)
	@mkdir -p $(dir $(DATA_PROBE_TARGET))
	$(CC) -o $@ $(LINKFLAG) -n --static $^ -L$(MPP_LIB_PATH) -Wl,--start-group $(MPP_LIBS) -Wl,--end-group

$(ALPHA_PROBE_OBJECT): overlay.c
	$(CC) $(CC_CFLAGS) $(MPP_USER_CFLGAS) -std=c11 -DPANEL_H=960 -DPANEL_Y=0 -DHIDE_BOXES=2 -c $< -o $@

$(ALPHA_PROBE_TARGET): $(ALPHA_PROBE_OBJECT)
	@mkdir -p $(dir $(ALPHA_PROBE_TARGET))
	$(CC) -o $@ $(LINKFLAG) -n --static $^ -L$(MPP_LIB_PATH) -Wl,--start-group $(MPP_LIBS) -Wl,--end-group

$(STANDALONE_PROBE_OBJECT): overlay.c
	$(CC) $(CC_CFLAGS) -I../kitchen_fire_yolov8 $(MPP_USER_CFLGAS) -std=c11 -DPANEL_H=960 -DPANEL_Y=0 -DSTANDALONE=1 -c $< -o $@

$(CONNECTOR_COMPAT_OBJECT): ../kitchen_fire_yolov8/connector_compat.c
	$(CC) $(CC_CFLAGS) -I../kitchen_fire_yolov8 $(MPP_USER_CFLGAS) -std=c11 -c $< -o $@

$(STANDALONE_PROBE_TARGET): $(STANDALONE_PROBE_OBJECT) $(CONNECTOR_COMPAT_OBJECT)
	@mkdir -p $(dir $(STANDALONE_PROBE_TARGET))
	$(CC) -o $@ $(LINKFLAG) -n --static $^ -L$(MPP_LIB_PATH) -Wl,--start-group $(MPP_LIBS) -Wl,--end-group

$(DATA_PAGE_OBJECT): data_page.c
	$(CC) $(CC_CFLAGS) -I../kitchen_fire_yolov8 $(MPP_USER_CFLGAS) -std=c11 -c $< -o $@

$(DATA_PAGE_TARGET): $(DATA_PAGE_OBJECT) $(CONNECTOR_COMPAT_OBJECT)
	@mkdir -p $(dir $(DATA_PAGE_TARGET))
	$(CC) -o $@ $(LINKFLAG) -n --static $^ -L$(MPP_LIB_PATH) -Wl,--start-group $(MPP_LIBS) -Wl,--end-group

.PHONY: all

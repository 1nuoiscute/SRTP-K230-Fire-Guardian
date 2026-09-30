include ../mkenv.mk

include $(MPP_SRC_DIR)/userapps/sample/mpp.mk
include $(MPP_SRC_DIR)/userapps/sample/rt-smart.mk

CURRECT_DIR_NAME=$(shell basename `pwd`)
BUILD := $(SDK_RTSMART_BUILD_DIR)/mpp/userapps/sample/$(CURRECT_DIR_NAME)

all:
	@mkdir -p $(BUILD)
	@cmake -S . -B $(BUILD)
	@cmake --build $(BUILD)
	@cmake --install $(BUILD)

clean:
	@rm -rf $(BUILD)

.PHONY: all clean

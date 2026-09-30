cmake_minimum_required(VERSION 3.16)
project(stock_fastboot_v06_baseline C CXX)

set(k230_sdk "$ENV{K230_SDK_ROOT}")
set(mpp_sdk_root "$ENV{K230_MPP_ROOT}")
set(nncase_sdk_root "${k230_sdk}/src/big/nncase")

if(NOT EXISTS "${mpp_sdk_root}/userapps/lib/libvo.a")
    message(FATAL_ERROR "K230_MPP_ROOT must point to the complete matching MPP tree")
endif()

set(CMAKE_EXE_LINKER_FLAGS "-T ${nncase_sdk_root}/examples/cmake/link.lds --static")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")

set(src
    main.cc
    model.cc
    mobile_retinaface.cc
    anchors_320.cc
    util.cc
    ${mpp_sdk_root}/userapps/src/connector/mpi_connector.c
    ${mpp_sdk_root}/userapps/src/sensor/mpi_sensor.c
    ${mpp_sdk_root}/userapps/src/sensor/mpi_sensor_type_to_mirror.c
)

add_compile_definitions(
    CONFIG_BOARD_K230_CANMV_RTT_EVB=1
    CONFIG_MPP_DSI_ENABLE_LCD_NT35516=1
    CONFIG_MPP_ENABLE_SENSOR_GC2093=1
    CONFIG_MPP_SENSOR_DEFAULT_CSI=2
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${nncase_sdk_root}/riscv64)
include_directories(${nncase_sdk_root}/riscv64/nncase/include)
include_directories(${nncase_sdk_root}/riscv64/nncase/include/nncase/runtime)
include_directories(${nncase_sdk_root}/riscv64/rvvlib/include)
include_directories(${mpp_sdk_root}/userapps/api)
include_directories(${mpp_sdk_root}/include)
include_directories(${mpp_sdk_root}/include/comm)
include_directories(${mpp_sdk_root}/include/ioctl)
include_directories(${mpp_sdk_root}/userapps/sample/sample_vo)
include_directories(${mpp_sdk_root}/userapps/src/sensor)

link_directories(${mpp_sdk_root}/userapps/lib)
link_directories(${nncase_sdk_root}/riscv64/rvvlib)
link_directories(${nncase_sdk_root}/riscv64/nncase/lib)

add_executable(stock_fastboot_baseline.elf ${src})
target_link_libraries(stock_fastboot_baseline.elf -Wl,--start-group
    rvv Nncase.Runtime.Native nncase.rt_modules.k230 functional_k230
    sys vicap vb cam_device cam_engine hal oslayer ebase fpga isp_drv binder
    auto_ctrol common cam_caldb isi 3a buffer_management cameric_drv video_in
    virtual_hal start_engine cmd_buffer switch cameric_reg_drv t_database_c
    t_mxml_c t_json_c t_common_c vo atomic dma
    -Wl,--end-group)

install(TARGETS stock_fastboot_baseline.elf DESTINATION bin)

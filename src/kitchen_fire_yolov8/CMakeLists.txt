cmake_minimum_required(VERSION 3.16)
project(kitchen_fire_yolov8 C CXX)

# Keep the application/model sources independent from a particular SDK checkout.
# The build command supplies a complete, matching MPP tree through K230_MPP_ROOT.
set(k230_sdk "$ENV{K230_SDK_ROOT}")
set(mpp_sdk_root "$ENV{K230_MPP_ROOT}")
set(nncase_sdk_root "${k230_sdk}/src/big/nncase")

if(NOT EXISTS "${mpp_sdk_root}/userapps/lib/libvo.a")
    message(FATAL_ERROR "K230_MPP_ROOT must point to a complete MPP tree")
endif()

set(CMAKE_EXE_LINKER_FLAGS "-T ${nncase_sdk_root}/examples/cmake/link.lds --static")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")

set(src
    main.cc
    utils.cc
    ai_base.cc
    ob_det.cc
    connector_compat.c
    ${mpp_sdk_root}/userapps/src/sensor/mpi_sensor.c
)
if(EXISTS "${mpp_sdk_root}/userapps/src/sensor/mpi_sensor_type_to_mirror.c")
    list(APPEND src ${mpp_sdk_root}/userapps/src/sensor/mpi_sensor_type_to_mirror.c)
endif()
set(bin kitchen_fire.elf)

# The board firmware uses the v1.6-era MPP ABI, with two vendor additions:
# connector 5 is NT35516 540x960 and sensor 52 is GC2093 CSI2.  Keep the
# original v1.6 structure layout and supply only those board-specific values.
add_compile_definitions(
    CONFIG_BOARD_K230_CANMV=1
    CONFIG_MPP_DSI_ENABLE_LCD_NT35516=1
    CONFIG_MPP_ENABLE_SENSOR_GC2093=1
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${nncase_sdk_root}/riscv64)
include_directories(${nncase_sdk_root}/riscv64/nncase/include)
include_directories(${nncase_sdk_root}/riscv64/nncase/include/nncase/runtime)
include_directories(${nncase_sdk_root}/riscv64/rvvlib/include)
include_directories(${mpp_sdk_root}/userapps/api/)
include_directories(${mpp_sdk_root}/include)
include_directories(${mpp_sdk_root}/include/comm)
include_directories(${mpp_sdk_root}/include/ioctl)
include_directories(${mpp_sdk_root}/userapps/sample/sample_vo)
include_directories(${mpp_sdk_root}/userapps/src/sensor)

set(k230_opencv ${k230_sdk}/src/big/utils/lib/opencv)
include_directories(${k230_opencv}/include/opencv4/)

link_directories(${mpp_sdk_root}/userapps/lib)
link_directories(${nncase_sdk_root}/riscv64/rvvlib/)
link_directories(${nncase_sdk_root}/riscv64/nncase/lib/)
link_directories(${k230_opencv}/lib ${k230_opencv}/lib/opencv4/3rdparty)

add_executable(${bin} ${src})
target_link_libraries(${bin} -Wl,--start-group rvv Nncase.Runtime.Native nncase.rt_modules.k230 functional_k230 sys vicap vb cam_device cam_engine
 hal oslayer ebase fpga isp_drv binder auto_ctrol common cam_caldb isi 3a buffer_management cameric_drv video_in virtual_hal start_engine cmd_buffer
 switch cameric_reg_drv t_database_c t_mxml_c t_json_c t_common_c vo atomic dma -Wl,--end-group)

target_link_libraries(${bin} opencv_imgcodecs opencv_imgproc opencv_core zlib libjpeg-turbo libopenjp2 libpng libtiff libwebp csi_cv)
install(TARGETS ${bin} DESTINATION bin)

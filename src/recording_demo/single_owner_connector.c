#include <stdio.h>
#include "mpi_connector_api.h"
#include "board_connector_profile.h"
int demo_connector_start(void){k_connector_info p;make_board_connector(&p);int fd=kd_mpi_connector_open(p.connector_name);if(fd<0)return -1;int r=kd_mpi_connector_power_set(fd,K_TRUE);if(!r)r=kd_mpi_connector_init(fd,p);printf("SINGLE_CONNECTOR fd=%d init=%d exact_nt35516=1\n",fd,r);if(r){kd_mpi_connector_close(fd);return -1;}return fd;}
void demo_connector_close(int fd){kd_mpi_connector_close(fd);}

/* GC2093 sensor52 has fixed focus. The installed board kernel predates the
 * optional focus-power ioctl; no actuator needs powering for this sensor. */
k_s32 __wrap_kd_mpi_sensor_set_focus_power(k_s32 fd,k_bool power){(void)power;return fd>=0?0:-1;}

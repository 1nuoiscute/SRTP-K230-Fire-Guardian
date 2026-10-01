/* Independent, bounded KEY1 candidate. Link the unchanged overlay_live.c
 * with main renamed and wrap only its OSD0 enable/disable API calls. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include "k_vo_comm.h"
#include "key1_mix_guard.h"

static volatile uint32_t *vo_regs;
static Key1MixGuard mix_guard;

int key1_panel_main(int argc, char **argv);
k_s32 __real_kd_mpi_vo_osd_enable(k_vo_osd osd);
k_s32 __real_kd_mpi_vo_osd_disable(k_vo_osd osd);

static int mix_map(void)
{
    if (vo_regs) return 0;
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("KEY1 candidate /dev/mem"); return -1; }
    void *mapping = mmap(NULL, KEY1_VO_BYTES, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, KEY1_VO_BASE);
    close(fd);
    if (mapping == MAP_FAILED) { perror("KEY1 candidate VO mmap"); return -1; }
    vo_regs = mapping;
    return 0;
}

static void mix_restore(void)
{
    if (!vo_regs || !mix_guard.owned) return;
    int result = key1_mix_release(vo_regs, &mix_guard);
    printf("KEY1 candidate mix restore=%s\n",
           result == 0 ? "readback-ok" : result == 1 ? "driver-changed-skip" : "readback-failed");
}

k_s32 __wrap_kd_mpi_vo_osd_enable(k_vo_osd osd)
{
    k_s32 result = __real_kd_mpi_vo_osd_enable(osd);
    if (result != 0 || osd != K_VO_OSD0) return result;
    if (mix_map() != 0 || key1_mix_acquire(vo_regs, &mix_guard) != 0) {
        fprintf(stderr, "KEY1 candidate mix profile/write rejected\n");
        mix_restore();
        (void)__real_kd_mpi_vo_osd_disable(osd);
        return -1;
    }
    printf("KEY1 candidate OSD0-over-OSD3 applied\n");
    return 0;
}

k_s32 __wrap_kd_mpi_vo_osd_disable(k_vo_osd osd)
{
    if (osd == K_VO_OSD0) mix_restore();
    return __real_kd_mpi_vo_osd_disable(osd);
}

int main(int argc, char **argv)
{
    /* Existing startup passes no duration, so accidental path replacement is
     * rejected. This candidate is not a permanent UI release. */
    if (argc != 3) {
        fprintf(stderr, "usage: %s asset_dir bounded_seconds(1..300)\n", argv[0]);
        return 2;
    }
    char *end;
    errno = 0;
    long seconds = strtol(argv[2], &end, 10);
    if (errno || !argv[2][0] || *end || seconds < 1 || seconds > 300) return 2;
    int result = key1_panel_main(argc, argv);
    mix_restore();
    if (vo_regs) munmap((void *)vo_regs, KEY1_VO_BYTES);
    return result;
}

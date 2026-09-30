/* Bounded RT-Smart probe for the baseboard 4x4 matrix KEY1.
 * Schematic: KEY1 connects row GPIO28 to column GPIO18. Restore row to input.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define GPIO_INPUT _IOW('G', 1, int)
#define GPIO_OUTPUT _IOW('G', 0, int)
#define GPIO_LOW _IOW('G', 4, int)
#define GPIO_READ _IOW('G', 12, int)

typedef struct { uint16_t pin, value; } GpioRequest;

static int set(int fd, unsigned long request, uint16_t pin)
{
    GpioRequest arg = {pin, 0};
    int result = ioctl(fd, request, &arg);
    if (result != 0) fprintf(stderr, "GPIO ioctl 0x%lx pin %u failed: %d\n", request, pin, errno);
    return result;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    int fd = open("/dev/gpio", O_RDWR);
    if (fd < 0) { perror("open /dev/gpio"); return 1; }
    if (set(fd, GPIO_INPUT, 18) || set(fd, GPIO_OUTPUT, 28) ||
        set(fd, GPIO_LOW, 28)) {
        set(fd, GPIO_INPUT, 28); close(fd); return 2;
    }
    puts("Matrix KEY1 probe: press KEY1; 45 seconds, then GPIO28 input restored");
    int last = -1, changes = 0;
    struct timespec pause = {0, 20000000L};
    for (int i = 0; i < 2250; ++i) {
        GpioRequest arg = {18, 0};
        if (ioctl(fd, GPIO_READ, &arg) != 0) {
            fprintf(stderr, "GPIO18 read failed: %d\n", errno);
            break;
        }
        int current = arg.value;
        if (current != last) {
            printf("KEY1 column GPIO18: %s at %.2f s\n", current ? "HIGH" : "LOW", i * .02);
            last = current;
            ++changes;
        }
        nanosleep(&pause, NULL);
    }
    set(fd, GPIO_INPUT, 28);
    close(fd);
    printf("Probe finished; transitions=%d\n", changes);
    return 0;
}

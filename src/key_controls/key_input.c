#include "key_input.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define IOMUX_BASE 0x91105000UL
#define INPUT_LEVEL (1U << 31)
#define FUNCTION_MASK (7U << 11)

static int pressed(const KeyInput *key)
{
    return !(key->iomux[key->pad] & INPUT_LEVEL);
}

int key_input_open(KeyInput *key, unsigned pad)
{
    memset(key, 0, sizeof(*key));
    if (pad != 43 && pad != 14) return -1;
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return -1; }
    void *mapping = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, IOMUX_BASE);
    close(fd);
    if (mapping == MAP_FAILED) { perror("mmap IOMUX"); return -1; }
    key->iomux = mapping;
    key->pad = pad;
    if (key->iomux[pad] & FUNCTION_MASK) {
        fprintf(stderr, "GPIO%u is not in GPIO mode\n", pad);
        key_input_close(key);
        return -1;
    }
    key->stable_pressed = pressed(key);
    key->candidate_pressed = key->stable_pressed;
    return 0;
}

void key_input_close(KeyInput *key)
{
    if (key->iomux) munmap((void *)key->iomux, 4096);
    key->iomux = NULL;
}

int key_input_poll_press(KeyInput *key)
{
    int current = pressed(key);
    if (current != key->candidate_pressed) {
        key->candidate_pressed = current;
        key->candidate_count = 1;
        return 0;
    }
    if (key->candidate_count < 3) ++key->candidate_count;
    if (key->candidate_count == 3 && current != key->stable_pressed) {
        key->stable_pressed = current;
        return current;
    }
    return 0;
}

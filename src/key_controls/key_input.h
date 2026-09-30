#ifndef SRTP_KEY_INPUT_H
#define SRTP_KEY_INPUT_H

#include <stdint.h>

typedef struct {
    volatile uint32_t *iomux;
    unsigned pad;
    int stable_pressed;
    int candidate_pressed;
    unsigned candidate_count;
} KeyInput;

int key_input_open(KeyInput *key, unsigned pad);
void key_input_close(KeyInput *key);
/* Called every 20 ms. Returns 1 on a debounced press, 0 otherwise. */
int key_input_poll_press(KeyInput *key);

#endif

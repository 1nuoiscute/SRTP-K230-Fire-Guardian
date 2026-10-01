/* Board-specific VO order transaction. No GPIO, frame or firmware changes. */
#ifndef SRTP_KEY1_MIX_GUARD_H
#define SRTP_KEY1_MIX_GUARD_H
#include <stdint.h>

#define KEY1_VO_BASE 0x90840000UL
#define KEY1_VO_BYTES 4096U
#define KEY1_MIX_DEFAULT 0xba987654U
#define KEY1_MIX_PANEL_TOP 0xba984657U
#define KEY1_MIX_LOW_OFFSET 0x3ccU
#define KEY1_MIX_HIGH_OFFSET 0x950U
#define KEY1_ENABLE_OFFSET 0x118U
#define KEY1_UPDATE_OFFSET 0x004U
#define KEY1_PANEL_SIZE_OFFSET 0x284U
#define KEY1_BOX_SIZE_OFFSET 0x854U

typedef struct { int owned; } Key1MixGuard;

static inline void key1_mix_fence(void)
{
#if defined(__riscv)
    __asm__ volatile("fence iorw,iorw" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static inline int key1_mix_profile(volatile uint32_t *regs)
{
    return regs[KEY1_MIX_LOW_OFFSET / 4] == 0x3210U &&
           regs[KEY1_PANEL_SIZE_OFFSET / 4] == 0x015c0218U &&
           regs[KEY1_BOX_SIZE_OFFSET / 4] == 0x03c0021cU;
}

/* Called on OSD0 enable only. A failed guard never writes the register page. */
static inline int key1_mix_acquire(volatile uint32_t *regs, Key1MixGuard *guard)
{
    if (guard->owned || !key1_mix_profile(regs) ||
        (regs[KEY1_ENABLE_OFFSET / 4] & 0x90U) != 0x90U ||
        regs[KEY1_MIX_HIGH_OFFSET / 4] != KEY1_MIX_DEFAULT)
        return -1;
    regs[KEY1_MIX_HIGH_OFFSET / 4] = KEY1_MIX_PANEL_TOP;
    key1_mix_fence();
    regs[KEY1_UPDATE_OFFSET / 4] = 0x11U;
    key1_mix_fence();
    guard->owned = 1;
    return regs[KEY1_MIX_HIGH_OFFSET / 4] == KEY1_MIX_PANEL_TOP ? 0 : -1;
}

/* Restore only a value acquired here, while the same visual profile is live.
 * If clock gating or a mode switch intervened, relinquish without writes. */
static inline int key1_mix_release(volatile uint32_t *regs, Key1MixGuard *guard)
{
    if (!guard->owned) return 0;
    guard->owned = 0;
    if (!key1_mix_profile(regs) ||
        !(regs[KEY1_ENABLE_OFFSET / 4] & 0x80U) ||
        regs[KEY1_MIX_HIGH_OFFSET / 4] != KEY1_MIX_PANEL_TOP)
        return 1;
    regs[KEY1_MIX_HIGH_OFFSET / 4] = KEY1_MIX_DEFAULT;
    key1_mix_fence();
    regs[KEY1_UPDATE_OFFSET / 4] = 0x11U;
    key1_mix_fence();
    return regs[KEY1_MIX_HIGH_OFFSET / 4] == KEY1_MIX_DEFAULT ? 0 : -1;
}
#endif

/* Host-only transaction tests: fake register page, no device operations. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "key1_mix_guard.h"

static uint32_t registers[KEY1_VO_BYTES / 4];
static uint32_t saved[KEY1_VO_BYTES / 4];
static Key1MixGuard guard;

static void reset(void)
{
    for (unsigned i = 0; i < KEY1_VO_BYTES / 4; ++i) registers[i] = 0xa5a5a5a5U;
    registers[KEY1_MIX_LOW_OFFSET / 4] = 0x3210U;
    registers[KEY1_MIX_HIGH_OFFSET / 4] = KEY1_MIX_DEFAULT;
    registers[KEY1_PANEL_SIZE_OFFSET / 4] = 0x015c0218U;
    registers[KEY1_BOX_SIZE_OFFSET / 4] = 0x03c0021cU;
    registers[KEY1_ENABLE_OFFSET / 4] = 0x92U;
    guard.owned = 0;
}

static void unchanged(void) { assert(memcmp(registers, saved, sizeof(saved)) == 0); }

int main(void)
{
    reset(); memcpy(saved, registers, sizeof(saved));
    assert(key1_mix_release(registers, &guard) == 0); unchanged();
    assert(key1_mix_acquire(registers, &guard) == 0);
    saved[KEY1_MIX_HIGH_OFFSET / 4] = KEY1_MIX_PANEL_TOP;
    saved[KEY1_UPDATE_OFFSET / 4] = 0x11U;
    unchanged(); /* Exactly the two intended words changed. */
    assert(key1_mix_acquire(registers, &guard) == -1); unchanged();
    registers[KEY1_ENABLE_OFFSET / 4] = 0x82U; /* KEY1 hidden, camera still running. */
    assert(key1_mix_release(registers, &guard) == 0);
    assert(registers[KEY1_MIX_HIGH_OFFSET / 4] == KEY1_MIX_DEFAULT && !guard.owned);
    assert(key1_mix_release(registers, &guard) == 0);

    /* Wrong order, geometry, OSD state and clock-off must never be written. */
    const unsigned slots[] = {KEY1_MIX_LOW_OFFSET / 4, KEY1_MIX_HIGH_OFFSET / 4,
                             KEY1_PANEL_SIZE_OFFSET / 4, KEY1_BOX_SIZE_OFFSET / 4,
                             KEY1_ENABLE_OFFSET / 4};
    for (unsigned i = 0; i < sizeof(slots) / sizeof(slots[0]); ++i) {
        reset(); registers[slots[i]] = 0; memcpy(saved, registers, sizeof(saved));
        assert(key1_mix_acquire(registers, &guard) == -1); unchanged();
    }
    memset(registers, 0xff, sizeof(registers)); guard.owned = 0;
    memcpy(saved, registers, sizeof(saved));
    assert(key1_mix_acquire(registers, &guard) == -1); unchanged();

    /* Ownership/mode changes: skip cleanup rather than restore stale state. */
    for (unsigned i = 0; i < sizeof(slots) / sizeof(slots[0]); ++i) {
        reset(); assert(key1_mix_acquire(registers, &guard) == 0);
        registers[slots[i]] = 0; memcpy(saved, registers, sizeof(saved));
        assert(key1_mix_release(registers, &guard) == 1 && !guard.owned); unchanged();
    }
    reset(); assert(key1_mix_acquire(registers, &guard) == 0);
    memset(registers, 0xff, sizeof(registers)); memcpy(saved, registers, sizeof(saved));
    assert(key1_mix_release(registers, &guard) == 1 && !guard.owned); unchanged();
    puts("KEY1 mix transaction: acquire, hide, exit, repeat, invalid profiles and driver changes passed");
    return 0;
}

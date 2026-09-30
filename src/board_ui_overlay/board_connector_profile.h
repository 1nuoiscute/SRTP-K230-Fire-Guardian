/* Board-specific factual profile from the verified vision ELF; SHA in docs.
 * The generic v0.6 connector table has different field layout and timing.
 * Pointer is rebound in the current process; other 104 bytes remain exact.
 */
#ifndef SRTP_BOARD_CONNECTOR_PROFILE_H
#define SRTP_BOARD_CONNECTOR_PROFILE_H
#include <stdint.h>
#include <string.h>
static const uint32_t board_connector_words[26] = {
    0U, 0U, 8421376U, 10U, 14U, 1U,
    0U, 0U, 9U, 196U, 23U, 163U,
    39600U, 475200U, 620U, 540U, 40U, 20U,
    20U, 1100U, 960U, 40U, 50U, 50U,
    5U, 0U
};
static void make_board_connector(k_connector_info *out)
{
    _Static_assert(sizeof(void *) == 8, "Requires RV64 pointer ABI");
    _Static_assert(sizeof(*out) == 112, "Unexpected connector record size");
    memset(out, 0, sizeof(*out));
    out->connector_name = "nt35516";
    memcpy((unsigned char *)out + sizeof(void *), board_connector_words,
           sizeof(board_connector_words));
}
#endif

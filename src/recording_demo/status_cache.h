#ifndef DEMO_STATUS_CACHE_H
#define DEMO_STATUS_CACHE_H
#include <stddef.h>
#define DEMO_HISTORY_BYTES 65536
/* Only the cache worker may access status files. Snapshot readers never do I/O. */
int demo_status_cache_start(void);
int demo_status_cache_stop(void);
int demo_status_snapshot(char *dst, size_t capacity);
int demo_history_snapshot(char *dst, size_t capacity);
#endif

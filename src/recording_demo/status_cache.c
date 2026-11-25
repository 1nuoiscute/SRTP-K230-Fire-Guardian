#define _POSIX_C_SOURCE 200809L
#include "status_cache.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_t worker;
static int started;
static char status[160], history[DEMO_HISTORY_BYTES];
/* RT threads have small stacks. Keep bounded transfer buffers out of the stack. */
static char incoming_status[160], incoming_history[DEMO_HISTORY_BYTES];
static void close_fd(void *arg) { int *fd=arg; if(*fd>=0) close(*fd); }
static int read_text(const char *path, char *dst, size_t capacity) {
    int fd=open(path,O_RDONLY); if(fd<0)return -1;
    size_t used=0; int good=1;
    pthread_cleanup_push(close_fd,&fd);
    while(used<capacity-1) {
        ssize_t n=read(fd,dst+used,capacity-1-used);
        if(n<0){if(errno==EINTR)continue;good=0;break;}
        if(n==0)break;used+=(size_t)n;
    }
    if(good && used==capacity-1){char extra;ssize_t n;do{n=read(fd,&extra,1);}while(n<0&&errno==EINTR);if(n!=0)good=0;}
    pthread_cleanup_pop(1);
    dst[used]=0;return good&&used>0?(int)used:-1;
}
static void *poll_status(void *unused) {
    (void)unused;
    for(;;){
        int a=read_text("/sharefs/srtp_clean/ui_overlay/status_ram/status.txt",incoming_status,sizeof incoming_status);
        int b=read_text("/sharefs/srtp_clean/ui_overlay/status_ram/history.txt",incoming_history,sizeof incoming_history);
        /* Never hold the lock across a filesystem call, sleep or cancellation point. */
        pthread_mutex_lock(&lock);
        if(a>0)memcpy(status,incoming_status,(size_t)a+1);
        if(b>0)memcpy(history,incoming_history,(size_t)b+1);
        pthread_mutex_unlock(&lock);
        struct timespec delay={0,500000000};nanosleep(&delay,NULL);
    }
    return NULL;
}
int demo_status_cache_start(void){
    if(started)return -1;
    status[0]=0;history[0]=0;
    int rc=pthread_create(&worker,NULL,poll_status,NULL);
    if(rc){printf("STATUS_CACHE_START_ERROR ret=%d\n",rc);return rc;}
    started=1;puts("STATUS_CACHE_START period_ms=500 render_IO=none");return 0;
}
int demo_status_cache_stop(void){
    if(!started)return 0;
    int rc=pthread_cancel(worker);if(rc)return rc;
    rc=pthread_join(worker,NULL);
    if(!rc)started=0;
    printf("STATUS_CACHE_STOP ret=%d\n",rc);return rc;
}
static int snapshot(char *dst,size_t capacity,const char *src){
    if(!dst||capacity==0)return -1;
    pthread_mutex_lock(&lock);
    size_t n=strlen(src);int rc=-1;
    if(n>0&&n<capacity){memcpy(dst,src,n+1);rc=(int)n;}else dst[0]=0;
    pthread_mutex_unlock(&lock);return rc;
}
int demo_status_snapshot(char *dst,size_t capacity){return snapshot(dst,capacity,status);}
int demo_history_snapshot(char *dst,size_t capacity){return snapshot(dst,capacity,history);}

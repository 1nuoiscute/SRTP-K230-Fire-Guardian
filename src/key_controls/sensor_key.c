#include "key_input.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define SENSOR_APP "/sharefs/srtp_clean/key_controls/sensor_mvp_oled_keys"
#define SENSOR_LOG "/tmp/sensor_mvp.log"

static volatile sig_atomic_t running = 1;
static pid_t child = -1;

static void on_stop(int signum)
{
    (void)signum;
    running = 0;
}

static void pause_ms(unsigned ms)
{
    struct timespec ts = {ms / 1000U, (long)(ms % 1000U) * 1000000L};
    nanosleep(&ts, NULL);
}

static int start_sensor(void)
{
    pid_t pid = fork();
    if (pid < 0) { perror("fork sensor"); return -1; }
    if (pid == 0) {
        int log_fd = open(SENSOR_LOG, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (log_fd >= 0) {
            dup2(log_fd, STDOUT_FILENO);
            dup2(log_fd, STDERR_FILENO);
            close(log_fd);
        }
        execl(SENSOR_APP, SENSOR_APP, (char *)NULL);
        _exit(127);
    }
    child = pid;
    printf("K2: sensor started, pid=%ld\n", (long)child);
    return 0;
}

static void stop_sensor(void)
{
    if (child <= 0) return;
    pid_t pid = child;
    kill(pid, SIGTERM);
    for (int i = 0; i < 150; ++i) {
        if (waitpid(pid, NULL, WNOHANG) == pid) {
            child = -1;
            puts("K2: sensor stopped");
            return;
        }
        pause_ms(20);
    }
    fprintf(stderr, "K2: sensor did not stop within 3 seconds, pid=%ld\n",
            (long)pid);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, on_stop);
    signal(SIGTERM, on_stop);
    KeyInput key;
    if (key_input_open(&key, 14)) return 1;
    if (argc > 1 && strcmp(argv[1], "probe") == 0) {
        puts("K2 GPIO14 input ready");
        key_input_close(&key);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "selftest") == 0) {
        for (int cycle = 1; cycle <= 2; ++cycle) {
            printf("K2 selftest cycle %d\n", cycle);
            if (start_sensor()) { key_input_close(&key); return 1; }
            pause_ms(3500);
            stop_sensor();
            if (child > 0) { key_input_close(&key); return 1; }
        }
        key_input_close(&key);
        return 0;
    }
    puts("K2 controls SHT31/BMP280 OLED; initial state OFF");
    while (running) {
        if (child > 0 && waitpid(child, NULL, WNOHANG) == child) {
            child = -1;
            fprintf(stderr, "K2: sensor exited unexpectedly\n");
        }
        if (key_input_poll_press(&key)) {
            if (child > 0) stop_sensor();
            else start_sensor();
        }
        pause_ms(20);
    }
    stop_sensor();
    key_input_close(&key);
    return 0;
}

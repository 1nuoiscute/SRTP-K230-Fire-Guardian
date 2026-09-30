#include "key_input.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define VISION_DIR "/sharefs/srtp_clean/vendor_ob_det"
#define VISION_APP VISION_DIR "/ob_det_fire_mvp_spaces.elf"
#define VISION_MODEL VISION_DIR "/best_vendor80_640.kmodel"
#define UI_APP "/sharefs/srtp_clean/ui_overlay/ui_overlay_live_onebuf.elf"
#define UI_ASSETS "/sharefs/srtp_clean/ui_overlay/live"
#ifndef DATA_APP
#define DATA_APP "/sharefs/srtp_clean/ui_overlay/data_page_live.elf"
#endif
#define DATA_ASSETS "/sharefs/srtp_clean/ui_overlay/data_page"
#define GPIO_INPUT _IOW('G', 1, int)
#define GPIO_OUTPUT _IOW('G', 0, int)
#define GPIO_LOW _IOW('G', 4, int)
#define GPIO_READ _IOW('G', 12, int)

static volatile sig_atomic_t running = 1;
static pid_t child = -1;
static int child_input = -1;
static pid_t ui_child = -1;
static int ui_input = -1;
static pid_t data_child = -1;
static int data_input = -1;
typedef struct { uint16_t pin, value; } GpioRequest;
typedef struct { int fd, stable, candidate, count; } MatrixKey;

static int gpio_action(int fd, unsigned long command, uint16_t pin, uint16_t *value)
{
    GpioRequest arg = {pin, 0};
    if (ioctl(fd, command, &arg) != 0) return -1;
    if (value) *value = arg.value;
    return 0;
}

static int matrix_open(MatrixKey *key)
{
    memset(key, 0, sizeof(*key));
    key->fd = open("/dev/gpio", O_RDWR);
    if (key->fd < 0 || gpio_action(key->fd, GPIO_INPUT, 19, NULL) != 0 ||
        gpio_action(key->fd, GPIO_OUTPUT, 28, NULL) != 0 ||
        gpio_action(key->fd, GPIO_LOW, 28, NULL) != 0) {
        if (key->fd >= 0) close(key->fd);
        key->fd = -1;
        return -1;
    }
    return 0;
}

static void matrix_close(MatrixKey *key)
{
    if (key->fd >= 0) {
        gpio_action(key->fd, GPIO_INPUT, 28, NULL);
        close(key->fd); key->fd = -1;
    }
}

static int matrix_key2_press(MatrixKey *key)
{
    if (key->fd < 0) return 0;
    uint16_t level = 1;
    if (gpio_action(key->fd, GPIO_READ, 19, &level) != 0) return 0;
    int pressed = level == 0;
    if (pressed != key->candidate) { key->candidate = pressed; key->count = 1; return 0; }
    if (key->count < 3) ++key->count;
    if (key->count == 3 && pressed != key->stable) {
        key->stable = pressed;
        return pressed;
    }
    return 0;
}

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

static void forget_child(void)
{
    if (child_input >= 0) close(child_input);
    child_input = -1;
    child = -1;
}

static int wait_for_vision_files(void)
{
    int reported = 0;
    while (running) {
        if (access(VISION_APP, R_OK) == 0 &&
            access(VISION_MODEL, R_OK) == 0) return 0;
        if (!reported) {
            puts("K1: waiting for vision files on /sharefs");
            reported = 1;
        }
        pause_ms(250);
    }
    return -1;
}

static int start_vision(void)
{
    int input_pipe[2];
    if (pipe(input_pipe)) { perror("pipe vision"); return -1; }
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork vision");
        close(input_pipe[0]);
        close(input_pipe[1]);
        return -1;
    }
    if (pid == 0) {
        close(input_pipe[1]);
        if (dup2(input_pipe[0], STDIN_FILENO) < 0) _exit(127);
        close(input_pipe[0]);
        if (chdir(VISION_DIR)) {
            perror("chdir vision");
            _exit(127);
        }
        char *const args[] = {
            (char *)VISION_APP, (char *)VISION_MODEL,
            "0.5", "0.6", "None", "0", NULL
        };
        execv(VISION_APP, args);
        perror("exec vision");
        _exit(127);
    }
    close(input_pipe[0]);
    child_input = input_pipe[1];
    child = pid;
    printf("K1: vision started, pid=%ld\n", (long)child);
    return 0;
}

static void start_ui(void)
{
    if (access(UI_APP, R_OK) != 0) {
        puts("K1: UI unavailable; vision continues");
        return;
    }
    int input_pipe[2];
    if (pipe(input_pipe)) { perror("pipe UI"); return; }
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork UI");
        close(input_pipe[0]); close(input_pipe[1]);
        return;
    }
    if (pid == 0) {
        close(input_pipe[1]);
        if (dup2(input_pipe[0], STDIN_FILENO) < 0) _exit(127);
        close(input_pipe[0]);
        char *const args[] = {(char *)UI_APP, (char *)UI_ASSETS, "0", NULL};
        execv(UI_APP, args);
        perror("exec UI");
        _exit(127);
    }
    close(input_pipe[0]);
    ui_input = input_pipe[1];
    ui_child = pid;
    printf("K1: UI starting, pid=%ld\n", (long)pid);
}

static void stop_ui(void)
{
    if (ui_child <= 0) return;
    if (write(ui_input, "q\n", 2) != 2) perror("send q to UI");
    close(ui_input);
    ui_input = -1;
    int status = 0;
    if (waitpid(ui_child, &status, 0) < 0) perror("wait UI");
    ui_child = -1;
    puts("K1: UI stopped");
}

static void stop_vision(void)
{
    if (child <= 0) return;
    pid_t pid = child;
    if (write(child_input, "q\n", 2) != 2) {
        perror("send q to vision");
    }
    for (int i = 0; i < 500; ++i) {
        int status = 0;
        int result = waitpid(pid, &status, WNOHANG);
        if (result == pid) {
            forget_child();
            puts("K1: vision stopped cleanly");
            return;
        }
        if (result < 0) {
            forget_child();
            puts("K1: previous vision process already exited");
            return;
        }
        pause_ms(20);
    }
    fprintf(stderr, "K1: vision did not exit within 10 seconds, pid=%ld\n",
            (long)pid);
}

static int start_data(void)
{
    if (access(DATA_APP, R_OK) != 0) { puts("K2: data page missing"); return -1; }
    int input_pipe[2];
    if (pipe(input_pipe) != 0) { perror("pipe data"); return -1; }
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork data"); close(input_pipe[0]); close(input_pipe[1]); return -1;
    }
    if (pid == 0) {
        close(input_pipe[1]);
        if (dup2(input_pipe[0], STDIN_FILENO) < 0) _exit(127);
        close(input_pipe[0]);
        char *const args[] = {(char *)DATA_APP, (char *)DATA_ASSETS,
                              (char *)UI_ASSETS, "0", NULL};
        execv(DATA_APP, args);
        perror("exec data"); _exit(127);
    }
    close(input_pipe[0]);
    data_child = pid; data_input = input_pipe[1];
    printf("K2: data page started, pid=%ld\n", (long)pid);
    return 0;
}

static void stop_data(void)
{
    if (data_child <= 0) return;
    if (write(data_input, "q\n", 2) != 2) perror("send q to data");
    close(data_input); data_input = -1;
    int status = 0;
    if (waitpid(data_child, &status, 0) < 0) perror("wait data");
    printf("K2: data page stopped, status=%d\n", status);
    data_child = -1;
}

static void start_vision_with_ui(void)
{
    if (start_vision() == 0) {
        pause_ms(4000);
        start_ui();
    }
}

static int user_requested_quit(void)
{
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(STDIN_FILENO, &read_fds);
    struct timeval timeout = {0, 0};
    if (select(STDIN_FILENO + 1, &read_fds, NULL, NULL, &timeout) > 0) {
        char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1 && ch == 'q') return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, on_stop);
    signal(SIGTERM, on_stop);
    signal(SIGPIPE, SIG_IGN);
    int probe = argc > 1 && strcmp(argv[1], "probe") == 0;
    int selftest = argc > 1 && strcmp(argv[1], "selftest") == 0;
    int selftest_data = argc > 1 && strcmp(argv[1], "selftest_data") == 0;
    if (!probe && !selftest && !selftest_data && wait_for_vision_files()) return 0;
    KeyInput key;
    if (key_input_open(&key, 43)) return 1;
    if (probe) {
        puts("K1 GPIO43 input ready");
        key_input_close(&key);
        return 0;
    }
    if (selftest) {
        for (int cycle = 1; cycle <= 2; ++cycle) {
            printf("K1 selftest cycle %d\n", cycle);
            if (start_vision()) { key_input_close(&key); return 1; }
            pause_ms(4000); /* Original app owns VO before the UI starts. */
            start_ui();
            pause_ms(7000);
            stop_ui();
            stop_vision();
            if (child > 0) { key_input_close(&key); return 1; }
        }
        key_input_close(&key);
        return 0;
    }
    if (selftest_data) {
        puts("K2 selftest: visual -> data -> visual");
        start_vision_with_ui();
        pause_ms(7000);
        stop_ui(); stop_vision();
        if (child > 0 || start_data() != 0) { key_input_close(&key); return 1; }
        pause_ms(7000);
        stop_data();
        start_vision_with_ui();
        pause_ms(7000);
        stop_ui(); stop_vision();
        key_input_close(&key);
        return child > 0 || data_child > 0 ? 1 : 0;
    }
    MatrixKey matrix;
    if (matrix_open(&matrix) != 0)
        puts("K2: matrix key unavailable; original K1 behavior remains");
    puts("K1 controls display; matrix KEY2 switches vision/data; initial state OFF; q exits controller");
    while (running) {
        /* RT-Smart waitpid ignores WNOHANG, so never call it while polling. */
        if (key_input_poll_press(&key)) {
            if (data_child > 0) stop_data();
            else if (child > 0) { stop_ui(); stop_vision(); }
            else start_vision_with_ui();
        }
        if (matrix_key2_press(&matrix)) {
            if (data_child > 0) {
                stop_data();
                start_vision_with_ui();
            } else if (child > 0) {
                stop_ui(); stop_vision();
                if (matrix.fd >= 0) {
                    gpio_action(matrix.fd, GPIO_OUTPUT, 28, NULL);
                    gpio_action(matrix.fd, GPIO_LOW, 28, NULL);
                }
                if (child <= 0 && start_data() != 0) start_vision_with_ui();
                else if (child > 0) start_ui();
            }
        }
        if (user_requested_quit()) break;
        pause_ms(20);
    }
    stop_ui();
    stop_vision();
    stop_data();
    matrix_close(&matrix);
    key_input_close(&key);
    return 0;
}

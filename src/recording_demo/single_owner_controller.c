#define _POSIX_C_SOURCE 200809L
#include "key_input.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#define DIR "/sharefs/srtp_clean/demo_last640_20261124"
static volatile sig_atomic_t running=1;
static pid_t child=-1;static int input=-1;static int fault;
static void stop_signal(int s){(void)s;running=0;}
static void delay(unsigned ms){struct timespec t={ms/1000,(long)(ms%1000)*1000000};nanosleep(&t,NULL);}
static int start(void){
 if(child>0)return 0;int f[2];if(pipe(f))return -1;pid_t pid=fork();
 if(pid==0){close(f[1]);if(dup2(f[0],STDIN_FILENO)<0)_exit(127);close(f[0]);char *const a[]={DIR "/low200.elf",DIR "/single_owner_vision.elf","control",NULL};execv(a[0],a);perror("single exec");_exit(127);}
 close(f[0]);if(pid<0){close(f[1]);return -1;}child=pid;input=f[1];fcntl(input,F_SETFD,FD_CLOEXEC);printf("DEMO_CONTROLLER_ON pid=%ld\n",(long)pid);return 0;
}
static int finish(void){
 if(child<0)return 0;int sent=write(input,"q\n",2)==2;close(input);input=-1;int st=0;pid_t w;do{w=waitpid(child,&st,0);}while(w<0&&errno==EINTR);
 printf("DEMO_CONTROLLER_OFF pid=%ld sent=%d waited=%ld status=%d\n",(long)child,sent,(long)w,st);child=-1;
 int bad=!(sent&&w>0&&WIFEXITED(st)&&WEXITSTATUS(st)==0);fault|=bad;return bad;
}
static int toggle(void){return child>0?finish():start();}
static int action(char c){if(c=='q'){running=0;return 0;}if(c=='k')return toggle();if(child>0&&(c=='1'||c=='2')){char b[2]={c=='1'?'t':'d','\n'};return write(input,b,2)==2?0:-1;}return 0;}
int main(void){
 setvbuf(stdout,NULL,_IONBF,0);signal(SIGINT,stop_signal);signal(SIGTERM,stop_signal);signal(SIGPIPE,SIG_IGN);
 puts("DEMO_CONTROLLER_BOOT READY_wait autostart=ON KEY1=panel KEY2=data K1=power K2=sensors");
 for(int i=0;running&&access(DIR "/READY",R_OK);++i){if(i>=360)return 2;delay(500);}
 if(!running)return 0;KeyInput key;if(key_input_open(&key,43))return 3;
 if(start()){key_input_close(&key);return 4;}
 while(running){
  int rc=0;if(key_input_poll_press(&key))rc=toggle();
  fd_set f;FD_ZERO(&f);FD_SET(STDIN_FILENO,&f);struct timeval t={0,0};if(select(1,&f,NULL,NULL,&t)>0){char c;if(read(0,&c,1)==1)rc=action(c);}
  /* Physical K1 and stdin must not wait for Linux diagnostic files. */
  if(child>0){int st=0;pid_t w=waitpid(child,&st,WNOHANG);if(w==child){printf("DEMO_CHILD_UNEXPECTED_EXIT pid=%ld status=%d\n",(long)child,st);close(input);input=-1;child=-1;fault=1;running=0;}else if(w<0&&errno!=EINTR){fault=1;running=0;}}
  if(rc){fault=1;running=0;}delay(20);
 }
 finish();key_input_close(&key);printf("DEMO_CONTROLLER_END fault=%d\n",fault);return fault?5:0;
}

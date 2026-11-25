#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <string.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <signal.h>
static volatile sig_atomic_t child_pid=-1;
static void forward_stop(int signum){(void)signum;if(child_pid>0)kill(child_pid,SIGTERM);}
#include "k_pm_ioctl.h"
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    printf("PM_QUERY abi profile=%zu governor=%zu profiles=%zu ioctl=%lx/%lx/%lx\n",sizeof(_k_pm_domain_profile),sizeof(_k_pm_domain_governor),sizeof(_k_pm_domain_profiles),(unsigned long)K_IOC_PM_GET_PROFILE,(unsigned long)K_IOC_PM_GET_GOVERNOR,(unsigned long)K_IOC_PM_GET_PROFILES);
    int fd=open("/dev/pm",O_RDWR);
    printf("PM_OPEN fd=%d errno=%d\n",fd,errno);
    if(fd<0)return 2;
    _k_pm_domain_profile current={.domain=PM_DOMAIN_KPU,.index=-999};
    _k_pm_domain_governor gov={.domain=PM_DOMAIN_KPU,.governor=-999};
    struct {k_pm_domain domain;uint32_t number;k_pm_profile profile[128];} table={.domain=PM_DOMAIN_KPU,.number=128};
    int r1=ioctl(fd,K_IOC_PM_GET_PROFILE,&current);
    int r2=ioctl(fd,K_IOC_PM_GET_GOVERNOR,&gov);
    int r3=ioctl(fd,K_IOC_PM_GET_PROFILES,&table);
    printf("KPU_BEFORE profile=%d governor=%d ret=%d/%d/%d\n",current.index,gov.governor,r1,r2,r3);
    if(r1||r2||r3||table.number<4||table.number>128||gov.governor!=PM_GOVERNOR_MANUAL) {close(fd);return 3;}
    if(table.profile[0].freq!=800000000||table.profile[1].freq!=400000000||table.profile[0].volt!=800000||table.profile[1].volt!=800000||table.profile[2].freq!=200000000||table.profile[2].volt!=800000||table.profile[3].freq!=100000000||table.profile[3].volt!=800000) {close(fd);return 4;}
    if(argc!=3 || access(argv[1],R_OK)!=0 || (strcmp(argv[2],"selftest")!=0 && strcmp(argv[2],"control")!=0) || current.index!=0) {
        puts("K1SESSION_REFUSED expected controller+mode and original profile0");close(fd);return 5;
    }
    int original=current.index;
    int target=2;
    _k_pm_domain_profile requested={.domain=PM_DOMAIN_KPU,.index=target};
    int ret=ioctl(fd,K_IOC_PM_SET_PROFILE,&requested);
    printf("KPU_SET profile=%d ret=%d errno=%d requested_hz=%d voltage_uv=%d\n",target,ret,errno,table.profile[target].freq,table.profile[target].volt);
    int readback=ioctl(fd,K_IOC_PM_GET_PROFILE,&current);
    printf("KPU_AFTER profile=%d ret=%d\n",current.index,readback);
    if(ret||readback||current.index!=target){close(fd);return 6;}
    signal(SIGINT,forward_stop);signal(SIGTERM,forward_stop);
    pid_t pid=fork();
    int result=7;
    if(pid==0) {
        char *const args[]={argv[1],argv[2],NULL};
        execv(argv[1],args);perror("K1SESSION_EXEC");_exit(127);
    }
    if(pid>0) {
        child_pid=pid;
        printf("K1SESSION_CHILD pid=%ld profile=2\n",(long)pid);
        int status=0;
        pid_t waited;
        do{waited=waitpid(pid,&status,0);}while(waited<0 && errno==EINTR);
        printf("K1SESSION_CHILD_END waited=%ld status=%d\n",(long)waited,status);
        result=waited==pid && WIFEXITED(status)?WEXITSTATUS(status):8;
    } else perror("K1SESSION_FORK");
    requested.index=original;
    int restored=ioctl(fd,K_IOC_PM_SET_PROFILE,&requested);
    int verified=ioctl(fd,K_IOC_PM_GET_PROFILE,&current);
    printf("K1SESSION_RESTORE ret=%d verify_ret=%d profile=%d\n",restored,verified,current.index);
    close(fd);
    if(restored || verified || current.index!=original)return 9;
    puts("K1SESSION_END");return result;
}

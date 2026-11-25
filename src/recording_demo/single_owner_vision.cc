#define _POSIX_C_SOURCE 200809L
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_tensor.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <time.h>
extern "C" {
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"
#include "mpi_vo_api.h"
#include "k_pm_ioctl.h"
#include "status_cache.h"
int panel_init(const char*);void panel_paint(unsigned char*);void panel_close(void);
int datapage_init(const char*,const char*);void datapage_paint(unsigned char*);void datapage_close(void);
int demo_connector_start(void);void demo_connector_close(int);
}
using namespace nncase;using namespace nncase::runtime;
static_assert(sizeof(k_video_frame_info)==224,"video ABI");
static_assert(sizeof(k_vicap_dev_attr)==120,"capture ABI");
static constexpr int W=536,H=960,SW=1280,SH=720,MW=640,N=8400;
static constexpr size_t RGB_BYTES=size_t(SW)*SH*3,PIXELS=size_t(W)*H,BYTES=PIXELS*4;
static constexpr int PREVIEW_W=960,PREVIEW_H=540;
static constexpr size_t PREVIEW_BYTES=(size_t(PREVIEW_W)*PREVIEW_H*3/2+1023)&~size_t(1023);
static constexpr const char *DIR="/sharefs/srtp_clean/demo_last640_20261124";
static const k_vicap_dev DEV=(k_vicap_dev)0;static const k_vicap_chn CH=(k_vicap_chn)1;
static std::atomic<int> pipeline_stage{0};static std::atomic<unsigned> pipeline_frames{0};
static std::atomic<bool> live{true};static std::atomic<unsigned> events{0};static std::atomic<int> key_error{0};
static void stopped(int){live.store(false);}
static void ms(unsigned n){timespec t{time_t(n/1000),long(n%1000)*1000000};nanosleep(&t,nullptr);}
static double seconds(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static void checked(const char *name,int rc){if(rc){printf("SINGLE_ERROR op=%s ret=%d\n",name,rc);throw std::runtime_error(name);}}
struct Box {float x1,y1,x2,y2,score;};
class Network {
 interpreter net;runtime_tensor input,output;
public:
 Network(){std::ifstream f(std::string(DIR)+"/last640.kmodel",std::ios::binary);if(!f)throw std::runtime_error("model missing");net.load_model(f).expect("model load");
  if(net.inputs_size()!=1||net.outputs_size()!=1||net.input_shape(0)!=dims_t({1,3,640,640})||net.output_shape(0)!=dims_t({1,84,8400})||net.input_desc(0).datatype!=dt_uint8||net.output_desc(0).datatype!=dt_float32)throw std::runtime_error("model ABI shape");
  input=hrt::create(dt_uint8,net.input_shape(0),hrt::pool_shared).expect("input alloc");output=hrt::create(dt_float32,net.output_shape(0),hrt::pool_shared).expect("output alloc");net.input_tensor(0,input).expect("input bind");net.output_tensor(0,output).expect("output bind");puts("SINGLE_MODEL_READY last640 shape1x84x8400");
 }
 std::vector<Box> infer(const unsigned char *rgb){
  {auto map=hrt::map(input,map_write).expect("input map");auto *v=reinterpret_cast<unsigned char*>(map.buffer().data());memset(v,114,3*MW*MW);
   for(int c=0;c<3;++c){const unsigned char *s=rgb+size_t(c)*SW*SH;unsigned char *d=v+size_t(c)*MW*MW;for(int y=0;y<SH/2;++y)for(int x=0;x<MW;++x){size_t a=size_t(2*y)*SW+2*x;d[size_t(y)*MW+x]=(unsigned(s[a])+s[a+1]+s[a+SW]+s[a+SW+1]+2)/4;}}
  }
  hrt::sync(input,sync_write_back,true).expect("input flush");net.run().expect("inference");hrt::sync(output,sync_invalidate,true).expect("output invalidate");auto map=hrt::map(output,map_read).expect("output map");const float *v=reinterpret_cast<const float*>(map.buffer().data());
  std::vector<Box> boxes;boxes.reserve(64);
  for(int i=0;i<N;++i){float score=v[7*N+i];if(!std::isfinite(score)||score<0.25f)continue;float x=v[i],y=v[N+i],w=v[2*N+i],h=v[3*N+i];if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(w)||!std::isfinite(h)||w<=0||h<=0)continue;
   Box b{std::clamp((x-w/2)*2,0.f,float(SW)),std::clamp((y-h/2)*2,0.f,float(SH)),std::clamp((x+w/2)*2,0.f,float(SW)),std::clamp((y+h/2)*2,0.f,float(SH)),score};if(b.x2>b.x1&&b.y2>b.y1)boxes.push_back(b);
  }
  std::sort(boxes.begin(),boxes.end(),[](const Box&a,const Box&b){return a.score>b.score;});if(boxes.size()>300)boxes.resize(300);
  std::vector<Box> kept;for(const auto &b:boxes){bool suppress=false;for(const auto &k:kept){float a=std::max(0.f,std::min(b.x2,k.x2)-std::max(b.x1,k.x1))*std::max(0.f,std::min(b.y2,k.y2)-std::max(b.y1,k.y1));float u=(b.x2-b.x1)*(b.y2-b.y1)+(k.x2-k.x1)*(k.y2-k.y1)-a;if(u>0&&a/u>0.6f){suppress=true;break;}}if(!suppress){kept.push_back(b);if(kept.size()==100)break;}}
  return kept;
 }
};
struct Frame {k_video_frame_info f{};bool held=false;void *map=nullptr;~Frame(){if(map)kd_mpi_sys_munmap(map,RGB_BYTES);if(held){int r=kd_mpi_vicap_dump_release(DEV,CH,&f);if(r)printf("SINGLE_RELEASE_FRAME ret=%d\n",r);}}};
struct Buffer {k_vb_blk_handle h=VB_INVALID_HANDLE;k_u64 phys=0;unsigned char *map=nullptr;};
static void pixel(std::vector<unsigned char>&v,int x,int y,unsigned char r,unsigned char g,unsigned char b){if(x<0||y<0||x>=W||y>=H)return;size_t i=(size_t(y)*W+x)*4;v[i]=b;v[i+1]=g;v[i+2]=r;v[i+3]=255;}
static const char *font(char c){switch(c){
 case 'F':return "11111100001111010000100001000010000";case 'I':return "11111001000010000100001000010011111";case 'R':return "11110100011000111110101001001010001";case 'E':return "11111100001000011110100001000011111";
 case '0':return "01110100011001110101110011000101110";case '1':return "00100011000010000100001000010001110";case '2':return "01110100010000100010001000100011111";case '3':return "11110000010000101110000010000111110";case '4':return "00010001100101010010111110001000010";case '5':return "11111100001000011110000010000111110";case '6':return "01110100001000011110100011000101110";case '7':return "11111000010001000100010000100001000";case '8':return "01110100011000101110100011000101110";case '9':return "01110100011000101111000010000101110";case '.':return "00000000000000000000000000110001100";default:return "00000000000000000000000000000000000";}}
static void label(std::vector<unsigned char>&v,int x,int y,const char *s){for(int n=0;s[n];++n){const char *f=font(s[n]);for(int j=0;j<7;++j)for(int i=0;i<5;++i)if(f[j*5+i]=='1')for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)pixel(v,x+n*12+i*2+dx,y+j*2+dy,255,255,255);}}
static void draw_boxes(std::vector<unsigned char>&v,const std::vector<Box>&boxes){for(const auto &b:boxes){int x1=std::clamp(int(540-b.y2*.75f-2),0,W-1),x2=std::clamp(int(540-b.y1*.75f-2),0,W-1),y1=std::clamp(int(b.x1*.75f),0,H-1),y2=std::clamp(int(b.x2*.75f),0,H-1);for(int t=0;t<2;++t){for(int x=x1;x<=x2;++x){pixel(v,x,y1+t,255,20,20);pixel(v,x,y2-t,255,20,20);}for(int y=y1;y<=y2;++y){pixel(v,x1+t,y,255,20,20);pixel(v,x2-t,y,255,20,20);}}char text[24];snprintf(text,sizeof(text),"FIRE %.2f",double(b.score));label(v,std::min(x1,W-120),std::max(0,y1-16),text);}}
#define G_INPUT _IOW('G',1,int)
#define G_OUTPUT _IOW('G',0,int)
#define G_LOW _IOW('G',4,int)
#define G_READ _IOW('G',12,int)
struct Gpio {uint16_t pin,value;};
static int gpio(int fd,unsigned long cmd,unsigned pin,uint16_t *v=nullptr){Gpio r{uint16_t(pin),0};int rc=ioctl(fd,cmd,&r);if(v)*v=r.value;return rc;}
static void key_worker(){
 int fd=open("/dev/gpio",O_RDWR);if(fd<0||gpio(fd,G_INPUT,18)||gpio(fd,G_INPUT,19)||gpio(fd,G_OUTPUT,28)||gpio(fd,G_LOW,28)){key_error=1;live=false;if(fd>=0)close(fd);return;}
 struct Deb {int stable=0,candidate=0,count=0;} k[2];unsigned ticks=0;
 while(live){for(int i=0;i<2;++i){uint16_t v=1;if(gpio(fd,G_READ,18+i,&v)){key_error=2;live=false;break;}int now=v==0;auto &d=k[i];if(now!=d.candidate){d.candidate=now;d.count=1;}else{if(d.count<3)++d.count;if(d.count==3&&now!=d.stable){d.stable=now;if(now){printf("SINGLE_GPIO_KEY key=%d t=%.3f\n",i+1,seconds());events.fetch_xor(1U<<i);}}}}
  fd_set f;FD_ZERO(&f);FD_SET(0,&f);timeval tv{0,0};if(select(1,&f,nullptr,nullptr,&tv)>0){char c;if(read(0,&c,1)==1){if(c=='q')live=false;if(c=='t')events.fetch_xor(1);if(c=='d')events.fetch_xor(2);if(c=='s')events.fetch_or(4);}}
  if((++ticks%250)==0)printf("SINGLE_HEARTBEAT frames=%u stage=%d remote_file_poll=disabled\n",pipeline_frames.load(),pipeline_stage.load());
  ms(20);
 }
 gpio(fd,G_INPUT,28);close(fd);
}
// Capture and submit preview frames independently of inference and OSD rendering.
static void preview_worker(bool &layer_enabled, k_video_frame_info &retained, std::atomic<int> &error)
{
 const auto channel=VICAP_CHN_ID_0;
 unsigned submitted=0, failures=0;
 double start=seconds();
 while(live){
  k_video_frame_info captured{};
  int result=kd_mpi_vicap_dump_frame(DEV,channel,VICAP_DUMP_YUV,&captured,150);
  if(result){
   if(!live)break;
   if(++failures>=8){printf("PREVIEW_CAPTURE_FAILED rc=%d\n",result);error=1;live=false;}
   continue;
  }
  failures=0;
  const auto &frame=captured.v_frame;
  auto handle=frame.phys_addr[0]?kd_mpi_vb_phyaddr_to_handle(frame.phys_addr[0]):VB_INVALID_HANDLE;
  const bool valid=frame.width==PREVIEW_W&&frame.height==PREVIEW_H&&
   frame.pixel_format==PIXEL_FORMAT_YUV_SEMIPLANAR_420&&
   frame.stride[0]==PREVIEW_W&&frame.stride[1]==PREVIEW_W&&
   frame.phys_addr[1]-frame.phys_addr[0]==size_t(PREVIEW_W)*PREVIEW_H&&handle!=VB_INVALID_HANDLE;
  if(!valid||submitted==0)printf("PREVIEW_SOURCE valid=%d fmt=%u width=%u height=%u stride=%u/%u/%u pool=%u addr=%llx/%llx/%llx pts=%llu\n",int(valid),unsigned(frame.pixel_format),frame.width,frame.height,frame.stride[0],frame.stride[1],frame.stride[2],captured.pool_id,(unsigned long long)frame.phys_addr[0],(unsigned long long)frame.phys_addr[1],(unsigned long long)frame.phys_addr[2],(unsigned long long)frame.pts);
  int enable=0,insert=0;
  if(live&&valid){
   if(!layer_enabled){enable=kd_mpi_vo_enable_video_layer(K_VO_LAYER1);layer_enabled=enable==0;}
   if(layer_enabled){
    k_video_frame_info display=captured;
    display.mod_id=K_ID_VO;
    display.v_frame.priv_data=0;
    insert=kd_mpi_vo_chn_insert_frame(K_VO_DISPLAY_CHN_ID1,&display);
    // The application owns the VICAP reference until VO has acquired its own.
    const int refs=insert?0:kd_mpi_vb_inquire_user_cnt(handle);
    if(!insert&&refs<2){
     printf("PREVIEW_REFERENCE_FAILED refs=%d\n",refs);
     // Keep the camera reference until the main thread stops displaying this frame.
     retained=captured;
     error=2;live=false;break;
    }
    if(!insert&&++submitted%150==0)printf("PREVIEW_PROGRESS frames=%u elapsed=%.3f\n",submitted,seconds()-start);
    if(!insert&&submitted==1)printf("PREVIEW_FIRST_SUBMIT pts=%llu refs=%d\n",(unsigned long long)frame.pts,refs);
   }
  }
  // The driver can report success with empty metadata. No buffer exists to release then.
  int release=handle==VB_INVALID_HANDLE?0:kd_mpi_vicap_dump_release(DEV,channel,&captured);
  if(!valid||enable||insert||release){
   printf("PREVIEW_FAILED valid=%d enable=%d insert=%d release=%d\n",int(valid),enable,insert,release);
   error=2;live=false;
  }
 }
 printf("PREVIEW_END frames=%u status=%d\n",submitted,error.load());
}
int main(int argc,char **argv){(void)argc;(void)argv;
 setvbuf(stdout,nullptr,_IONBF,0);signal(SIGINT,stopped);signal(SIGTERM,stopped);puts("SINGLE_OWNER_BEGIN last640 hardware_preview overlay KEY1_KEY2");
 int status=1,connector=-1;bool vb=false,cam=false,stream=false,osd=false,cache=false,video=false;unsigned frames=0,submissions=0;unsigned full=0;bool panel=true;std::thread keys,preview;std::atomic<int> preview_error{0};k_video_frame_info retained_preview{};std::unique_ptr<Network> net;k_u32 pool=VB_INVALID_POOLID;Buffer buf[2];
 try{
  int pm=open("/dev/pm",O_RDWR);if(pm<0)throw std::runtime_error("pm open");_k_pm_domain_profile cur{};cur.domain=PM_DOMAIN_KPU;cur.index=-1;int r=ioctl(pm,K_IOC_PM_GET_PROFILE,&cur);close(pm);if(r||cur.index!=2)throw std::runtime_error("requires verified KPU200 wrapper");
  k_mpp_chn previous_vi{K_ID_VI,0,0},previous_vo{K_ID_VO,K_VO_DISPLAY_DEV_ID,K_VO_DISPLAY_CHN_ID1};
  printf("SINGLE_PREVIOUS_VIDEO_UNBIND ret=%d\n",kd_mpi_sys_unbind(&previous_vi,&previous_vo));
  printf("SINGLE_PREVIOUS_VO_DISABLE ret=%d\n",kd_mpi_vo_disable());ms(100);
  {FILE *f=fopen((std::string(DIR)+"/control.txt").c_str(),"a");if(f)fclose(f);}
  net=std::make_unique<Network>();
  if(panel_init((std::string(DIR)+"/live").c_str())||datapage_init((std::string(DIR)+"/data_assets").c_str(),(std::string(DIR)+"/live").c_str()))throw std::runtime_error("UI assets");
  checked("status cache",demo_status_cache_start());cache=true;
  k_vb_config conf{};conf.max_pool_cnt=64;
  conf.comm_pool[0].blk_cnt=10;conf.comm_pool[0].blk_size=PREVIEW_BYTES;conf.comm_pool[0].mode=VB_REMAP_MODE_NOCACHE;
  conf.comm_pool[1].blk_cnt=6;conf.comm_pool[1].blk_size=(RGB_BYTES+1023)&~size_t(1023);conf.comm_pool[1].mode=VB_REMAP_MODE_NOCACHE;
  checked("vb_config",kd_mpi_vb_set_config(&conf));checked("vb_init",kd_mpi_vb_init());vb=true;
  connector=demo_connector_start();if(connector<0)throw std::runtime_error("connector");
  k_vb_pool_config pc{};pc.blk_cnt=2;pc.blk_size=(BYTES+1023)&~size_t(1023);pc.mode=VB_REMAP_MODE_NOCACHE;pool=kd_mpi_vb_create_pool(&pc);if(pool==VB_INVALID_POOLID)throw std::runtime_error("OSD pool");
  for(auto &b:buf){b.h=kd_mpi_vb_get_block(pool,BYTES,nullptr);if(b.h==VB_INVALID_HANDLE)throw std::runtime_error("OSD block");b.phys=kd_mpi_vb_handle_to_phyaddr(b.h);b.map=(unsigned char*)kd_mpi_sys_mmap(b.phys,BYTES);if(!b.phys||!b.map||b.map==(void*)-1)throw std::runtime_error("OSD map");memset(b.map,0,BYTES);}
  k_vo_video_osd_attr oa{};oa.img_size.width=W;oa.img_size.height=H;oa.pixel_format=PIXEL_FORMAT_ARGB_8888;oa.stride=W*4/8;oa.global_alptha=255;checked("OSD attr",kd_mpi_vo_set_video_osd_attr(K_VO_OSD0,&oa));
  k_vicap_sensor_info sensor{};checked("sensor52",kd_mpi_vicap_get_sensor_info((k_vicap_sensor_type)52,&sensor));k_vicap_dev_attr dev{};dev.acq_win.width=sensor.width;dev.acq_win.height=sensor.height;dev.mode=VICAP_WORK_ONLINE_MODE;dev.pipe_ctrl.data=0xfdfffff9;dev.sensor_info=sensor;checked("camera dev",kd_mpi_vicap_set_dev_attr(DEV,dev));
  for(int channel=0;channel<3;++channel){
   const bool preview=channel==0;
   const k_vb_pool_config &storage=conf.comm_pool[preview?0:1];
   k_vicap_chn_attr attr{};attr.out_win.width=preview?PREVIEW_W:SW;attr.out_win.height=preview?PREVIEW_H:SH;attr.crop_win=dev.acq_win;attr.scale_win=attr.out_win;
   attr.pix_format=preview?PIXEL_FORMAT_YUV_SEMIPLANAR_420:PIXEL_FORMAT_RGB_888_PLANAR;
   attr.buffer_num=storage.blk_cnt;attr.buffer_size=storage.blk_size;attr.chn_enable=channel<2?K_TRUE:K_FALSE;
   checked("camera channel",kd_mpi_vicap_set_chn_attr(DEV,(k_vicap_chn)channel,attr));
  }
  k_vo_video_layer_attr preview_attr{};
  preview_attr.img_size.width=PREVIEW_H;preview_attr.img_size.height=PREVIEW_W;
  preview_attr.pixel_format=PIXEL_FORMAT_YUV_SEMIPLANAR_420;preview_attr.func=K_ROTATION_90;
  preview_attr.stride=(PREVIEW_H/8-1)+((PREVIEW_W-1)<<16);
  checked("preview layer attr",kd_mpi_vo_set_video_layer_attr(K_VO_LAYER1,&preview_attr));
  checked("camera init",kd_mpi_vicap_init(DEV));cam=true;checked("camera stream",kd_mpi_vicap_start_stream(DEV));stream=true;
  preview=std::thread(preview_worker,std::ref(video),std::ref(retained_preview),std::ref(preview_error));
  std::vector<unsigned char> rgb(RGB_BYTES),canvas(BYTES);keys=std::thread(key_worker);double start=seconds(),next_refresh=0;int failed=0;bool had_boxes=false;
  puts("SINGLE_READY VI0_submit_VO1_960x540 VI1_RGB1280x720 one_OSD two_buffers");
  while(live){double begun=seconds();pipeline_stage=1;
   {Frame f;int rc=kd_mpi_vicap_dump_frame(DEV,CH,VICAP_DUMP_YUV,&f.f,500);if(rc){printf("SINGLE_CAPTURE_ERROR ret=%d\n",rc);if(++failed>=3)throw std::runtime_error("camera capture");continue;}f.held=true;failed=0;const auto &v=f.f.v_frame;
    if(v.width!=SW||v.height!=SH||v.pixel_format!=PIXEL_FORMAT_RGB_888_PLANAR||v.stride[0]!=SW||(v.phys_addr[1]&&v.phys_addr[1]!=v.phys_addr[0]+SW*SH)||(v.phys_addr[2]&&v.phys_addr[2]!=v.phys_addr[0]+2*SW*SH))throw std::runtime_error("capture frame layout");
    f.map=kd_mpi_sys_mmap(v.phys_addr[0],RGB_BYTES);if(!f.map||f.map==(void*)-1){f.map=nullptr;throw std::runtime_error("capture mmap");}memcpy(rgb.data(),f.map,RGB_BYTES);
   }
   pipeline_stage=2;auto boxes=net->infer(rgb.data());if(!live)break;unsigned ev=events.exchange(0);if(ev&1){panel=!panel;printf("SINGLE_KEY1 %s\n",panel?"shown":"hidden");}if(ev&2){full^=1;printf("SINGLE_KEY2 %s same_camera=1\n",full?"data":"camera");}
   const bool has_boxes=!full&&!boxes.empty();
   const bool refresh=!submissions||(ev&7)||has_boxes||had_boxes||((full||panel)&&begun>=next_refresh);
   if(refresh){
   pipeline_stage=3;if(full)datapage_paint(canvas.data());else{std::fill(canvas.begin(),canvas.end(),0);draw_boxes(canvas,boxes);if(panel)panel_paint(canvas.data()+size_t(612)*W*4);}
   if(ev&4){std::string name=std::string(DIR)+(full?"/frame_data.bgra":panel?"/frame_panel.bgra":"/frame_camera.bgra");FILE *f=fopen((name+".new").c_str(),"wb");bool ok=false;if(f){ok=fwrite(canvas.data(),1,BYTES,f)==BYTES;ok=(fclose(f)==0)&&ok;if(ok)ok=rename((name+".new").c_str(),name.c_str())==0;}printf("SINGLE_CANVAS_SNAPSHOT mode=%s panel=%d bytes=%zu ok=%d\n",full?"data":"camera",int(panel),BYTES,int(ok));}
   pipeline_stage=4;Buffer &b=buf[submissions%2];int users=kd_mpi_vb_inquire_user_cnt(b.h);for(int tries=0;users>1&&tries<5;++tries){ms(20);users=kd_mpi_vb_inquire_user_cnt(b.h);}if(users!=1)throw std::runtime_error("target display buffer still owned by VO");
   memcpy(b.map,canvas.data(),BYTES);
   // Do not enable an OSD layer before its first frame is ready.
   if(!osd){
    if(!live)break;
    checked("OSD enable",kd_mpi_vo_osd_enable(K_VO_OSD0));
    osd=true;
   }
   k_video_frame_info f{};f.mod_id=K_ID_VO;f.pool_id=pool;f.v_frame.width=W;f.v_frame.height=H;f.v_frame.stride[0]=W;f.v_frame.pixel_format=PIXEL_FORMAT_ARGB_8888;f.v_frame.phys_addr[0]=b.phys;f.v_frame.priv_data=0;checked("OSD submit",kd_mpi_vo_chn_insert_frame(K_VO_OSD0+3,&f));
   ++submissions;had_boxes=has_boxes;next_refresh=begun+0.5;
   }
   ++frames;pipeline_frames=frames;pipeline_stage=0;if(frames%10==0)printf("SINGLE_PROGRESS frames=%u elapsed=%.3f frame_ms=%.3f mode=%s boxes=%zu osd=%u\n",frames,seconds()-start,1000*(seconds()-begun),full?"data":"camera",boxes.size(),submissions);
  }
  status=0;
 }catch(const std::exception &e){printf("SINGLE_EXCEPTION %s\n",e.what());status=2;}
 live=false;if(preview.joinable())preview.join();if(keys.joinable())keys.join();
 if(!status&&(key_error||preview_error))status=4;
 auto clean=[&](const char *s,int rc){printf("SINGLE_CLEANUP %s ret=%d\n",s,rc);if(rc)status=3;};
 if(cache)clean("status_cache_stop",demo_status_cache_stop());
 if(stream&&!retained_preview.v_frame.phys_addr[0])clean("stream_stop",kd_mpi_vicap_stop_stream(DEV));
 int preview_disabled=video?kd_mpi_vo_disable_video_layer(K_VO_LAYER1):0;
 if(video)clean("preview_layer_disable",preview_disabled);
 if(retained_preview.v_frame.phys_addr[0]){
  if(preview_disabled&&connector>=0){
   preview_disabled=kd_mpi_vo_disable();
   clean("preview_VO_fallback_disable",preview_disabled);
  }
  if(!preview_disabled)clean("preview_retained_release",kd_mpi_vicap_dump_release(DEV,VICAP_CHN_ID_0,&retained_preview));
  else status=3;
 }
 if(stream&&retained_preview.v_frame.phys_addr[0])clean("stream_stop",kd_mpi_vicap_stop_stream(DEV));
 if(cam)clean("camera_deinit",kd_mpi_vicap_deinit(DEV));
 if(osd&&submissions)ms(50);
 if(osd)clean("OSD_disable",kd_mpi_vo_osd_disable(K_VO_OSD0));if(connector>=0)clean("VO_disable",kd_mpi_vo_disable());ms(250);
 for(auto &b:buf){if(b.map&&b.map!=(void*)-1)clean("OSD_unmap",kd_mpi_sys_munmap(b.map,BYTES));if(b.phys){int n=kd_mpi_vb_inquire_user_cnt(b.h);printf("SINGLE_CLEANUP buffer_refs=%d\n",n);if(n==1)clean("VB_release",kd_mpi_vb_release_block(b.h));else status=3;}}
 if(pool!=VB_INVALID_POOLID)clean("VB_pool_destroy",kd_mpi_vb_destory_pool(pool));net.reset();if(vb)clean("VB_exit",kd_mpi_vb_exit());if(connector>=0)demo_connector_close(connector);panel_close();datapage_close();printf("SINGLE_END status=%d frames=%u\n",status,frames);return status;
}

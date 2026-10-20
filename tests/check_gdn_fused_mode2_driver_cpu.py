#!/usr/bin/env python3
"""Compile/run the real fixed64 Mode2 scheduler, driver and epilogues with CPU stubs.

Three host threads emulate one Cube and two Vector lanes. Tile-copy/MMAD stubs
exercise the shared preload BlockMmadTla, L1 adapter and producer/consumer flags.
This is not a CANN build or a model of asynchronous hardware pipelines.
"""
import ast
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = ast.parse((root / 'tests/check_gdn_fused_mode2_epilogues_cpu.py').read_text())
stub = next(ast.literal_eval(n.value) for n in source.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'stub' for t in n.targets))
stub = stub[:stub.index('namespace Catlass::Arch {')]
stub = stub.replace('inline ', 'inline thread_local ')  # only namespace variables below
# Function declarations must not be thread_local.
import re
stub = re.sub(r'inline thread_local (void|uint32_t) (\w+\s*\()', r'inline \1 \2', stub)
stub = stub.replace('enum class HardEvent {MTE2_V,V_MTE3,MTE3_V,MTE3_MTE2,V_MTE2}',
                    'enum class HardEvent {MTE2_V,V_MTE3,MTE3_V,MTE3_MTE2,MTE2_MTE1,MTE1_MTE2,MTE1_M,M_FIX,FIX_M,M_MTE1,V_MTE2}')
stub = stub.replace('eventSets[5][8]', 'eventSets[11][8]')
stub = stub.replace('template<HardEvent E> void SetFlag(uint32_t id) {++eventSets[static_cast<unsigned>(E)][id];}',
                    'template<HardEvent E> void SetFlag(uint32_t id);')
stub = stub.replace('template<HardEvent> void WaitFlag(uint32_t) {}',
                    'template<HardEvent E> void WaitFlag(uint32_t id);')
stub = stub.replace(' T* GetPhyAddr() {return p;}', ' template<class U> LocalTensor<U> ReinterpretCast() {return {reinterpret_cast<U*>(p)};}\n T* GetPhyAddr() {return p;}')
stub = stub.replace('LocalTensor operator[](uint32_t i)', 'LocalTensor operator[](uint64_t i)')
stub = stub.replace('GlobalTensor operator[](uint32_t i)', 'GlobalTensor operator[](uint64_t i)')
stub = stub.replace(' T* p=nullptr;\n GlobalTensor', ' T* p=nullptr;\n void SetGlobalBuffer(T* ptr) {p=ptr;}\n GlobalTensor')
stub = stub.replace('constexpr int PIPE_MTE3=3;', 'constexpr int PIPE_MTE3=3;\nconstexpr int PIPE_FIX=4;')
stub = stub.replace('template<class T> struct GlobalTensor {', 'template<class T> struct GlobalTensor {\n using PrimType=T;')
stub += r'''
#define CATLASS_ARCH 3510
#include <tuple>
#include <algorithm>
using half=_Float16;
using bfloat16_t=uint16_t;
#define __gm__
#define GM_ADDR unsigned char*
#define ASCEND_IS_AIC (AscendC::role==0)
#define ASCEND_IS_AIV (AscendC::role!=0)
namespace AscendC {
inline thread_local unsigned role=0,coreIndex=0;
inline thread_local bool localPending[11][8]{};
template<HardEvent E> void SetFlag(uint32_t id) {
 auto event=static_cast<unsigned>(E);++eventSets[event][id];
 if(role==0 || E==HardEvent::V_MTE2) {
  if(localPending[event][id])throw std::runtime_error("Cube local event set twice without wait");
  localPending[event][id]=true;
 }
}
template<HardEvent E> void WaitFlag(uint32_t id) {
 if(role==0 || E==HardEvent::V_MTE2) {
  auto event=static_cast<unsigned>(E);
  if(!localPending[event][id])throw std::runtime_error("Cube local event wait without set");
  localPending[event][id]=false;
 }
}
inline uint32_t GetBlockIdx() {return role==0?coreIndex:coreIndex*2+role-1;}
inline uint32_t GetSubBlockNum() {return 2;}
inline void SetHF32Mode(bool) {}
inline void SetMMLayoutTransform(bool) {}
template<auto,auto> void CrossCoreSetFlag(uint32_t) {throw std::runtime_error("unexpected intra-core ready flag");}
template<auto,auto> void CrossCoreWaitFlag(uint32_t) {throw std::runtime_error("unexpected intra-core ready wait");}
}
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>
namespace Model {
struct Memory {alignas(32) std::array<unsigned char,512*1024> bytes{};};
inline Memory l1,ub[2],l0a,l0b,l0c;
inline std::mutex mutex;
inline std::condition_variable cv;
inline unsigned pending[9][2]{},sets[9][3]{},waits[9][3]{};
inline bool stop=false;
inline void reset() {
 l1.bytes.fill(0);for(auto& b:ub)b.bytes.fill(0);l0a.bytes.fill(0);l0b.bytes.fill(0);l0c.bytes.fill(0);
 std::memset(pending,0,sizeof(pending));std::memset(sets,0,sizeof(sets));std::memset(waits,0,sizeof(waits));stop=false;
}
}
namespace Catlass {
struct EmptyClass {};
struct MatrixCoord {unsigned r,c;unsigned row()const{return r;}unsigned column()const{return c;}};
using std::min;
template<unsigned D> constexpr unsigned CeilDiv(unsigned v){return(v+D-1)/D;}
namespace layout {struct RowMajor {};struct ColumnMajor {};struct zN {};struct nZ {};}
struct GemmCoord {
 uint32_t mv,nv,kv;
 uint32_t m() const{return mv;} uint32_t n() const{return nv;} uint32_t k() const{return kv;}
};
namespace Arch {
struct AtlasA2 {};
struct Ascend950 {static constexpr uint32_t L1_SIZE=512*1024,L0A_SIZE=64*1024,L0B_SIZE=64*1024,L0C_SIZE=128*1024;};
struct PositionBias {};
struct PositionL1 {};struct PositionL0A {};struct PositionL0B {};struct PositionL0C {};struct PositionGM {};struct PositionUB {};
using CrossCoreFlag=uint32_t;
template<auto Mode,auto Pipe> void CrossCoreSetFlag(uint32_t id) {
 static_assert(Mode==2 && (Pipe==PIPE_MTE3||Pipe==PIPE_FIX));
 std::lock_guard<std::mutex> lock(Model::mutex);
 ++Model::sets[id][AscendC::role];
 if(AscendC::role==0) {++Model::pending[id][0];++Model::pending[id][1];}
 else ++Model::pending[id][AscendC::role-1];
 Model::cv.notify_all();
}
inline void CrossCoreWaitFlag(uint32_t id) {
 std::unique_lock<std::mutex> lock(Model::mutex);
 bool cube=AscendC::role==0;
 bool ready=Model::cv.wait_for(lock,std::chrono::seconds(15),[&]{
  return Model::stop || (cube?(Model::pending[id][0]&&Model::pending[id][1]):Model::pending[id][AscendC::role-1]);
 });
 if(!ready||Model::stop) throw std::runtime_error("cross-core wait timed out at flag "+std::to_string(id));
 ++Model::waits[id][AscendC::role];
 if(cube) {--Model::pending[id][0];--Model::pending[id][1];}
 else --Model::pending[id][AscendC::role-1];
}
template<int Kind> struct Buffer {
 template<class T> AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset) {
  auto* mem=Kind==0?&Model::ub[AscendC::role==0?0:AscendC::role-1]:
   Kind==1?&Model::l1:Kind==2?&Model::l0a:Kind==3?&Model::l0b:&Model::l0c;
  if(offset>=mem->bytes.size()) throw std::runtime_error("buffer offset overflow");
  return {reinterpret_cast<T*>(mem->bytes.data()+offset)};
 }
};
template<class> struct Resource {Buffer<0> ubBuf;Buffer<1> l1Buf;Buffer<2> l0ABuf;Buffer<3> l0BBuf;Buffer<4> l0CBuf;};
}
}
namespace tla {
template<int N> struct Int {constexpr operator unsigned()const{return N;}};
template<class... Ts> struct Shape {};
template<class> struct is_tuple:std::false_type {};
template<class... Ts> struct is_tuple<Shape<Ts...>>:std::true_type {};
template<class> struct is_static:std::true_type {};
template<int I,class... Ts> constexpr unsigned get(Shape<Ts...>){return std::tuple_element_t<I,std::tuple<Ts...>>{};}
template<int I> unsigned get(std::array<unsigned,2> v){return v[I];}
inline auto MakeCoord(unsigned r,unsigned c=0){return std::array<unsigned,2>{r,c};}
inline auto MakeShape(unsigned r,unsigned c=1){return std::array<unsigned,2>{r,c};}
namespace detail {
template<class L> struct isRowMajor:std::is_same<L,Catlass::layout::RowMajor> {};
template<class> struct isVector:std::false_type {};
template<class T,class L> struct iszN:std::is_same<L,Catlass::layout::zN> {};
}
struct Layout {
 unsigned rows,cols;int kind;
 unsigned offset(unsigned r,unsigned c) const {
  if(kind==1)return c*rows+r;
  if(kind==2)return(c/16)*rows*16+r*16+c%16;
  return r*cols+c;
 }
};
template<class T,class L> constexpr Layout MakeLayout(unsigned r,unsigned c) {
 return {r,c,std::is_same_v<L,Catlass::layout::ColumnMajor>?1:std::is_same_v<L,Catlass::layout::zN>?2:0};
}
template<class T> constexpr Layout MakeLayout(T n){return{1,unsigned(n),0};}
inline Layout MakeLayoutL0C(unsigned r,unsigned c) {return{r,c,0};}
template<class Storage,class Pos> struct Tensor {Storage storage;Layout shape;using Position=Pos;
 Storage data()const{return storage;}
 auto coord()const{return std::array<unsigned,2>{0,0};}
};
template<class S,class P> auto MakeTensor(S s,Layout l,P) {return Tensor<S,P>{s,l};}
template<class S,class P> auto GetTile(Tensor<S,P> t,std::array<unsigned,2> coord,std::array<unsigned,2> shape){
 if(coord[0]||coord[1]||shape[0]!=t.shape.rows||shape[1]!=t.shape.cols)
  throw std::runtime_error("Mode2 test expects one complete tile");
 return t;
}
}
namespace Catlass::Gemm {
template<class T,class L> struct GemmType {using Element=T;};
template<class Arch,bool> struct MmadBase {using ArchTag=Arch;};
struct CpuCopy {
 template<class D,class S> void operator()(D d,S s) {
  for(unsigned r=0;r<d.shape.rows;++r)for(unsigned c=0;c<d.shape.cols;++c)
   d.storage.p[d.shape.offset(r,c)]=s.storage.p[s.shape.offset(r,c)];
 }
};
template<unsigned Offset,unsigned Capacity> struct CpuCopyGmToL1:CpuCopy {
 template<class D,class S> void operator()(D d,S s) {
  auto* expected=Model::l1.bytes.data()+Offset;
  if(reinterpret_cast<unsigned char*>(d.storage.p)!=expected ||
     d.shape.rows*d.shape.cols*sizeof(*d.storage.p)>Capacity)
   throw std::runtime_error("GM-to-L1 operand escaped its fixed A/B scratch slot");
  CpuCopy::operator()(d,s);
 }
};
struct CpuMmad {
 template<class C,class A,class B> void operator()(C c,A a,B b,unsigned m,unsigned n,unsigned k,bool,uint8_t) {
  for(unsigned r=0;r<m;++r)for(unsigned col=0;col<n;++col) {
   float sum=0;for(unsigned j=0;j<k;++j)sum+=float(a.storage.p[a.shape.offset(r,j)])*float(b.storage.p[b.shape.offset(j,col)]);
   c.storage.p[c.shape.offset(r,col)]=sum;
  }
 }
};
namespace Tile {
enum class CopyL0CToUBMode {SPLIT_M};
template<class Arch,class A,class LA,class B,class LB,class C,class LC,class Bias=void> struct PackedTileCopyTla {
 using LayoutA=LA;using LayoutB=LB;using LayoutC=LC;
 using ElementAccumulator=float;
 using CopyL1ToBT=CpuCopy;
 using L1AAlignHelper=void;using L1BAlignHelper=void;
 static constexpr bool HAS_BIAS=false;
 using LayoutTagL1A=layout::zN;using LayoutTagL1B=layout::zN;
 using LayoutTagL0A=layout::RowMajor;using LayoutTagL0B=layout::RowMajor;
 template<class> using CopyGmToL1A=CpuCopyGmToL1<0,16*1024>;
 template<class> using CopyGmToL1B=CpuCopyGmToL1<16*1024,48*1024>;
 using CopyL1ToL0A=CpuCopy;using CopyL1ToL0B=CpuCopy;
 template<class> using CopyL0CToDst=CpuCopy;
};
struct CpuCopyUB {
 template<class D,class S> void operator()(D d,S s) {
  auto offset=reinterpret_cast<unsigned char*>(d.storage.p)-Model::ub[0].bytes.data();
  for(unsigned lane=0;lane<2;++lane) {
   auto* dst=reinterpret_cast<float*>(Model::ub[lane].bytes.data()+offset);
   unsigned rows=d.shape.rows/2;
   for(unsigned r=0;r<rows;++r)for(unsigned c=0;c<d.shape.cols;++c)
    dst[r*d.shape.cols+c]=s.storage.p[s.shape.offset(lane*rows+r,c)];
  }
 }
};
template<class Arch,class A,class LA,class B,class LB,class C,class LC,class Bias,CopyL0CToUBMode>
struct PackedTileCopyTlaToUB:PackedTileCopyTla<Arch,A,LA,B,LB,C,LC,Bias> {
 template<class> using CopyL0CToDst=CpuCopyUB;
};
}
namespace Block {
template<class Policy,class L1,class L0,class A,class B,class C,class Bias=void,class Copy=void,class Mmad=CpuMmad>
struct BlockMmadTla;
}
}
'''
cpp = r'''
#include <thread>
#include <iostream>
#include <algorithm>
#include "gemm/kernel/gdn_fwd_ho_fused_mode2_kernel.hpp"
using Tiling=GDN::ChunkFwdHOFusedTilingData;
void check(bool value,const char* reason) {if(!value)throw std::runtime_error(reason);}
struct Bf16 {
 uint16_t bits=0;Bf16()=default;
 explicit Bf16(float x){uint32_t v;std::memcpy(&v,&x,4);v+=0x7fff+((v>>16)&1);bits=v>>16;}
 operator float()const{uint32_t v=uint32_t(bits)<<16;float x;std::memcpy(&x,&v,4);return x;}
};
Tiling tiling(unsigned chunks,unsigned kv=1,unsigned vh=1,unsigned tasks=1) {
 Tiling t{};t.executionMode=2;t.shapeBatch=1;t.seqlen=chunks*64;t.chunkSize=64;
 t.kHeadDim=t.vHeadDim=128;t.kNumHead=kv;t.vNumHead=vh;t.numChunksPerBatch=chunks;
 t.singleCoreTaskNum=tasks;t.useG=true;t.scale=0.125f;
 unsigned pos=0;
 for(auto member:{&Tiling::vWorkspaceOffset,&Tiling::hWorkspaceOffset,&Tiling::vUpdateWorkspaceOffset,
                 &Tiling::oAttnWorkspaceOffset,&Tiling::oAfterMaskWorkspaceOffset,&Tiling::oHWorkspaceOffset,&Tiling::oVWorkspaceOffset}) {
  t.*member=pos;pos+=tasks*128*128*4;
 }
 return t;
}
void schedulerChecks() {
 using S=Catlass::Gemm::Block::BlockSchedulerGdnFwdHOFusedMode2;
 auto t=tiling(3,2,8,8);t.shapeBatch=2;t.singleCoreTaskNum=16;
 for(unsigned task=0;task<16;++task) {
  S s;s.Init(&t,task);unsigned chunks=0;
  do {
   auto c=s.offsets;unsigned batch=task/8,head=task%8,start=chunks*64;
   check(c.qk==((batch*2+head/4)*192+start)*128,"GQA Q/K offset");
   check(c.w==((batch*8+head)*192+start)*128,"value-head W offset");
   check(c.uv==c.w&&c.gate==(batch*8+head)*192+start,"U/gate offset");
   check(c.state==uint64_t(task)*128*128,"state offset");
   check(c.hInput==(uint64_t(task)*3+chunks)*128*128,"H offset");
   check(c.hOutput==c.hInput+128*128,"next H offset");
   check(c.attnWork==uint64_t(task)*64*64&&c.tokenWork==uint64_t(task)*64*128&&
         c.stateWork==uint64_t(task)*128*128,"single-slot workspace offset");
   check(c.first==(chunks==0)&&c.last==(chunks==2),"chunk flags");++chunks;
  }while(s.Next());check(chunks==3,"chunk count");
 }
 S inactive;inactive.Init(&t,16);check(!inactive.isRunning,"inactive core");
 t.seqlen=(int64_t(1)<<30);t.numChunksPerBatch=t.seqlen/64;t.shapeBatch=2;
 S large;large.Init(&t,15);check(large.offsets.w>uint64_t(1)<<32,"64-bit token offset");
 auto good=tiling(1);check(S::Supports(&good),"fixed64 routing");
 for(unsigned mode=0;mode<6;++mode) {
  auto bad=good;
  if(mode==0)bad.chunkSize=128;if(mode==1)bad.seqlen=65;if(mode==2)bad.useExp2=true;
  if(mode==3)bad.useGk=true;if(mode==4)bad.isVariedLen=1;if(mode==5)bad.vHeadDim=256;
  check(!S::Supports(&bad),"fallback routing");
 }
}
template<class D,class G,class S,bool Direct,bool L1,bool Defaults=false>
void run(unsigned chunks,bool initial,bool store,unsigned core=0,unsigned batches=1,unsigned kHeads=1,unsigned vHeads=1) {
 unsigned tasks=batches*vHeads;
 auto t=tiling(chunks,kHeads,vHeads,tasks);t.shapeBatch=batches;t.useInitialState=initial;t.storeFinalState=store;
 const unsigned tokens=chunks*64;
 const unsigned tokenBase=core*tokens*128,gateBase=core*tokens;
 const unsigned qkBase=((core/vHeads)*kHeads+(core%vHeads)/(vHeads/kHeads))*tokens*128;
 const unsigned stateBase=core*128*128,hBase=core*chunks*128*128;
 std::vector<D> k(batches*kHeads*tokens*128),q(k.size()),w(tasks*tokens*128),u(w.size()),
                h(tasks*chunks*128*128),vn(w.size()),o(w.size(),D(-9.f));
 std::vector<G> g(tasks*tokens);
 std::vector<S> init(tasks*128*128),final(init.size(),S(-9.f));
 std::vector<unsigned char> workspace(7*tasks*128*128*4);
 for(unsigned i=0;i<k.size();++i) {
  k[i]=D(std::sin(i*0.013f)*0.02f);q[i]=D(std::cos(i*0.017f)*0.02f);
 }
 for(unsigned i=0;i<w.size();++i) {
  w[i]=D(std::sin(i*0.021f)*0.02f);u[i]=D(std::cos(i*0.007f)*0.1f);
 }
 for(unsigned i=0;i<g.size();++i)g[i]=G(-float(i%64)*0.002f);
 for(unsigned i=0;i<init.size();++i)init[i]=S(std::sin(i*0.005f)*0.02f);
 std::vector<float> state(128*128),expected(o.size(),-9.f);
 std::vector<D> expectedH(h.size()),expectedVnew(vn.size());
 for(unsigned i=0;i<state.size();++i)state[i]=initial?float(init[stateBase+i]):0.f;
 for(unsigned chunk=0;chunk<chunks;++chunk) {
  std::vector<D> cubeH(state.size()),vnew(64*128),decay(64*128),mask(64*64);
  for(unsigned i=0;i<state.size();++i) {
   cubeH[i]=D(state[i]);if(!L1)expectedH[hBase+chunk*128*128+i]=cubeH[i];
  }
  for(unsigned r=0;r<64;++r)for(unsigned c=0;c<128;++c) {
   float wh=0;for(unsigned j=0;j<128;++j)wh+=float(w[tokenBase+(chunk*64+r)*128+j])*float(cubeH[j*128+c]);
   float v=float(u[tokenBase+(chunk*64+r)*128+c])-wh;
   vnew[r*128+c]=D(v);if(!L1)expectedVnew[tokenBase+(chunk*64+r)*128+c]=D(v);decay[r*128+c]=D(v*std::exp(float(g[gateBase+chunk*64+63])-float(g[gateBase+chunk*64+r])));
  }
  for(unsigned r=0;r<64;++r)for(unsigned c=0;c<64;++c) {
   float dot=0;for(unsigned j=0;j<128;++j)dot+=float(q[qkBase+(chunk*64+r)*128+j])*float(k[qkBase+(chunk*64+c)*128+j]);
   mask[r*64+c]=D(r>=c?dot*std::exp(std::min(float(g[gateBase+chunk*64+r])-float(g[gateBase+chunk*64+c]),0.f)):0.f);
  }
  for(unsigned r=0;r<64;++r)for(unsigned c=0;c<128;++c) {
   float inter=0,intra=0;
   for(unsigned j=0;j<128;++j)inter+=float(q[qkBase+(chunk*64+r)*128+j])*float(cubeH[j*128+c]);
   for(unsigned j=0;j<64;++j)intra+=float(mask[r*64+j])*float(vnew[j*128+c]);
   expected[tokenBase+(chunk*64+r)*128+c]=float(D((inter*std::exp(float(g[gateBase+chunk*64+r]))+intra)*t.scale));
  }
  for(unsigned r=0;r<128;++r)for(unsigned c=0;c<128;++c) {
   float delta=0;for(unsigned j=0;j<64;++j)delta+=float(k[qkBase+(chunk*64+j)*128+r])*float(decay[j*128+c]);
   state[r*128+c]=state[r*128+c]*std::exp(float(g[gateBase+chunk*64+63]))+delta;
  }
 }
 Model::reset();std::exception_ptr failure[3];std::thread threads[3];
 for(unsigned role=0;role<3;++role)threads[role]=std::thread([&,role] {
  try {
   AscendC::role=role;AscendC::coreIndex=core;AscendC::lane=role?role-1:0;
   using Driver=std::conditional_t<Defaults,
    Catlass::Gemm::Kernel::GDNFwdHOFusedMode2Kernel<D,G,S>,
    Catlass::Gemm::Kernel::GDNFwdHOFusedMode2Kernel<D,G,S,Direct,L1>>;
   Driver driver;
   auto address=[](auto& v){return reinterpret_cast<unsigned char*>(v.data());};
   driver.Init(address(k),address(w),address(u),address(g),address(init),address(q),address(h),address(vn),
               address(final),address(o),reinterpret_cast<unsigned char*>(&t),workspace.data());
   unsigned cubeGmReads=0;
   if(role!=0) AscendC::beforeCopy=[&](const void* src) {
    auto begin=reinterpret_cast<uintptr_t>(workspace.data()),addr=reinterpret_cast<uintptr_t>(src);
    if(addr>=begin&&addr<begin+workspace.size())++cubeGmReads;
   };
   driver.Process();
   if(role==0) {
    for(auto& event:AscendC::localPending)for(bool pending:event)check(!pending,"Cube local event leaked");
    unsigned calls=chunks*5-(store?0:1);
    for(unsigned id:{0U,1U}) {
     check(AscendC::eventSets[unsigned(AscendC::HardEvent::MTE1_MTE2)][id]==calls+1,
           "L1 slot must be seeded once and released once per Mmad");
     check(AscendC::eventSets[unsigned(AscendC::HardEvent::M_MTE1)][id]==calls+1,
           "L0 operand slot must be seeded once and released once per Mmad");
    }
    check(AscendC::eventSets[unsigned(AscendC::HardEvent::FIX_M)][0]==calls+1,
          "L0C slot must be seeded once and released once per Mmad");
   } else {
    check(cubeGmReads==(Direct?0:chunks*5-(store?0:1)),"Cube-to-UB must skip GM input copies");
    check(AscendC::eventSets[unsigned(AscendC::HardEvent::V_MTE2)][7]==chunks,
          "OV2 must release gate/scratch once per chunk");
    check(!AscendC::localPending[unsigned(AscendC::HardEvent::V_MTE2)][7],
          "last chunk gate release event must be consumed at exit");
   }
  }catch(...) {
   failure[role]=std::current_exception();std::lock_guard<std::mutex> lock(Model::mutex);Model::stop=true;Model::cv.notify_all();
  }
 });
 for(auto& thread:threads)thread.join();for(auto& e:failure)if(e)std::rethrow_exception(e);
 for(unsigned i=0;i<o.size();++i)if(std::abs(float(o[i])-expected[i])>=1.e-6f) {
  std::cerr<<"O mismatch index="<<i<<" actual="<<float(o[i])<<" expected="<<expected[i]<<" direct="<<Direct<<" l1="<<L1<<" chunks="<<chunks<<" init="<<initial<<" store="<<store<<"\n";
  check(false,"O reference mismatch");
 }
 for(unsigned i=0;i<final.size();++i) {
  float value=store&&i>=stateBase&&i<stateBase+state.size()?float(S(state[i-stateBase])):float(S(-9.f));
  check(std::abs(float(final[i])-value)<1.e-6f,"final state mismatch or write into another task");
 }
 for(unsigned i=0;i<h.size();++i)check(float(h[i])==float(expectedH[i]),"GM H handoff snapshot or task bounds mismatch");
 for(unsigned i=0;i<vn.size();++i)check(float(vn[i])==float(expectedVnew[i]),"GM Vnew handoff or task bounds mismatch");
 for(unsigned flag=0;flag<9;++flag)for(unsigned lane=0;lane<2;++lane)check(Model::pending[flag][lane]==0,"unconsumed token");
 for(unsigned flag=0;flag<9;++flag) {
  bool cube=flag<5;unsigned count=chunks; // initial H plus nonfinal HV2 = chunks
  if(cube) {
   check(Model::sets[flag][0]==count,"Cube set count");
   for(unsigned role=1;role<3;++role)check(Model::waits[flag][role]==count,"Vector wait count");
  }else {
   for(unsigned role=1;role<3;++role)check(Model::sets[flag][role]==count,"Vector set count");
   check(Model::waits[flag][0]==count,"Cube wait count");
  }
 }
}
template<class D,class G,class S> void routes() {
 for(unsigned chunks:{1U,3U})for(bool initial:{false,true})for(bool store:{false,true}) {
  run<D,G,S,false,false>(chunks,initial,store);run<D,G,S,true,false>(chunks,initial,store);
  run<D,G,S,false,true>(chunks,initial,store);run<D,G,S,true,true>(chunks,initial,store);
 }
}
int main() {
 schedulerChecks();
 for(unsigned chunks:{1U,3U})for(bool initial:{false,true})for(bool store:{false,true}) {
  run<_Float16,float,float,true,false,true>(chunks,initial,store);
  run<Bf16,float,float,true,false,true>(chunks,initial,store);
 }
 for(unsigned core=0;core<16;++core) {
  run<_Float16,float,float,true,false>(3,true,true,core,2,2,8);
  run<_Float16,float,float,true,true>(3,true,true,core,2,2,8);
 }
 routes<_Float16,float,float>();routes<_Float16,float,_Float16>();
 routes<_Float16,_Float16,float>();routes<_Float16,_Float16,_Float16>();
 routes<Bf16,float,float>();routes<Bf16,float,Bf16>();
 routes<Bf16,Bf16,float>();routes<Bf16,Bf16,Bf16>();
 std::cout<<"PASS: real Mode2 scheduler/driver/Cube block and epilogues; 304 cases (including default transport and multi-task GQA), all four transports, 1/3 chunks, 8 input/gate/state dtype combinations, initial/final state switches, three threads, reference O/state, fixed L1 A/B slots and once-per-loop event seeding/drain; GQA/64-bit/fallback scheduler checks\n";
}
'''
with tempfile.TemporaryDirectory(prefix='mode2-driver-') as folder:
    path = Path(folder)
    (path / 'kernel_operator.h').write_text(stub)
    for name in ['catlass/catlass.hpp', 'catlass/arch/resource.hpp', 'catlass/arch/cross_core_sync.hpp',
                 'catlass/gemm/gemm_type.hpp', 'catlass/gemm/block/block_mmad.hpp',
                 'kernel_utils/tile/copy_l0c_to_ub.hpp', 'catlass/coord.hpp', 'catlass/gemm_coord.hpp',
                 'catlass/gemm/dispatch_policy.hpp', 'catlass/gemm/helper.hpp',
                 'catlass/gemm/tile/tile_copy.hpp', 'catlass/gemm/tile/tile_mmad.hpp',
                 'tla/layout.hpp', 'tla/tensor.hpp']:
        header=path / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text('#pragma once\n#include "kernel_operator.h"\n')
    header=path / 'catlass/epilogue/block/block_epilogue.hpp'
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text('#pragma once\nnamespace Catlass::Epilogue::Block {template<class...> class BlockEpilogue;}\n')
    (path / 'check.cpp').write_text(cpp)
    subprocess.run(['clang++', '-std=c++17', '-O1', '-pthread', '-ffp-contract=off',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-I',str(path), '-I',
                    str(root / 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h_o_fused/op_kernel/arch35'),
                    '-I', str(root / 'fla/ops/ascendc/common'),
                    str(path / 'check.cpp'), '-o', str(path / 'check')], check=True)
    subprocess.run([str(path / 'check')], check=True)

#!/usr/bin/env python3
"""Execute the new MODE2 HV/OV C++ headers with synchronous CPU API emulation.

Requires clang++ with _Float16 and ASan/UBSan. This exercises the actual C++
computations and address slices, not Ascend compilation, DMA ordering or flags.
Run from any directory: python3 tests/check_gdn_fused_mode2_epilogues_cpu.py
"""
from pathlib import Path
import argparse
import subprocess,tempfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--skip-qkmask',action='store_true',
                    help='Isolate HV1/HV2/OV2 checks from independent QK-mask edits.')
parser.add_argument('--output-only',action='store_true',
                    help='Verify OV2 output and producer notification independently.')
parser.add_argument('--update-only',action='store_true',
                    help='Verify only HV2 state initialization and recurrence paths.')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
base=root/'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h_o_fused/op_kernel/arch35/epilogue/block'
stub=r'''
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <stdexcept>
#include <functional>
#define CATLASS_DEVICE inline
#define __simd_vf__
#define __ubuf__
constexpr int PIPE_V=0;
constexpr int PIPE_MTE3=3;
constexpr uint32_t EVENT_ID0=0;
constexpr uint32_t EVENT_ID1=1;
constexpr uint32_t EVENT_ID2=2;
constexpr uint32_t EVENT_ID3=3;
constexpr uint32_t EVENT_ID4=4;
constexpr uint32_t EVENT_ID5=5;
constexpr uint32_t EVENT_ID6=6;
constexpr uint32_t EVENT_ID7=7;
namespace AscendC {
enum class HardEvent {MTE2_V,V_MTE3,MTE3_V,MTE3_MTE2,V_MTE2};
enum class RoundMode {CAST_NONE,CAST_RINT};
enum class CMPMODE {GT};
inline unsigned gateCopies=0,gateCasts=0;
inline const void* guardedInput=nullptr;
inline bool guardedReady=false;
inline uint32_t expectedReadyFlag=0,readyWaits=0;
inline bool gateMaskPrepared=false;
inline const float* preparedDecay=nullptr;
inline std::array<float,2048> expectedDecay;
inline unsigned eventSets[5][8]{};
inline std::function<void(uint32_t)> beforeCrossWait;
inline std::function<void(uint32_t)> beforeCrossSet;
inline std::function<void()> beforeVectorSub;
inline std::function<void(const void*)> beforeCopy;
inline std::function<void(const void*)> beforeLoad;
template<HardEvent E> void SetFlag(uint32_t id) {++eventSets[static_cast<unsigned>(E)][id];}
template<HardEvent> void WaitFlag(uint32_t) {}
template<int> void PipeBarrier() {}
inline void ResetMask() {}
template<class T> struct LocalTensor {
 T* p=nullptr;
 LocalTensor operator[](uint32_t i) const {return {p+i};}
 void SetSize(uint32_t) {}
 T* GetPhyAddr() {return p;}
};
template<class T> struct GlobalTensor {
 T* p=nullptr;
 GlobalTensor operator[](uint32_t i) const {return {p+i};}
};
template<class D,class S> void DataCopy(D d,S s,uint32_t n) {
 if(beforeCopy) beforeCopy(s.p);
 if(guardedInput==s.p && !guardedReady) throw std::runtime_error("QK read before OC1 ready");
 if(n==64) ++gateCopies;
 for(uint32_t i=0;i<n;++i) d.p[i]=s.p[i];
}
struct DataCopyParams {
 uint16_t blockCount,blockLen,srcGap,dstGap;
};
template<class T> void DataCopy(LocalTensor<T> d,LocalTensor<T> s,DataCopyParams p) {
 for(unsigned b=0;b<p.blockCount;++b) for(unsigned i=0;i<p.blockLen*32/sizeof(T);++i)
  d.p[b*(p.blockLen+p.dstGap)*32/sizeof(T)+i]=s.p[b*(p.blockLen+p.srcGap)*32/sizeof(T)+i];
}
template<class D,class S> void Cast(LocalTensor<D> d,LocalTensor<S> s,RoundMode,uint32_t n) {
 if(n==64) ++gateCasts;
 for(uint32_t i=0;i<n;++i) d.p[i]=static_cast<D>(static_cast<float>(s.p[i]));
}
template<class T> void Duplicate(LocalTensor<T> d,float v,uint32_t n) {
 for(uint32_t i=0;i<n;++i) d.p[i]=static_cast<T>(v);
}
struct CopyParams {unsigned dstBlockStride,srcBlockStride,dstRepeatStride,srcRepeatStride;};
template<class T> void Copy(LocalTensor<T> d,LocalTensor<T> s,uint32_t n,uint32_t repeats,CopyParams p) {
 for(unsigned r=0;r<repeats;++r) for(unsigned i=0;i<n;++i)
  d.p[r*p.dstRepeatStride*32/sizeof(T)+i]=s.p[r*p.srcRepeatStride*32/sizeof(T)+i];
}
template<class T,int Rank,int Axis> void Broadcast(LocalTensor<T> d,LocalTensor<T> s,
 uint32_t* dst,uint32_t*,LocalTensor<uint8_t>) {
 static_assert(Rank==2);
 for(unsigned r=0;r<dst[0];++r) for(unsigned c=0;c<dst[1];++c)
  d.p[r*dst[1]+c]=s.p[Axis==0?c:r];
}
template<class T> void Mul(LocalTensor<T> d,LocalTensor<T> a,LocalTensor<T> b,uint32_t n) {
 if(beforeLoad) {beforeLoad(a.p); beforeLoad(b.p);}
 for(unsigned i=0;i<n;++i) d.p[i]=a.p[i]*b.p[i];
}
template<class T> void Sub(LocalTensor<T> d,LocalTensor<T> a,LocalTensor<T> b,uint32_t n) {
 if(beforeVectorSub) beforeVectorSub();
 if(beforeLoad) {beforeLoad(a.p); beforeLoad(b.p);}
 for(unsigned i=0;i<n;++i) d.p[i]=a.p[i]-b.p[i];
}
template<class T> void Add(LocalTensor<T> d,LocalTensor<T> a,LocalTensor<T> b,uint32_t n) {
 if(beforeLoad) {beforeLoad(a.p); beforeLoad(b.p);}
 for(unsigned i=0;i<n;++i) d.p[i]=a.p[i]+b.p[i];
}
template<class T> void Muls(LocalTensor<T> d,LocalTensor<T> a,float b,uint32_t n) {
 for(unsigned i=0;i<n;++i) d.p[i]=a.p[i]*b;
}
template<auto F,class... Args> void VF_CALL(Args... args) {
 if(guardedInput && !guardedReady) {
  F(args...); gateMaskPrepared=true;
 } else F(args...);
}
inline uint32_t lane=0;
inline uint32_t GetSubBlockIdx() {return lane;}
namespace MicroAPI {
template<class T> struct RegTensor {std::array<T,64> v;};
struct MaskReg {std::array<bool,64> v; MaskReg() {v.fill(true);}};
enum class MaskPattern {ALL};
enum class LoadDist {DIST_NORM,DIST_BRC_B32};
template<class T,MaskPattern> MaskReg CreateMask() {return {};}
template<class T,LoadDist D=LoadDist::DIST_NORM> void LoadAlign(RegTensor<T>& d,T* p) {
 if(beforeLoad) beforeLoad(p);
 for(int i=0;i<64;++i) d.v[i]=p[D==LoadDist::DIST_BRC_B32 ? 0:i];
}
template<class T> void StoreAlign(T* p,const RegTensor<T>& s,MaskReg) {
 for(int i=0;i<64;++i) p[i]=s.v[i];
}
template<class T> void Duplicate(RegTensor<T>& d,float x,MaskReg) {d.v.fill(x);}
inline void Mins(RegTensor<float>& d,const RegTensor<float>& a,float x,MaskReg) {
 for(int i=0;i<64;++i) d.v[i]=std::min(a.v[i],x);
}
template<class T,CMPMODE> void Compare(MaskReg& d,const RegTensor<T>& a,const RegTensor<T>& b,MaskReg) {
 for(int i=0;i<64;++i) d.v[i]=a.v[i]>b.v[i];
}
template<class T> void Select(RegTensor<T>& d,const RegTensor<T>& a,const RegTensor<T>& b,MaskReg m) {
 for(int i=0;i<64;++i) d.v[i]=m.v[i]?a.v[i]:b.v[i];
}
inline void Sub(RegTensor<float>& d,const RegTensor<float>& a,const RegTensor<float>& b,MaskReg) {
 for(int i=0;i<64;++i) d.v[i]=a.v[i]-b.v[i];
}
inline void Mul(RegTensor<float>& d,const RegTensor<float>& a,const RegTensor<float>& b,MaskReg) {
 for(int i=0;i<64;++i) d.v[i]=a.v[i]*b.v[i];
}
inline void Add(RegTensor<float>& d,const RegTensor<float>& a,const RegTensor<float>& b,MaskReg) {
 for(int i=0;i<64;++i) d.v[i]=a.v[i]+b.v[i];
}
inline void Exp(RegTensor<float>& d,const RegTensor<float>& a,MaskReg) {
 for(int i=0;i<64;++i) d.v[i]=std::exp(a.v[i]);
}
inline void Muls(RegTensor<float>& d,const RegTensor<float>& a,float b,MaskReg) {
 for(int i=0;i<64;++i) d.v[i]=a.v[i]*b;
}
}
}
namespace Catlass::Arch {
struct Ascend950 {};
using CrossCoreFlag=uint32_t;
template<auto Mode,auto Pipe> void CrossCoreSetFlag(CrossCoreFlag flag) {
 static_assert(Mode==0x2 && Pipe==PIPE_MTE3);
 if(AscendC::beforeCrossSet) AscendC::beforeCrossSet(flag);
}
inline void CrossCoreWaitFlag(CrossCoreFlag flag) {
 if(AscendC::beforeCrossWait) AscendC::beforeCrossWait(flag);
 if(AscendC::guardedInput && flag==AscendC::expectedReadyFlag) {
  if(!AscendC::gateMaskPrepared) throw std::runtime_error("OC1 waited before independent gate/mask computation");
  for(unsigned i=0;i<2048;++i) {
   if(std::abs(AscendC::preparedDecay[i]-AscendC::expectedDecay[i])>1e-6f)
    throw std::runtime_error("gate/mask value incomplete at OC1 wait");
  }
  AscendC::guardedReady=true; ++AscendC::readyWaits;
 }
}
struct UbBuf {
 alignas(32) std::array<unsigned char,256*1024> memory;
 UbBuf() {memory.fill(0xA5);}
 template<class T> AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset) {
  return {reinterpret_cast<T*>(memory.data()+offset)};
 }
};
template<class> struct Resource {UbBuf ubBuf;};
}
'''
cpp=r'''
#include <iostream>
#include <vector>
#include <stdexcept>
#include "block_epilogue_gdn_fused_qkmask.hpp"
#include "block_epilogue_gdn_fused_vnew.hpp"
#include "block_epilogue_gdn_fused_update.hpp"
#include "block_epilogue_gdn_fused_output.hpp"
using namespace Catlass::Epilogue::Block;
using U=Catlass::Epilogue::gdnFusedMode2UbMallocTable;
using Resource=Catlass::Arch::Resource<Catlass::Arch::Ascend950>;
struct Bf16 {
 uint16_t bits=0;
 Bf16()=default;
 explicit Bf16(float x) {uint32_t v;std::memcpy(&v,&x,4);v+=0x7fff+((v>>16)&1);bits=v>>16;}
 operator float() const {uint32_t v=uint32_t(bits)<<16;float f;std::memcpy(&f,&v,4);return f;}
};
static_assert(sizeof(Bf16)==2);
template<class T> AscendC::GlobalTensor<T> gm(std::vector<T>& a) {return {a.data()};}
void requireAt(bool b,int line) {if(!b) throw std::runtime_error("CPU emulation mismatch at C++ line "+std::to_string(line));}
#define require(condition) requireAt((condition),__LINE__)
template<class T> void close(T actual,float expected) {
 float want=float(static_cast<T>(expected));
 require(std::abs(float(actual)-want)<=1e-6f*std::max(1.f,std::abs(want)));
}
void maskUntouched(Resource& r) {
 for(unsigned i=U::mask;i<U::mask+4096;++i) require(r.ubBuf.memory[i]==0xA5);
}
template<class T> struct Type {using Element=T;};
unsigned zn(unsigned rows,unsigned row,unsigned col) {return 16+(col/16)*rows*16+row*16+col%16;}
template<class D> void checkL1(const std::vector<D>& a,const std::vector<D>& before,
 unsigned rows,unsigned cols,unsigned lane,bool enabled) {
 for(unsigned j=0;j<16;++j) {close(a[j],-13.f);close(a[rows*cols+16+j],-13.f);}
 for(unsigned row=0;row<rows;++row) for(unsigned col=0;col<cols;++col) {
  unsigned idx=zn(rows,row,col);
  if(!enabled || row/(rows/2)!=lane) require(float(a[idx])==float(before[idx]));
 }
}
template<class D,class G> void hv1ov2(bool fromFixpipe,bool toL1) {
 std::vector<G> g(64);
 std::vector<float> qk(64*64); std::vector<D> masked(64*64,D(-11.f));
 std::vector<D> l1(64*64+32,D(-13.f));
 for(unsigned i=0;i<qk.size();++i) qk[i]=std::sin(float(i)*0.002f);
 std::vector<D> u(64*128),vn(u.size()),vu(u.size()),out(u.size());
 std::vector<float> wh(u.size()),inter(u.size()),intra(u.size());
 std::vector<D> l1New(u.size()+32,D(-13.f)),l1Update(l1New),l1O(l1New);
 for(int i=0;i<64;++i) g[i]=G(-0.015625f*i);
 for(unsigned i=0;i<u.size();++i) {
  u[i]=D(std::sin(float(i)*0.03f)); wh[i]=std::cos(float(i)*0.01f)*0.1f;
  inter[i]=std::sin(float(i)*0.04f); intra[i]=std::cos(float(i)*0.02f);
 }
 for(unsigned lane=0;lane<2;++lane) {
  Resource r; AscendC::lane=lane;
  mode2_fixed64_detail::GateBuffer<G> gate(r);
  auto mask=r.ubBuf.GetBufferByByte<float>(U::mask);
  BlockEpilogue<Catlass::Epilogue::EpilogueAtlasGDNFusedMode2Qkmask,
    Type<D>,Type<G>,Type<float>,Type<float>> ov1(r,mask,gate.Tensor(),fromFixpipe,toL1,AscendC::LocalTensor<D>{l1.data()+16});
  GdnFusedMode2Vnew<D,G> hv1(r,gate.Tensor(),fromFixpipe,toL1,
    AscendC::LocalTensor<D>{l1New.data()+16},AscendC::LocalTensor<D>{l1Update.data()+16});
  GdnFusedMode2Output<D,G> ov2(r,gate.Tensor(),fromFixpipe,toL1,AscendC::LocalTensor<D>{l1O.data()+16});
  maskUntouched(r); // Constructors must not create the caller-owned mask.
  AscendC::Duplicate(mask,0.f,32*32);
  for(unsigned row=0;row<32;++row) AscendC::Duplicate(mask[row*32],1.f,row+1);
  std::array<unsigned char,4096> maskBefore;
  std::memcpy(maskBefore.data(),mask.p,4096);
  for(unsigned chunk=0;chunk<3;++chunk) {
   for(int i=0;i<64;++i) g[i]=G(-0.015625f*(i+chunk));
   AscendC::gateCopies=AscendC::gateCasts=0;
   gate.Prepare(gm(g));
   std::array<float,64> gateBefore; std::memcpy(gateBefore.data(),gate.Tensor().p,256);
   #ifndef MODE2_SKIP_QKMASK
   AscendC::guardedInput=qk.data()+lane*2048;
   AscendC::guardedReady=false; AscendC::readyWaits=0; AscendC::gateMaskPrepared=false;
   AscendC::expectedReadyFlag=7+chunk;
   AscendC::preparedDecay=r.ubBuf.GetBufferByByte<float>(U::g_left_ping).p;
   for(unsigned row=0;row<32;++row) for(unsigned col=0;col<64;++col) {
    unsigned absRow=lane*32+row;
    AscendC::expectedDecay[row*64+col]=absRow>=col?
     std::exp(std::min(float(g[absRow])-float(g[col]),0.f)):0.f;
   }
   auto l1Before=l1;
   unsigned qkCopies=0;
   auto attnUb=r.ubBuf.GetBufferByByte<float>(U::attn_ping).p;
   std::memset(AscendC::eventSets,0,sizeof(AscendC::eventSets));
   AscendC::beforeCopy=[&](const void* src) {
    if(src==qk.data()+lane*2048) {require(!fromFixpipe);++qkCopies;}
   };
   AscendC::beforeLoad=[&](const void* src) {
    if(src==attnUb) require(AscendC::guardedReady);
   };
   AscendC::beforeCrossWait=[&](uint32_t flag) {
    require(flag==AscendC::expectedReadyFlag);
    if(fromFixpipe) std::memcpy(attnUb,qk.data()+lane*2048,2048*sizeof(float));
   };
   AscendC::GlobalTensor<D> gmOutput=toL1?AscendC::GlobalTensor<D>{}:gm(masked);
   AscendC::GlobalTensor<float> gmInput=fromFixpipe?AscendC::GlobalTensor<float>{}:gm(qk);
   ov1(gmOutput,gmInput,AscendC::expectedReadyFlag);
   require(qkCopies==(fromFixpipe?0:1));
   require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::MTE2_V)][0]==(fromFixpipe?0:1));
   require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::V_MTE3)][0]==1);
   AscendC::beforeCopy={}; AscendC::beforeLoad={}; AscendC::beforeCrossWait={};
   require(AscendC::readyWaits==1);
   AscendC::guardedInput=nullptr;
   for(unsigned row=lane*32;row<(lane+1)*32;++row) for(unsigned col=0;col<64;++col) {
    unsigned i=row*64+col;
    D actual=toL1?l1[16+(col/16)*64*16+row*16+col%16]:masked[i];
    close(actual,row>=col?qk[i]*std::exp(std::min(float(g[row])-float(g[col]),0.f)):0.f);
    if(toL1) close(masked[i],-11.f);
   }
   for(unsigned guard=0;guard<16;++guard) {close(l1[guard],-13.f);close(l1[64*64+16+guard],-13.f);}
   if(!toL1) for(auto value:l1) close(value,-13.f);
   // All rows belonging to the other AIV must retain their previous contents.
   for(unsigned row=(1-lane)*32;row<(2-lane)*32;++row) for(unsigned col=0;col<64;++col) {
    unsigned idx=16+(col/16)*64*16+row*16+col%16;
    require(float(l1[idx])==float(l1Before[idx]));
   }
   #endif
   auto newBefore=l1New,updateBefore=l1Update,oBefore=l1O;
   auto gmNewBefore=vn,gmUpdateBefore=vu,gmOBefore=out;
   unsigned hc1Waits=0,oc2Waits=0,oc3Waits=0,whCopies=0,interCopies=0,intraCopies=0;
   auto temp=r.ubBuf.GetBufferByByte<float>(U::shard_mem).p;
   auto uf=r.ubBuf.GetBufferByByte<float>(U::u_fp_ping).p;
   auto interUb=r.ubBuf.GetBufferByByte<float>(U::o_inter_ping).p;
   auto whBegin=reinterpret_cast<uintptr_t>(r.ubBuf.memory.data()+U::v_work_ping);
   AscendC::beforeLoad=[&](const void* src) {
    auto addr=reinterpret_cast<uintptr_t>(src);
    if(addr>=whBegin && addr<whBegin+4096*4) require(hc1Waits==1);
   };
   AscendC::beforeCopy=[&](const void* src) {
    if(src==wh.data()+lane*4096) {require(!fromFixpipe && hc1Waits==1);++whCopies;}
   };
   AscendC::beforeCrossWait=[&](uint32_t flag) {
    require(flag==1); ++hc1Waits; require(hc1Waits==1);
    for(unsigned j=0;j<64;++j) close(temp[j],std::exp(float(g[63])-float(g[j])));
    for(unsigned j=0;j<4096;++j) close(uf[j],float(u[lane*4096+j]));
    if(fromFixpipe) std::memcpy(r.ubBuf.memory.data()+U::v_work_ping,wh.data()+lane*4096,4096*4);
   };
   std::array<float,4096> vnewFpBefore;
   auto vnewFp=r.ubBuf.GetBufferByByte<float>(U::v_new_ping).p;
   std::memcpy(vnewFpBefore.data(),vnewFp,4096*4);
   unsigned decaySignals=0,newSignals=0;
   AscendC::beforeVectorSub=[&]() {require(decaySignals==1 && newSignals==0);};
   AscendC::beforeCrossSet=[&](uint32_t flag) {
    if(flag==11) {
     ++decaySignals; require(decaySignals==1 && newSignals==0);
     // Vnew has not even been computed when HC2 is notified.
     require(std::memcmp(vnewFpBefore.data(),vnewFp,4096*4)==0);
     for(unsigned row=lane*32;row<(lane+1)*32;++row) for(unsigned col=0;col<128;++col) {
      unsigned i=row*128+col;
      D decay=toL1?l1Update[zn(64,row,col)]:vu[i];
      close(decay,(float(u[i])-wh[i])*std::exp(float(g[63])-float(g[row])));
      require(float(vn[i])==float(gmNewBefore[i]));
      require(float(l1New[zn(64,row,col)])==float(newBefore[zn(64,row,col)]));
      close(uf[(row-lane*32)*128+col],float(u[i]));
     }
     require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::MTE3_V)][4]==1);

     // Model HC2 already writing its disjoint UB destination while Vnew runs.
     std::memset(r.ubBuf.memory.data()+U::h_work_ping,0x5A,8192*4);
    } else {
     require(flag==12 && decaySignals==1); ++newSignals; require(newSignals==1);
     for(unsigned row=lane*32;row<(lane+1)*32;++row) for(unsigned col=0;col<128;++col) {
      unsigned i=row*128+col;
      D value=toL1?l1New[zn(64,row,col)]:vn[i];close(value,float(u[i])-wh[i]);
     }
     require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::MTE3_V)][4]==1);

    }
   };
   std::memset(AscendC::eventSets,0,sizeof(AscendC::eventSets));
   hv1(toL1?AscendC::GlobalTensor<D>{}:gm(vn),toL1?AscendC::GlobalTensor<D>{}:gm(vu),gm(u),
       fromFixpipe?AscendC::GlobalTensor<float>{}:gm(wh),1,11,12);
   require(decaySignals==1 && newSignals==1);
   AscendC::beforeCrossSet={}; AscendC::beforeVectorSub={};
   require(whCopies==(fromFixpipe?0:1));
   require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::MTE2_V)][4]==(fromFixpipe?1:2));
   require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::MTE3_V)][4]==1);
   checkL1(l1New,newBefore,64,128,lane,toL1); checkL1(l1Update,updateBefore,64,128,lane,toL1);
   if(toL1) for(unsigned row=lane*32;row<(lane+1)*32;++row) for(unsigned col=0;col<128;++col) {
    unsigned i=row*128+col; require(float(vn[i])==float(gmNewBefore[i]));require(float(vu[i])==float(gmUpdateBefore[i]));
    vn[i]=l1New[zn(64,row,col)];vu[i]=l1Update[zn(64,row,col)];
   }
   require(hc1Waits==1);
   AscendC::beforeLoad={};
   AscendC::beforeCopy=[&](const void* src) {
    if(src==inter.data()+lane*4096) {require(!fromFixpipe && oc2Waits==1);++interCopies;}
    if(src==intra.data()+lane*4096) {require(!fromFixpipe && oc3Waits==1);++intraCopies;}
   };
   AscendC::beforeCrossWait=[&](uint32_t flag) {
    if(flag==2) {
     ++oc2Waits; require(oc2Waits==1 && oc3Waits==0);
     for(unsigned j=0;j<64;++j) close(temp[j],std::exp(float(g[j])));
     if(fromFixpipe) std::memcpy(interUb,inter.data()+lane*4096,4096*4);
    } else {
     require(flag==3 && oc2Waits==1); ++oc3Waits; require(oc3Waits==1);
     for(unsigned j=0;j<4096;++j)
      close(interUb[j],inter[lane*4096+j]*std::exp(float(g[lane*32+j/128])));
     if(fromFixpipe) std::memcpy(r.ubBuf.memory.data()+U::o_intra_ping,intra.data()+lane*4096,4096*4);
    }
   };
   std::memset(AscendC::eventSets,0,sizeof(AscendC::eventSets));
   ov2(toL1?AscendC::GlobalTensor<D>{}:gm(out),fromFixpipe?AscendC::GlobalTensor<float>{}:gm(inter),
       fromFixpipe?AscendC::GlobalTensor<float>{}:gm(intra),0.125f,2,3,15);
   require(interCopies==(fromFixpipe?0:1) && intraCopies==(fromFixpipe?0:1));
   require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::MTE2_V)][6]==(fromFixpipe?0:2));
   require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::MTE3_V)][6]==1);
   checkL1(l1O,oBefore,64,128,lane,toL1);
   if(toL1) for(unsigned row=lane*32;row<(lane+1)*32;++row) for(unsigned col=0;col<128;++col) {
    unsigned i=row*128+col;require(float(out[i])==float(gmOBefore[i]));out[i]=l1O[zn(64,row,col)];
   }
   require(oc2Waits==1 && oc3Waits==1);
   AscendC::beforeCopy={}; AscendC::beforeCrossWait={};
   require(AscendC::gateCopies==1);
   require(AscendC::gateCasts==(std::is_same_v<G,float>?0:1));
   require(std::memcmp(maskBefore.data(),mask.p,4096)==0);
   require(std::memcmp(gateBefore.data(),gate.Tensor().p,256)==0);
  }
 }
 for(unsigned i=0;i<u.size();++i) {
  unsigned row=i/128;
  float value=float(u[i])-wh[i];
  close(vn[i],value);
  close(vu[i],value*std::exp(float(g[63])-float(g[row])));
  close(out[i],(inter[i]*std::exp(float(g[row]))+intra[i])*0.125f);
 }
}
template<class D,class G,class S> void hv2(bool store,bool initial,bool fromFixpipe,bool toL1) {
 constexpr unsigned n=128*128;
 std::vector<D> hOut(n,D(-9.f));
 std::vector<S> init(n),final(n,S(-7.f));
 std::vector<float> hRef(n),update(n);
 std::vector<G> g(64);
 std::array<Resource,2> resources;
 std::vector<D> l1H(n+32,D(-13.f));
 for(unsigned i=0;i<n;++i) {init[i]=S(std::sin(float(i)*0.002f));hRef[i]=initial?float(init[i]):0.f;}
 // Simulate driver initialization before the loop. Each AIV owns half H.
 for(unsigned lane=0;lane<2;++lane) {
  auto hUb=resources[lane].ubBuf.GetBufferByByte<float>(U::h_ping);
  for(unsigned j=0;j<8192;++j) {
   unsigned i=lane*8192+j;
   hUb.p[j]=hRef[i];
  }
 }
 for(unsigned chunk=0;chunk<3;++chunk) {
  bool last=chunk==2;
  for(unsigned j=0;j<64;++j) g[j]=G(-0.0078125f*(j+chunk));
  for(unsigned i=0;i<n;++i) update[i]=std::cos(float(i)*0.007f+chunk)*0.03125f;
  std::vector<float> next(n);
  for(unsigned i=0;i<n;++i) {
   float prev=hRef[i];
   next[i]=prev*std::exp(float(g[63]))+update[i];
  }
  for(unsigned lane=0;lane<2;++lane) {
   Resource& r=resources[lane]; AscendC::lane=lane;
   auto l1Before=l1H;
   unsigned workCopies=0;
   mode2_fixed64_detail::GateBuffer<G> gate(r); gate.Prepare(gm(g));
   GdnFusedMode2Update<D,G,S> hv(r,gate.Tensor(),fromFixpipe,toL1,AscendC::LocalTensor<D>{l1H.data()+16});
   std::memset(AscendC::eventSets,0,sizeof(AscendC::eventSets));
   unsigned hc2Waits=0,hReadySignals=0;
   auto hUb=r.ubBuf.GetBufferByByte<float>(U::h_ping).p;
   auto workBegin=reinterpret_cast<uintptr_t>(r.ubBuf.memory.data()+U::h_work_ping);
   AscendC::beforeLoad=[&](const void* src) {
    auto addr=reinterpret_cast<uintptr_t>(src);
    if(addr>=workBegin && addr<workBegin+8192*4) require(hc2Waits==1);
   };
   AscendC::beforeCopy=[&](const void* src) {
    if(src==update.data()+lane*8192) {require(!fromFixpipe && hc2Waits==1);++workCopies;}
   };
   AscendC::beforeCrossWait=[&](uint32_t flag) {
    require(flag==4); ++hc2Waits; require(hc2Waits==1);
    if(last&&!store) return;
    for(unsigned j=0;j<8192;++j) {
     unsigned i=lane*8192+j; float prev=hRef[i];
     close(hUb[j],prev*std::exp(float(g[63])));
    }
    if(fromFixpipe) std::memcpy(r.ubBuf.memory.data()+U::h_work_ping,update.data()+lane*8192,8192*4);
   };
   AscendC::beforeCrossSet=[&](uint32_t flag) {
    require(flag==14 && !last && hc2Waits==1);
    ++hReadySignals; require(hReadySignals==1);
    // Both destination routes must already contain this lane's H operand.
    for(unsigned j=0;j<8192;++j) {
     unsigned i=lane*8192+j;
     D value=toL1?l1H[zn(128,i/128,i%128)]:hOut[i];
     close(value,next[i]);
    }
    require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::V_MTE3)][5]==1);
   };
   hv(toL1?AscendC::GlobalTensor<D>{}:gm(hOut),gm(final),
      fromFixpipe?AscendC::GlobalTensor<float>{}:gm(update),4,14,last,store);
   require(hReadySignals==(last?0:1));
   AscendC::beforeCrossSet={};
   require(workCopies==((last&&!store)||fromFixpipe?0:1));
   checkL1(l1H,l1Before,128,128,lane,toL1&&!last);
   if(last) for(unsigned j=0;j<l1H.size();++j) require(float(l1H[j])==float(l1Before[j]));
   require(hc2Waits==1);
   AscendC::beforeCrossWait={}; AscendC::beforeLoad={}; AscendC::beforeCopy={};
   unsigned writes=last&&!store?0:1;
   unsigned reads=(last&&!store)||fromFixpipe?0:1;
   require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::MTE2_V)][5]==reads);
   require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::V_MTE3)][5]==writes);
   // No per-call local drain is required by the chunk dependency chain:
   // MTE3 H-ready -> HC1 -> HV1 -> HC2 -> next HV2.
   // Keep V_MTE3 and the producer notification checks above. This synchronous
   // stub does not prove real device ordering or the last-task reuse fence.
   for(unsigned j=0;j<64;++j) close(gate.Tensor().p[j],float(g[j]));
   maskUntouched(r);
  }
  if(last&&!store) {for(unsigned i=0;i<n;++i) close(final[i],-7.f);continue;}
  if(!last) {
   for(unsigned i=0;i<n;++i) {
    D value=toL1?l1H[zn(128,i/128,i%128)]:hOut[i]; close(value,next[i]);
    if(toL1) close(hOut[i],-9.f);
    // Cube receives DataT; Vec's recurrence must retain the full FP32 result.
    unsigned lane=i/8192,j=i%8192;
    float retained=resources[lane].ubBuf.GetBufferByByte<float>(U::h_ping).p[j];
    require(std::abs(retained-next[i])<1.e-6f);
    hRef[i]=next[i];
   }
  }
  if(last&&store) for(unsigned i=0;i<n;++i) close(final[i],next[i]);
  else for(unsigned i=0;i<n;++i) close(final[i],-7.f);
  // Outputs are not state inputs: corrupt them to catch any later reload.
  if(!last) {
   if(!toL1) std::fill(hOut.begin(),hOut.end(),D(5.f));
   else std::fill(l1H.begin()+16,l1H.begin()+16+n,D(5.f));
   std::fill(final.begin(),final.end(),S(-7.f));
  }
 }
}

template<class D,class G> void ov2Only(bool direct,bool toL1) {
 std::vector<G> g(64);
 std::vector<float> inter(64*128),intra(64*128);
 std::vector<D> out(64*128,D(-9.f)),l1(64*128+32,D(-13.f));
 for(unsigned i=0;i<inter.size();++i) {inter[i]=std::sin(i*0.04f);intra[i]=std::cos(i*0.02f);}
 for(unsigned chunk=0;chunk<3;++chunk) {
  for(unsigned i=0;i<64;++i) g[i]=G(-0.015625f*(i+chunk));
  for(unsigned lane=0;lane<2;++lane) {
   Resource r; AscendC::lane=lane;
   mode2_fixed64_detail::GateBuffer<G> gate(r);gate.Prepare(gm(g));
   GdnFusedMode2Output<D,G> op(r,gate.Tensor(),direct,toL1,{l1.data()+16});
   auto before=l1;auto gmBefore=out;
   unsigned waits=0,signals=0,copies=0;
   std::memset(AscendC::eventSets,0,sizeof(AscendC::eventSets));
   AscendC::beforeCrossWait=[&](uint32_t flag) {
    require(flag==(waits==0?2:3));++waits;
    if(direct) {
     auto dst=r.ubBuf.GetBufferByByte<float>(waits==1?U::o_inter_ping:U::o_intra_ping);
     std::memcpy(dst.p,(waits==1?inter.data():intra.data())+lane*4096,4096*4);
    }
   };
   AscendC::beforeCopy=[&](const void* src) {
    if(src==inter.data()+lane*4096) {require(!direct&&waits==1);++copies;}
    if(src==intra.data()+lane*4096) {require(!direct&&waits==2);++copies;}
   };
   AscendC::beforeCrossSet=[&](uint32_t flag) {
    require(flag==15 && waits==2);++signals;require(signals==1);
    require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::V_MTE3)][6]==1);
    for(unsigned row=lane*32;row<(lane+1)*32;++row) for(unsigned col=0;col<128;++col) {
     unsigned i=row*128+col;
     close(toL1?l1[zn(64,row,col)]:out[i],(inter[i]*std::exp(float(g[row]))+intra[i])*0.125f);
    }
   };
   op(toL1?AscendC::GlobalTensor<D>{}:gm(out),direct?AscendC::GlobalTensor<float>{}:gm(inter),
      direct?AscendC::GlobalTensor<float>{}:gm(intra),0.125f,2,3,15);
   require(waits==2&&signals==1&&copies==(direct?0:2));
   require(AscendC::eventSets[static_cast<unsigned>(AscendC::HardEvent::MTE2_V)][6]==(direct?0:2));
   checkL1(l1,before,64,128,lane,toL1);
   if(toL1) for(unsigned i=0;i<out.size();++i) require(float(out[i])==float(gmBefore[i]));
   AscendC::beforeCrossSet={};AscendC::beforeCrossWait={};AscendC::beforeCopy={};
  }
 }
}
template<class D,class G> void run() {
 #ifdef MODE2_OUTPUT_ONLY
 for(bool direct:{false,true}) for(bool l1:{false,true}) ov2Only<D,G>(direct,l1);
 #else
 #ifndef MODE2_UPDATE_ONLY
 for(bool direct:{false,true}) for(bool l1:{false,true}) hv1ov2<D,G>(direct,l1);
 #endif
 for(bool direct:{false,true}) for(bool l1:{false,true})
 for(bool store:{false,true}) for(bool initial:{false,true}) {
  hv2<D,G,float>(store,initial,direct,l1);
  hv2<D,G,D>(store,initial,direct,l1);
 }
 #endif
}
int main() {
 run<_Float16,float>();run<_Float16,_Float16>();run<_Float16,Bf16>();
 run<Bf16,float>();run<Bf16,_Float16>();run<Bf16,Bf16>();
 #if defined(MODE2_OUTPUT_ONLY)
 std::cout<<"PASS: OV2 only; 6 dtype combinations, 4 GM/Fixpipe and GM/L1 routes, 3 chunks, both AIV lanes; output and V_MTE3 checked at exactly one MTE3 completion notification\n";
 #else
 #if defined(MODE2_UPDATE_ONLY)
 const char* verified="HV2 only";
 #elif defined(MODE2_SKIP_QKMASK)
 const char* verified="HV1/HV2/OV2 (QK-mask skipped)";
 #else
 const char* verified="OV1/HV1/HV2/OV2";
 #endif
 std::cout<<"PASS: "<<verified<<" in synchronous CPU API emulation; 6 data/gate combinations, "
  <<"four Fixpipe/GM input and L1/GM output combinations across 3 chunks, "
  <<"one gate preparation per chunk, 192 three-chunk HV2 scenarios with resident FP32 H and poisoned GM outputs, "
  <<"driver-prepared initial state, H output checked at MTE3 HC1 notification, independent-computation-before-wait, both AIV lanes"
 #if !defined(MODE2_UPDATE_ONLY)
  <<", decay-complete/HC2-notify before Vnew-compute/OC3-notify, producer-read guards and resident-mask canaries"
 #endif
  <<"\n";
 #endif
}
'''
with tempfile.TemporaryDirectory(prefix='mode2-hv-ov-') as t:
 t=Path(t)
 (t/'catlass/arch').mkdir(parents=True)
 (t/'catlass/epilogue/block').mkdir(parents=True)
 (t/'catlass/epilogue/block/block_epilogue.hpp').write_text('#pragma once\nnamespace Catlass::Epilogue::Block {template<class... T> class BlockEpilogue;}\n')
 (t/'kernel_operator.h').write_text(stub)
 for p in ['catlass/catlass.hpp','catlass/arch/resource.hpp','catlass/arch/cross_core_sync.hpp']:
  (t/p).write_text('#pragma once\n#include "kernel_operator.h"\n')
 (t/'check.cpp').write_text(cpp)
 subprocess.run(['clang++',*(['-DMODE2_OUTPUT_ONLY'] if args.output_only else []),*(['-DMODE2_UPDATE_ONLY'] if args.update_only else []),*(['-DMODE2_SKIP_QKMASK'] if args.skip_qkmask else []),'-std=c++17','-O1','-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I',str(t),'-I',str(base),str(t/'check.cpp'),'-o',str(t/'check')],check=True)
 subprocess.run([str(t/'check')],check=True)

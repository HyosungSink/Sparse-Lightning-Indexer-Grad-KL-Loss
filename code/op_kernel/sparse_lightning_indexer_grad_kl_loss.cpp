#include "kernel_operator.h"
#include "sparse_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_key_sparse_lightning_indexer_grad_kl_loss.h"
using namespace AscendC;

__aicore__ inline float SliExp(float x) {
    if (x < -87.0f) return 0.0f;
    float z = x * 1.4426950408889634f;
    int n = static_cast<int>(z);
    if (z < static_cast<float>(n)) --n;
    float r = (x - n * 0.693145751953125f) - n * 1.428606765330187e-6f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.1666666666666667f +
        r * (0.0416666666666667f + r * (0.0083333333333333f + r *
        (0.0013888888888889f + r * (0.0001984126984127f + r * 0.0000248015873016f)))))));
    union { uint32_t u; float f; } power;
    power.u = static_cast<uint32_t>(n + 127) << 23;
    return p * power.f;
}
__aicore__ inline float SliLog(float x) {
    union { uint32_t u; float f; } bits;
    bits.f = x;
    int exponent = static_cast<int>(bits.u >> 23) - 127;
    bits.u = (bits.u & 0x7fffffU) | (127U << 23);
    float y = (bits.f - 1.0f) / (bits.f + 1.0f), y2 = y*y;
    float p = 1.0f/17.0f;
    p = 1.0f/15.0f + y2*p; p = 1.0f/13.0f + y2*p;
    p = 1.0f/11.0f + y2*p; p = 1.0f/9.0f + y2*p;
    p = 1.0f/7.0f + y2*p; p = 1.0f/5.0f + y2*p;
    p = 1.0f/3.0f + y2*p; p = 1.0f + y2*p;
    return exponent * 0.6931471805599453f + 2*y*p;
}
__aicore__ inline void SliSum(float value, float &sum, float &correction) {
    float y = value - correction;
    float t = sum + y;
    correction = (t - sum) - y;
    sum = t;
}

// A two-float expansion retains probability residuals through cancellation.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
struct SliPair {
    float hi, lo;
    __aicore__ inline SliPair(float h=0, float l=0) : hi(h), lo(l) {}
    __aicore__ inline float Value() const { return hi+lo; }
};
__aicore__ inline SliPair SliNormalize(float hi, float lo) {
    float sum=hi+lo;
    return SliPair(sum,lo-(sum-hi));
}
__aicore__ inline SliPair operator+(SliPair a, SliPair b) {
    float s=a.hi+b.hi, v=s-a.hi;
    float e=(a.hi-(s-v))+(b.hi-v);
    return SliNormalize(s, e+a.lo+b.lo);
}
__aicore__ inline SliPair operator-(SliPair a, SliPair b) {
    return a+SliPair(-b.hi,-b.lo);
}
__aicore__ inline SliPair operator*(SliPair a, SliPair b) {
    union {float f; uint32_t u;} aa,bb;
    aa.f=a.hi; bb.f=b.hi;
    aa.u &= 0xfffff000U; bb.u &= 0xfffff000U;
    float al=a.hi-aa.f, bl=b.hi-bb.f, p=a.hi*b.hi;
    float e=((aa.f*bb.f-p)+aa.f*bl+al*bb.f)+al*bl;
    return SliNormalize(p,e+a.hi*b.lo+a.lo*b.hi+a.lo*b.lo);
}
__aicore__ inline SliPair operator/(SliPair a, SliPair b) {
    float q=a.hi/b.hi;
    SliPair r=a-b*SliPair(q);
    float q2=(r.hi+r.lo)/b.hi;
    return SliNormalize(q,q2);
}
__aicore__ inline bool operator>(SliPair a, SliPair b) {
    return a.hi>b.hi || (a.hi==b.hi && a.lo>b.lo);
}
__aicore__ inline SliPair SliPairExp(SliPair x) {
    if(x.hi < -87.0f) return SliPair(0);
    if(x.hi==0 && x.lo==0) return SliPair(1);
    float z=x.hi*1.4426950408889634f;
    int n=static_cast<int>(z+(z>=0?0.5f:-0.5f));
    SliPair r=x-SliPair(0.693147182464599609375f,-1.904654323148236e-9f)*SliPair(static_cast<float>(n));
    // Centering the interval bounds the degree-12 Taylor remainder below 3e-16.
    SliPair sum=SliPair(2.08767558795841524e-09f,1.10828398092043424e-16f);
    sum=sum*r+SliPair(2.50521079436794025e-08f,4.41762304464836646e-16f);
    sum=sum*r+SliPair(2.75573199814971304e-07f,-7.57511220905119492e-15f);
    sum=sum*r+SliPair(2.75573188446287531e-06f,3.79357122429722915e-14f);
    sum=sum*r+SliPair(2.48015876422869042e-05f,-3.40699609366681977e-13f);
    sum=sum*r+SliPair(1.98412701138295233e-04f,-2.72559687493345582e-12f);
    sum=sum*r+SliPair(1.38888892251998186e-03f,-3.36310944371032150e-11f);
    sum=sum*r+SliPair(8.33333376795053482e-03f,-4.34617203337595015e-10f);
    sum=sum*r+SliPair(4.16666679084300995e-02f,-1.24176346982807218e-09f);
    sum=sum*r+SliPair(1.66666671633720398e-01f,-4.96705387931228870e-09f);
    sum=sum*r+SliPair(5.00000000000000000e-01f,0.00000000000000000e+00f);
    sum=sum*r+SliPair(1.00000000000000000e+00f,0.00000000000000000e+00f);
    sum=sum*r+SliPair(1.00000000000000000e+00f,0.00000000000000000e+00f);
    union {uint32_t u;float f;} power;
    power.u=static_cast<uint32_t>(n+127)<<23;
    return sum*SliPair(power.f);
}
__aicore__ inline SliPair SliPairLog(SliPair x) {
    float seed=SliLog(x.hi);
    SliPair e=SliPairExp(SliPair(seed));
    SliPair residual=(x-e)/e;
    return SliPair(seed)+residual-residual*residual*SliPair(0.5f);
}

template<class T, class W, class QR, class KR>
class SparseGrad {
    GlobalTensor<T> q, k, qi, ki, dq, dk;
    GlobalTensor<W> w, dw;
    GlobalTensor<QR> qr;
    GlobalTensor<KR> kr;
    GlobalTensor<int32_t> indices, ql32, kl32;
    GlobalTensor<int64_t> ql64, kl64;
    GlobalTensor<float> maxima, sums, scratch, allScratch, losses, loss;
    SparseLightningIndexerGradKlLossTilingData t;
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    LocalTensor<float> a, b, product, temporary, result, qiCache, dqCache, keyCache, dkCache, gradTemp, weightOut;
    LocalTensor<uint8_t> raw;
    LocalTensor<float> mainQuery, mainRope, mainKeys, mainKeyRopes;
    bool cachedMain;
    LocalTensor<float> pairMainScores;
    bool cachePairMain, pairMainReady, vectorPairMain, vectorPairExp, vectorPairGrad;
    LocalTensor<float> pairVectorWork;
    uint32_t stride;
    uint64_t simOffset, massOffset, logitOffset;

    template<class U>
    __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<U> src, uint64_t offset, uint32_t count) {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        if constexpr (AscendC::IsSameType<U,float>::value) {
            DataCopyPad(dst, src[offset], p, DataCopyPadExtParams<float>{false,0,0,0});
        } else {
            auto input = raw.ReinterpretCast<U>();
            DataCopyPad(input, src[offset], p, DataCopyPadExtParams<U>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            Cast(dst, input, RoundMode::CAST_NONE, count);
        }
        PipeBarrier<PIPE_ALL>();
    }
    template<class U>
    __aicore__ inline void Store(GlobalTensor<U> dst, uint64_t offset, LocalTensor<float> src, uint32_t count) {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        if constexpr (AscendC::IsSameType<U,float>::value) {
            DataCopyPad(dst[offset], src, p);
        } else {
            auto output = raw.ReinterpretCast<U>();
            Cast(output, src, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(dst[offset], output, p);
        }
        PipeBarrier<PIPE_ALL>();
    }
    template<class A, class B>
    __aicore__ inline float Dot(GlobalTensor<A> x, uint64_t xo, GlobalTensor<B> y, uint64_t yo, uint32_t count) {
        Load(a, x, xo, count); Load(b, y, yo, count);
        Mul(product, a, b, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(result, product, temporary, count);
        PipeBarrier<PIPE_ALL>();
        return result.GetValue(0);
    }
    __aicore__ inline float MainScore(uint32_t batch, uint32_t row, uint32_t head, uint32_t key) {
        uint64_t qo = (static_cast<uint64_t>(batch)*t.queryRows + row)*t.queryHeads + head;
        uint64_t ko = static_cast<uint64_t>(batch)*t.keyRows + key;
        return (Dot(q, qo*t.queryDim, k, ko*t.queryDim, t.queryDim) +
                Dot(qr, qo*t.ropeDim, kr, ko*t.ropeDim, t.ropeDim)) * t.scale;
    }
    template<class U>
    __aicore__ inline void LoadMatrix(LocalTensor<float> dst, GlobalTensor<U> src,
                                     uint64_t offset, uint32_t count) {
        for (uint32_t start=0; start<count; start+=1024) {
            uint32_t n=count-start<1024?count-start:1024;
            Load(dst[start],src,offset+start,n);
        }
    }
    __aicore__ inline void MainScores(uint32_t batch, uint32_t row, uint32_t key, float *scores) {
        if (!cachedMain) {
            for (uint32_t h=0; h<t.queryHeads; ++h) scores[h]=MainScore(batch,row,h,key);
            return;
        }
        uint8_t span=static_cast<uint8_t>(t.queryDim/8);
        Mul(product,mainQuery,mainKeys[key*t.queryDim],static_cast<uint64_t>(t.queryDim),
            static_cast<uint8_t>(t.queryHeads),{1,1,1,span,span,0});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(result,product,t.queryDim,t.queryHeads,1,1,span);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t h=0; h<t.queryHeads; ++h) scores[h]=result.GetValue(h);
        span=static_cast<uint8_t>(t.ropeDim/8);
        Mul(product,mainRope,mainKeyRopes[key*t.ropeDim],static_cast<uint64_t>(t.ropeDim),
            static_cast<uint8_t>(t.queryHeads),{1,1,1,span,span,0});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(result,product,t.ropeDim,t.queryHeads,1,1,span);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t h=0; h<t.queryHeads; ++h)
            scores[h]=(scores[h]+result.GetValue(h))*t.scale;
    }
    __aicore__ inline uint32_t Length(bool query, uint32_t batch) {
        uint32_t type = query ? t.queryLengthType : t.keyLengthType;
        uint32_t bound = query ? t.queryRows : t.keyRows;
        if (!type) return bound;
        int64_t value = type == 1 ? (query ? ql32.GetValue(batch) : kl32.GetValue(batch)) :
                                   (query ? ql64.GetValue(batch) : kl64.GetValue(batch));
        return value < 0 ? 0 : (value > bound ? bound : static_cast<uint32_t>(value));
    }
    __aicore__ inline bool Valid(int32_t key, uint32_t row, uint32_t qlen, uint32_t klen) {
        return key >= 0 && key < static_cast<int32_t>(klen) &&
               key <= static_cast<int64_t>(row) + klen - qlen;
    }
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
        GM_ADDR weights, GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
        GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR actualQ, GM_ADDR actualK,
        GM_ADDR dQuery, GM_ADDR dKey, GM_ADDR dWeight, GM_ADDR outputLoss, GM_ADDR workspace,
        const SparseLightningIndexerGradKlLossTilingData &tiling) {
        t = tiling;
        q.SetGlobalBuffer((__gm__ T*)query); k.SetGlobalBuffer((__gm__ T*)key);
        qi.SetGlobalBuffer((__gm__ T*)queryIndex); ki.SetGlobalBuffer((__gm__ T*)keyIndex);
        w.SetGlobalBuffer((__gm__ W*)weights); indices.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
        qr.SetGlobalBuffer((__gm__ QR*)queryRope); kr.SetGlobalBuffer((__gm__ KR*)keyRope);
        maxima.SetGlobalBuffer((__gm__ float*)softmaxMax); sums.SetGlobalBuffer((__gm__ float*)softmaxSum);
        ql32.SetGlobalBuffer((__gm__ int32_t*)actualQ); ql64.SetGlobalBuffer((__gm__ int64_t*)actualQ);
        kl32.SetGlobalBuffer((__gm__ int32_t*)actualK); kl64.SetGlobalBuffer((__gm__ int64_t*)actualK);
        dq.SetGlobalBuffer((__gm__ T*)dQuery); dk.SetGlobalBuffer((__gm__ T*)dKey);
        dw.SetGlobalBuffer((__gm__ W*)dWeight); loss.SetGlobalBuffer((__gm__ float*)outputLoss);
        scratch.SetGlobalBuffer((__gm__ float*)workspace + GetBlockIdx()*t.batchScratch);
        allScratch.SetGlobalBuffer((__gm__ float*)workspace);
        losses.SetGlobalBuffer((__gm__ float*)workspace + t.cores*t.batchScratch);
        stride = (t.indexDim + 7)/8*8;
        uint32_t cacheSize = t.indexHeads*stride;
        cachedMain = !AscendC::IsSameType<T,float>::value &&
            t.queryDim<=64 && t.ropeDim<=64 && t.queryDim%8==0 && t.ropeDim%8==0 &&
            t.queryHeads*t.queryDim<=1024 && t.queryHeads*t.ropeDim<=256 &&
            static_cast<uint64_t>(t.keyRows)*t.queryDim<=4096 &&
            static_cast<uint64_t>(t.keyRows)*t.ropeDim<=1024;
        cachePairMain = AscendC::IsSameType<T,float>::value &&
            static_cast<uint64_t>(t.keyRows)*t.queryHeads <= 4096;
        pairMainReady = false;
        uint32_t pairCacheSize = cachePairMain ? (2*t.keyRows*t.queryHeads+7)/8*8 : 0;
        vectorPairMain = cachePairMain && t.queryDim>=8 && t.queryDim<=64 &&
            t.ropeDim>=8 && t.ropeDim<=64 && (t.queryDim&(t.queryDim-1))==0 &&
            (t.ropeDim&(t.ropeDim-1))==0 && t.queryHeads*t.queryDim<=1024 &&
            t.queryHeads*t.ropeDim<=256 && static_cast<uint64_t>(t.keyRows)*t.queryDim<=4096 &&
            static_cast<uint64_t>(t.keyRows)*t.ropeDim<=1024;
        vectorPairExp=vectorPairMain && t.keyRows*t.queryHeads<=1024 && cacheSize<=4096;
        vectorPairGrad=vectorPairExp && t.indexDim>=8 && t.indexDim<=64 && t.indexDim%8==0 &&
            t.indexHeads*t.indexDim<=1024;
        pipe.InitBuffer(storage, (2*cacheSize + 512*8 + 128*4 + 64 +
            (cachedMain?6400:0) + pairCacheSize + (vectorPairMain?15616:0) + (vectorPairExp?6144:0))*sizeof(float));
        auto mem = storage.Get<float>();
        a = mem; b = mem[512]; product = mem[1024]; temporary = mem[1536]; result = mem[2048];
        raw = mem[2560].ReinterpretCast<uint8_t>();
        qiCache = mem[3072]; dqCache = mem[3072+cacheSize];
        keyCache = mem[3072+2*cacheSize]; dkCache = keyCache[128];
        gradTemp = dkCache[128]; weightOut = gradTemp[128];
        if (cachedMain) {
            mainQuery=weightOut[128]; mainRope=mainQuery[1024];
            mainKeys=mainRope[256]; mainKeyRopes=mainKeys[4096];
        }
        if (cachePairMain) pairMainScores=weightOut[128];
        if (vectorPairMain) {
            mainQuery=pairMainScores[pairCacheSize]; mainRope=mainQuery[1024];
            mainKeys=mainRope[256]; mainKeyRopes=mainKeys[4096];
            pairVectorWork=mainKeyRopes[1024];
        }
        simOffset = (static_cast<uint64_t>(t.keyRows)*t.indexDim+15)/16*16;
        massOffset = simOffset + static_cast<uint64_t>(t.sparseSize)*t.indexHeads;
        logitOffset = massOffset + t.sparseSize;
    }
    __aicore__ inline SliPair PairAt(uint64_t offset) {
        return SliPair(scratch.GetValue(offset),scratch.GetValue(offset+1));
    }
    __aicore__ inline void PutPair(uint64_t offset,SliPair value) {
        scratch.SetValue(offset,value.hi); scratch.SetValue(offset+1,value.lo);
    }
    template<class A,class B>
    __aicore__ inline SliPair PairDot(GlobalTensor<A> x,uint64_t xo,GlobalTensor<B> y,uint64_t yo,uint32_t count) {
        SliPair sum(0);
        for(uint32_t d=0;d<count;++d) {
            float av=static_cast<float>(x.GetValue(xo+d)), bv=static_cast<float>(y.GetValue(yo+d));
            if(av!=0 && bv!=0) sum=sum+SliPair(av)*SliPair(bv);
        }
        return sum;
    }
    __aicore__ inline void PairExpInput(uint32_t index,SliPair x) {
        SliPair r(0); float scale=0.0f;
        if(x.hi>=-87.0f) {
            float z=x.hi*1.4426950408889634f;
            int n=static_cast<int>(z+(z>=0?0.5f:-0.5f));
            r=n==0?x:x-SliPair(0.693147182464599609375f,-1.904654323148236e-9f)*SliPair(static_cast<float>(n));
            union {uint32_t u;float f;} power;
            power.u=static_cast<uint32_t>(n+127)<<23; scale=power.f;
        }
        pairVectorWork.SetValue(index,r.hi); pairVectorWork.SetValue(1024+index,r.lo);
        pairVectorWork.SetValue(14336+index,scale);
    }
    __aicore__ inline void PairPolyStep(uint32_t count,float ch,float cl) {
        auto r=pairVectorWork,rlow=pairVectorWork[1024];
        auto rh=pairVectorWork[2048],rt=pairVectorWork[3072];
        auto y=pairVectorWork[4096],ylow=pairVectorWork[5120];
        auto p=pairVectorWork[6144],e=pairVectorWork[7168];
        auto yh=pairVectorWork[8192],yt=pairVectorWork[9216];
        auto temp=pairVectorWork[10240],sum=pairVectorWork[11264],v=pairVectorWork[12288];
        ShiftRight(yh.ReinterpretCast<uint32_t>(),y.ReinterpretCast<uint32_t>(),12U,count); PipeBarrier<PIPE_V>();
        ShiftLeft(yh.ReinterpretCast<uint32_t>(),yh.ReinterpretCast<uint32_t>(),12U,count); PipeBarrier<PIPE_V>();
        Sub(yt,y,yh,count);
        Mul(p,y,r,count); Mul(e,yh,rh,count); PipeBarrier<PIPE_V>();
        Sub(e,e,p,count); Mul(temp,yh,rt,count); PipeBarrier<PIPE_V>();
        Add(e,e,temp,count); PipeBarrier<PIPE_V>();
        Mul(temp,yt,rh,count); PipeBarrier<PIPE_V>();
        Add(e,e,temp,count); PipeBarrier<PIPE_V>();
        Mul(temp,yt,rt,count); PipeBarrier<PIPE_V>();
        Add(e,e,temp,count); PipeBarrier<PIPE_V>();
        Mul(temp,y,rlow,count); PipeBarrier<PIPE_V>();
        Add(e,e,temp,count); PipeBarrier<PIPE_V>();
        Mul(temp,ylow,r,count); PipeBarrier<PIPE_V>();
        Add(e,e,temp,count); PipeBarrier<PIPE_V>();
        Mul(temp,ylow,rlow,count); PipeBarrier<PIPE_V>();
        Add(e,e,temp,count); Adds(sum,p,ch,count); PipeBarrier<PIPE_V>();
        Sub(v,sum,p,count); PipeBarrier<PIPE_V>();
        Sub(temp,sum,v,count); PipeBarrier<PIPE_V>();
        Sub(temp,p,temp,count); Muls(v,v,-1.0f,count); PipeBarrier<PIPE_V>();
        Adds(v,v,ch,count); PipeBarrier<PIPE_V>();
        Add(temp,temp,v,count); PipeBarrier<PIPE_V>();
        Add(e,e,temp,count); PipeBarrier<PIPE_V>();
        Adds(e,e,cl,count); PipeBarrier<PIPE_V>();
        Add(y,sum,e,count); PipeBarrier<PIPE_V>();
        Sub(v,y,sum,count); PipeBarrier<PIPE_V>();
        Sub(ylow,e,v,count); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PairExpBatch(uint32_t count) {
        auto r=pairVectorWork,rh=pairVectorWork[2048],rt=pairVectorWork[3072];
        auto y=pairVectorWork[4096],ylow=pairVectorWork[5120];
        PipeBarrier<PIPE_ALL>();
        ShiftRight(rh.ReinterpretCast<uint32_t>(),r.ReinterpretCast<uint32_t>(),12U,count); PipeBarrier<PIPE_V>();
        ShiftLeft(rh.ReinterpretCast<uint32_t>(),rh.ReinterpretCast<uint32_t>(),12U,count); PipeBarrier<PIPE_V>();
        Sub(rt,r,rh,count);
        Duplicate(y,2.08767558795841524e-09f,count);
        Duplicate(ylow,1.10828398092043424e-16f,count); PipeBarrier<PIPE_V>();
        PairPolyStep(count,2.50521079436794025e-08f,4.41762304464836646e-16f);
        PairPolyStep(count,2.75573199814971304e-07f,-7.57511220905119492e-15f);
        PairPolyStep(count,2.75573188446287531e-06f,3.79357122429722915e-14f);
        PairPolyStep(count,2.48015876422869042e-05f,-3.40699609366681977e-13f);
        PairPolyStep(count,1.98412701138295233e-04f,-2.72559687493345582e-12f);
        PairPolyStep(count,1.38888892251998186e-03f,-3.36310944371032150e-11f);
        PairPolyStep(count,8.33333376795053482e-03f,-4.34617203337595015e-10f);
        PairPolyStep(count,4.16666679084300995e-02f,-1.24176346982807218e-09f);
        PairPolyStep(count,1.66666671633720398e-01f,-4.96705387931228870e-09f);
        PairPolyStep(count,0.5f,0.0f);
        PairPolyStep(count,1.0f,0.0f);
        PairPolyStep(count,1.0f,0.0f);
        Mul(y,y,pairVectorWork[14336],count);
        Mul(ylow,ylow,pairVectorWork[14336],count); PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline SliPair PairExpAt(uint32_t index) {
        return SliPair(pairVectorWork.GetValue(4096+index),pairVectorWork.GetValue(5120+index));
    }
    __aicore__ inline void PairVectorDots(LocalTensor<float> query,LocalTensor<float> key,uint32_t dim) {
        auto hi=pairVectorWork,lo=pairVectorWork[1024];
        auto xh=pairVectorWork[2048],yh=pairVectorWork[3072];
        auto xl=pairVectorWork[4096],yl=pairVectorWork[5120];
        auto sum=pairVectorWork[6144],v=pairVectorWork[7168],e=pairVectorWork[8192];
        uint32_t count=t.queryHeads*dim;
        uint8_t span=static_cast<uint8_t>(dim/8),heads=static_cast<uint8_t>(t.queryHeads);
        BinaryRepeatParams repeated{1,1,1,span,span,0};
        ShiftRight(xh.ReinterpretCast<uint32_t>(),query.ReinterpretCast<uint32_t>(),12U,count);
        ShiftRight(yh.ReinterpretCast<uint32_t>(),key.ReinterpretCast<uint32_t>(),12U,dim);
        PipeBarrier<PIPE_V>();
        ShiftLeft(xh.ReinterpretCast<uint32_t>(),xh.ReinterpretCast<uint32_t>(),12U,count);
        ShiftLeft(yh.ReinterpretCast<uint32_t>(),yh.ReinterpretCast<uint32_t>(),12U,dim);
        PipeBarrier<PIPE_V>();
        Sub(xl,query,xh,count); Sub(yl,key,yh,dim);
        Mul(hi,query,key,static_cast<uint64_t>(dim),heads,repeated);
        Mul(lo,xh,yh,static_cast<uint64_t>(dim),heads,repeated);
        PipeBarrier<PIPE_V>();
        Sub(lo,lo,hi,count);
        Mul(sum,xh,yl,static_cast<uint64_t>(dim),heads,repeated); PipeBarrier<PIPE_V>();
        Add(lo,lo,sum,count); PipeBarrier<PIPE_V>();
        Mul(sum,xl,yh,static_cast<uint64_t>(dim),heads,repeated); PipeBarrier<PIPE_V>();
        Add(lo,lo,sum,count); PipeBarrier<PIPE_V>();
        Mul(sum,xl,yl,static_cast<uint64_t>(dim),heads,repeated); PipeBarrier<PIPE_V>();
        Add(lo,lo,sum,count); PipeBarrier<PIPE_V>();
        // Error-free pairwise summation; packed even/odd lanes stay within each head.
        for(uint32_t width=dim;width>1;width/=2) {
            uint64_t selected=0;
            GatherMask(xh,hi,static_cast<uint8_t>(1),true,count,{1,1,8,0},selected);
            GatherMask(yh,hi,static_cast<uint8_t>(2),true,count,{1,1,8,0},selected);
            GatherMask(xl,lo,static_cast<uint8_t>(1),true,count,{1,1,8,0},selected);
            GatherMask(yl,lo,static_cast<uint8_t>(2),true,count,{1,1,8,0},selected);
            count/=2; PipeBarrier<PIPE_V>();
            Add(sum,xh,yh,count); PipeBarrier<PIPE_V>();
            Sub(v,sum,xh,count); PipeBarrier<PIPE_V>();
            Sub(e,sum,v,count); PipeBarrier<PIPE_V>();
            Sub(e,xh,e,count); Sub(v,yh,v,count); PipeBarrier<PIPE_V>();
            Add(e,e,v,count); PipeBarrier<PIPE_V>();
            Add(e,e,xl,count); PipeBarrier<PIPE_V>();
            Add(e,e,yl,count); PipeBarrier<PIPE_V>();
            Add(hi,sum,e,count); PipeBarrier<PIPE_V>();
            Sub(v,hi,sum,count); PipeBarrier<PIPE_V>();
            Sub(lo,e,v,count); PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void PreparePairMain(uint32_t batch,uint32_t row,int64_t visible) {
        uint64_t token=static_cast<uint64_t>(batch)*t.queryRows+row;
        LoadMatrix(mainQuery,q,token*t.queryHeads*t.queryDim,t.queryHeads*t.queryDim);
        LoadMatrix(mainRope,qr,token*t.queryHeads*t.ropeDim,t.queryHeads*t.ropeDim);
        for(int64_t key=0;key<visible;++key) {
            PairVectorDots(mainQuery,mainKeys[key*t.queryDim],t.queryDim);
            SliPair scores[128];
            for(uint32_t h=0;h<t.queryHeads;++h)
                scores[h]=SliPair(pairVectorWork.GetValue(h),pairVectorWork.GetValue(1024+h));
            PairVectorDots(mainRope,mainKeyRopes[key*t.ropeDim],t.ropeDim);
            for(uint32_t h=0;h<t.queryHeads;++h) {
                SliPair value=(scores[h]+SliPair(pairVectorWork.GetValue(h),pairVectorWork.GetValue(1024+h)))*SliPair(t.scale);
                uint32_t offset=2*(h*t.keyRows+key);
                pairMainScores.SetValue(offset,value.hi); pairMainScores.SetValue(offset+1,value.lo);
            }
        }
    }
    __aicore__ inline SliPair PairMainRaw(uint32_t batch,uint32_t row,uint32_t head,uint32_t key) {
        uint64_t qo=(static_cast<uint64_t>(batch)*t.queryRows+row)*t.queryHeads+head;
        uint64_t ko=static_cast<uint64_t>(batch)*t.keyRows+key;
        return (PairDot(q,qo*t.queryDim,k,ko*t.queryDim,t.queryDim)+
                PairDot(qr,qo*t.ropeDim,kr,ko*t.ropeDim,t.ropeDim))*SliPair(t.scale);
    }
    __aicore__ inline SliPair PairMain(uint32_t batch,uint32_t row,uint32_t head,uint32_t key) {
        if (pairMainReady) {
            uint32_t offset=2*(head*t.keyRows+key);
            return SliPair(pairMainScores.GetValue(offset),pairMainScores.GetValue(offset+1));
        }
        return PairMainRaw(batch,row,head,key);
    }
    __aicore__ inline void PairVectorAdd(LocalTensor<float> dh,LocalTensor<float> dl,
        LocalTensor<float> ah,LocalTensor<float> al,LocalTensor<float> bh,LocalTensor<float> bl,uint32_t count) {
        auto sum=pairVectorWork[12288],v=pairVectorWork[13312],e=pairVectorWork[14336];
        Add(sum,ah,bh,count); PipeBarrier<PIPE_V>();
        Sub(v,sum,ah,count); PipeBarrier<PIPE_V>();
        Sub(e,sum,v,count); PipeBarrier<PIPE_V>();
        Sub(e,ah,e,count); Sub(v,bh,v,count); PipeBarrier<PIPE_V>();
        Add(e,e,v,count); PipeBarrier<PIPE_V>();
        Add(e,e,al,count); PipeBarrier<PIPE_V>();
        Add(e,e,bl,count); PipeBarrier<PIPE_V>();
        Add(dh,sum,e,count); PipeBarrier<PIPE_V>();
        Sub(v,dh,sum,count); PipeBarrier<PIPE_V>();
        Sub(dl,e,v,count); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PairGradientProduct(bool queryInput) {
        auto hi=pairVectorWork[2048],lo=pairVectorWork[3072];
        auto coefficient=pairVectorWork[6144],coefficientLow=pairVectorWork[7168];
        auto xh=pairVectorWork[8192],xl=pairVectorWork[9216];
        auto yh=pairVectorWork[10240],yl=pairVectorWork[11264];
        auto temp=pairVectorWork[12288],sum=pairVectorWork[13312],v=pairVectorWork[14336];
        auto input=queryInput?qiCache:keyCache;
        uint32_t count=t.indexHeads*t.indexDim, inputCount=queryInput?count:t.indexDim;
        uint8_t span=static_cast<uint8_t>(t.indexDim/8),heads=static_cast<uint8_t>(t.indexHeads);
        BinaryRepeatParams repeated{1,1,1,span,span,static_cast<uint8_t>(queryInput?span:0)};
        ShiftRight(xh.ReinterpretCast<uint32_t>(),coefficient.ReinterpretCast<uint32_t>(),12U,count);
        ShiftRight(yh.ReinterpretCast<uint32_t>(),input.ReinterpretCast<uint32_t>(),12U,inputCount);
        PipeBarrier<PIPE_V>();
        ShiftLeft(xh.ReinterpretCast<uint32_t>(),xh.ReinterpretCast<uint32_t>(),12U,count);
        ShiftLeft(yh.ReinterpretCast<uint32_t>(),yh.ReinterpretCast<uint32_t>(),12U,inputCount);
        PipeBarrier<PIPE_V>();
        Sub(xl,coefficient,xh,count); Sub(yl,input,yh,inputCount);
        Mul(hi,coefficient,input,static_cast<uint64_t>(t.indexDim),heads,repeated);
        Mul(lo,xh,yh,static_cast<uint64_t>(t.indexDim),heads,repeated); PipeBarrier<PIPE_V>();
        Sub(lo,lo,hi,count);
        Mul(temp,xh,yl,static_cast<uint64_t>(t.indexDim),heads,repeated); PipeBarrier<PIPE_V>();
        Add(lo,lo,temp,count); PipeBarrier<PIPE_V>();
        Mul(temp,xl,yh,static_cast<uint64_t>(t.indexDim),heads,repeated); PipeBarrier<PIPE_V>();
        Add(lo,lo,temp,count); PipeBarrier<PIPE_V>();
        Mul(temp,xl,yl,static_cast<uint64_t>(t.indexDim),heads,repeated); PipeBarrier<PIPE_V>();
        Add(lo,lo,temp,count); PipeBarrier<PIPE_V>();
        Mul(temp,coefficientLow,input,static_cast<uint64_t>(t.indexDim),heads,repeated); PipeBarrier<PIPE_V>();
        Add(lo,lo,temp,count); PipeBarrier<PIPE_V>();
        Add(sum,hi,lo,count); PipeBarrier<PIPE_V>();
        Sub(v,sum,hi,count); PipeBarrier<PIPE_V>();
        Sub(lo,lo,v,count); Adds(hi,sum,0.0f,count); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PairGradientStep(uint32_t batch,uint32_t key) {
        uint32_t dim=t.indexDim, count=t.indexHeads*dim;
        auto qh=pairVectorWork,ql=pairVectorWork[1024];
        auto ph=pairVectorWork[2048],pl=pairVectorWork[3072];
        auto kh=pairVectorWork[4096],kl=pairVectorWork[5120];
        Load(keyCache,ki,(static_cast<uint64_t>(batch)*t.keyRows+key)*dim,dim);
        PairGradientProduct(false);
        PairVectorAdd(qh,ql,qh,ql,ph,pl,count);
        PairGradientProduct(true);
        Load(pairVectorWork[8192],scratch,2*static_cast<uint64_t>(key)*dim,2*dim);
        uint64_t selected=0;
        GatherMask(kh,pairVectorWork[8192],static_cast<uint8_t>(1),true,2*dim,{1,1,8,0},selected);
        GatherMask(kl,pairVectorWork[8192],static_cast<uint8_t>(2),true,2*dim,{1,1,8,0},selected);
        PipeBarrier<PIPE_V>();
        // Preserve the original head order, including repeated sparse indices.
        for(uint32_t h=0;h<t.indexHeads;++h)
            PairVectorAdd(kh,kl,kh,kl,ph[h*dim],pl[h*dim],dim);
        Brcb(pairVectorWork[8192],kh,static_cast<uint8_t>(dim/8),{1,8});
        Brcb(pairVectorWork[9216],kl,static_cast<uint8_t>(dim/8),{1,8});
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams interleaved{static_cast<uint16_t>(dim),4,0,4,0};
        DataCopyPad(scratch[2*static_cast<uint64_t>(key)*dim],pairVectorWork[8192],interleaved);
        DataCopyPad(scratch[2*static_cast<uint64_t>(key)*dim+1],pairVectorWork[9216],interleaved);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void FinishPairVector(uint32_t first,uint64_t offset,uint32_t count) {
        auto hi=pairVectorWork,lo=pairVectorWork[1024];
        auto ph=pairVectorWork[2048],pl=pairVectorWork[3072],packed=pairVectorWork[4096];
        Duplicate(hi,0.0f,count); Duplicate(lo,0.0f,count); PipeBarrier<PIPE_V>();
        for(uint32_t peer=0;peer<t.rowWorkers;++peer) {
            Load(packed,allScratch,(first+peer)*t.batchScratch+2*offset,2*count);
            uint64_t selected=0;
            GatherMask(ph,packed,static_cast<uint8_t>(1),true,2*count,{1,1,8,0},selected);
            GatherMask(pl,packed,static_cast<uint8_t>(2),true,2*count,{1,1,8,0},selected);
            PipeBarrier<PIPE_V>();
            PairVectorAdd(hi,lo,hi,lo,ph,pl,count);
        }
        Add(a,hi,lo,count); PipeBarrier<PIPE_V>();
    }
    template<bool Accurate>
    __aicore__ inline void FinishBatch(uint32_t batch, uint64_t keyCount) {
        uint32_t worker = GetBlockIdx() % t.rowWorkers;
        uint32_t first = GetBlockIdx() - worker;
        if (t.rowWorkers > 1) {
            DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(scratch);
            SyncAll();
        }
        // Workers own disjoint output ranges and reduce partials in a fixed order.
        for (uint64_t offset = worker * 512; offset < keyCount; offset += t.rowWorkers * 512) {
            uint32_t n = keyCount-offset < 512 ? keyCount-offset : 512;
            if constexpr (Accurate) {
                if(vectorPairGrad) FinishPairVector(first,offset,n);
                else for (uint32_t d = 0; d < n; ++d) {
                    SliPair value(0);
                    for (uint32_t peer = 0; peer < t.rowWorkers; ++peer) {
                        uint64_t position = (first+peer)*t.batchScratch + 2*(offset+d);
                        value = value + SliPair(allScratch.GetValue(position),allScratch.GetValue(position+1));
                    }
                    a.SetValue(d,value.Value());
                }
            } else {
                Load(a,allScratch,first*t.batchScratch+offset,n);
                for (uint32_t peer = 1; peer < t.rowWorkers; ++peer) {
                    Load(b,allScratch,(first+peer)*t.batchScratch+offset,n);
                    Add(a,a,b,n);
                    PipeBarrier<PIPE_V>();
                }
            }
            Store(dk,static_cast<uint64_t>(batch)*keyCount+offset,a,n);
        }
    }
    __aicore__ inline void ProcessAccurate() {
        uint64_t keyCount=static_cast<uint64_t>(t.keyRows)*t.indexDim;
        uint64_t qOffset=(2*keyCount+15)/16*16;
        uint64_t sOffset=qOffset+2*t.indexHeads*t.indexDim;
        uint64_t mOffset=sOffset+2*static_cast<uint64_t>(t.sparseSize)*t.indexHeads;
        uint64_t iOffset=mOffset+2*t.sparseSize;
        SliPair total(0);
        for(uint32_t batch=GetBlockIdx()/t.rowWorkers;batch<t.batch;batch+=t.cores/t.rowWorkers) {
            DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(scratch);
            Duplicate(a,0.0f,512);
            for(uint64_t offset=0;offset<2*keyCount;offset+=512) {
                uint32_t n=2*keyCount-offset<512?2*keyCount-offset:512;
                Store(scratch,offset,a,n);
            }
            uint32_t qlen=t.empty?0:Length(true,batch),klen=Length(false,batch);
            if(vectorPairMain && qlen) {
                LoadMatrix(mainKeys,k,static_cast<uint64_t>(batch)*t.keyRows*t.queryDim,t.keyRows*t.queryDim);
                LoadMatrix(mainKeyRopes,kr,static_cast<uint64_t>(batch)*t.keyRows*t.ropeDim,t.keyRows*t.ropeDim);
            }
            for(uint32_t row=GetBlockIdx()%t.rowWorkers;row<t.queryRows;row+=t.rowWorkers) {
                uint64_t token=static_cast<uint64_t>(batch)*t.queryRows+row;
                if(!vectorPairGrad)
                    for(uint32_t d=0;d<t.indexHeads*t.indexDim;++d) PutPair(qOffset+2*d,SliPair(0));
                Duplicate(weightOut,0.0f,64); PipeBarrier<PIPE_ALL>();
                if(row<qlen) {
                    SliPair headNorm[128], headScore[128], headDenom[128], gradW[64];
                    for(uint32_t h=0;h<t.indexHeads;++h) gradW[h]=SliPair(0);
                    int64_t visible=static_cast<int64_t>(row)+klen-qlen+1;
                    pairMainReady=false;
                    if (cachePairMain) {
                        // The same high-precision score feeds max, sum, and sparse mass.
                        if(vectorPairMain) PreparePairMain(batch,row,visible);
                        else for(uint32_t h=0;h<t.queryHeads;++h) {
                            for(int64_t key=0;key<visible;++key) {
                                SliPair value=PairMainRaw(batch,row,h,key);
                                uint32_t offset=2*(h*t.keyRows+key);
                                pairMainScores.SetValue(offset,value.hi);
                                pairMainScores.SetValue(offset+1,value.lo);
                            }
                        }
                        pairMainReady=true;
                    }
                    // FP32 compatibility inputs may carry rounded statistics. Reconstruct
                    // the visible normalizer with the same precision as the sparse scores.
                    for(uint32_t h=0;h<t.queryHeads;++h) {
                        SliPair mx(-3.4e38f),sm(0);
                        for(int64_t key=0;key<visible;++key) {
                            SliPair score=PairMain(batch,row,h,key);
                            if(score>mx) mx=score;
                        }
                        if(vectorPairExp) {
                            headNorm[h]=mx;
                            for(int64_t key=0;key<visible;++key)
                                PairExpInput(h*visible+key,PairMain(batch,row,h,key)-mx);
                        } else {
                            for(int64_t key=0;key<visible;++key)
                                sm=sm+SliPairExp(PairMain(batch,row,h,key)-mx);
                            headNorm[h]=mx+SliPairLog(sm);
                        }
                    }
                    if(vectorPairExp && visible>0) {
                        PairExpBatch(t.queryHeads*visible);
                        for(uint32_t h=0;h<t.queryHeads;++h) {
                            SliPair sm(0);
                            for(int64_t key=0;key<visible;++key) sm=sm+PairExpAt(h*visible+key);
                            headDenom[h]=sm;
                            headNorm[h]=headNorm[h]+SliPairLog(sm);
                        }
                    }
                    uint32_t validCount=0;
                    SliPair maxMass(-3.4e38f),maxLogit(-3.4e38f);
                    for(uint32_t j=0;j<t.sparseSize;++j) {
                        int32_t key=indices.GetValue(token*t.sparseSize+j);
                        if(!Valid(key,row,qlen,klen)) continue;
                        ++validCount;
                        SliPair mh(-3.4e38f),mass(0),logit(0);
                        SliPair logmass(0);
                        if(vectorPairExp) {
                            for(uint32_t h=0;h<t.queryHeads;++h)
                                mass=mass+PairExpAt(h*visible+key)/headDenom[h];
                        }
                        if(vectorPairExp && mass.hi>1e-20f) logmass=SliPairLog(mass);
                        else {
                            mass=SliPair(0);
                            for(uint32_t h=0;h<t.queryHeads;++h) {
                                headScore[h]=PairMain(batch,row,h,key)-headNorm[h];
                                if(headScore[h]>mh) mh=headScore[h];
                            }
                            for(uint32_t h=0;h<t.queryHeads;++h) mass=mass+SliPairExp(headScore[h]-mh);
                            logmass=mh+SliPairLog(mass);
                        }
                        PutPair(mOffset+2*j,logmass);
                        if(logmass>maxMass) maxMass=logmass;
                        for(uint32_t h=0;h<t.indexHeads;++h) {
                            SliPair similarity=PairDot(qi,(token*t.indexHeads+h)*t.indexDim,
                                ki,(static_cast<uint64_t>(batch)*t.keyRows+key)*t.indexDim,t.indexDim);
                            PutPair(sOffset+2*(static_cast<uint64_t>(j)*t.indexHeads+h),similarity);
                            if(similarity>SliPair(0)) logit=logit+SliPair(static_cast<float>(w.GetValue(token*t.indexHeads+h)))*similarity;
                        }
                        PutPair(iOffset+2*j,logit);
                        if(logit>maxLogit) maxLogit=logit;
                    }
                    if(vectorPairGrad) {
                        LoadMatrix(qiCache,qi,token*t.indexHeads*t.indexDim,t.indexHeads*t.indexDim);
                        Duplicate(pairVectorWork,0.0f,t.indexHeads*t.indexDim);
                        Duplicate(pairVectorWork[1024],0.0f,t.indexHeads*t.indexDim);
                        PipeBarrier<PIPE_ALL>();
                    }
                    if(validCount) {
                        SliPair mt(0),ip(0);
                        for(uint32_t j=0;j<t.sparseSize;++j) {
                            int32_t key=indices.GetValue(token*t.sparseSize+j);
                            if(!Valid(key,row,qlen,klen)) continue;
                            mt=mt+SliPairExp(PairAt(mOffset+2*j)-maxMass);
                            ip=ip+SliPairExp(PairAt(iOffset+2*j)-maxLogit);
                        }
                        SliPair logmt=SliPairLog(mt),logip=SliPairLog(ip);
                        for(uint32_t j=0;j<t.sparseSize;++j) {
                            int32_t key=indices.GetValue(token*t.sparseSize+j);
                            if(!Valid(key,row,qlen,klen)) continue;
                            SliPair tm=PairAt(mOffset+2*j)-maxMass, im=PairAt(iOffset+2*j)-maxLogit;
                            SliPair target=SliPairExp(tm)/mt,predicted=SliPairExp(im)/ip,di=predicted-target;
                            total=total+target*((tm-logmt)-(im-logip));
                            for(uint32_t h=0;h<t.indexHeads;++h) {
                                SliPair similarity=PairAt(sOffset+2*(static_cast<uint64_t>(j)*t.indexHeads+h));
                                if(vectorPairGrad) {
                                    SliPair ds(0);
                                    if(similarity>SliPair(0)) {
                                        gradW[h]=gradW[h]+di*similarity;
                                        ds=di*SliPair(static_cast<float>(w.GetValue(token*t.indexHeads+h)));
                                    }
                                    Duplicate(pairVectorWork[6144+h*t.indexDim],ds.hi,t.indexDim);
                                    Duplicate(pairVectorWork[7168+h*t.indexDim],ds.lo,t.indexDim);
                                    continue;
                                }
                                if(!(similarity>SliPair(0))) continue;
                                gradW[h]=gradW[h]+di*similarity;
                                SliPair ds=di*SliPair(static_cast<float>(w.GetValue(token*t.indexHeads+h)));
                                if(ds.hi==0 && ds.lo==0) continue;
                                for(uint32_t d=0;d<t.indexDim;++d) {
                                    uint64_t qoff=qOffset+2*(h*t.indexDim+d),koff=2*(static_cast<uint64_t>(key)*t.indexDim+d);
                                    PutPair(qoff,PairAt(qoff)+ds*SliPair(static_cast<float>(ki.GetValue(
                                        (static_cast<uint64_t>(batch)*t.keyRows+key)*t.indexDim+d))));
                                    PutPair(koff,PairAt(koff)+ds*SliPair(static_cast<float>(qi.GetValue(
                                        (token*t.indexHeads+h)*t.indexDim+d))));
                                }
                            }
                            if(vectorPairGrad) {
                                PipeBarrier<PIPE_ALL>();
                                PairGradientStep(batch,static_cast<uint32_t>(key));
                            }
                        }
                        for(uint32_t h=0;h<t.indexHeads;++h) weightOut.SetValue(h,gradW[h].Value());
                    }
                }
                if(vectorPairGrad) {
                    uint32_t count=t.indexHeads*t.indexDim;
                    if(row<qlen) Add(dqCache,pairVectorWork,pairVectorWork[1024],count);
                    else Duplicate(dqCache,0.0f,count);
                    PipeBarrier<PIPE_V>();
                    Store(dq,token*count,dqCache,count);
                } else for(uint32_t h=0;h<t.indexHeads;++h) {
                    for(uint32_t d=0;d<t.indexDim;++d) a.SetValue(d,PairAt(qOffset+2*(h*t.indexDim+d)).Value());
                    Store(dq,(token*t.indexHeads+h)*t.indexDim,a,t.indexDim);
                }
                Store(dw,token*t.indexHeads,weightOut,t.indexHeads);
            }
            FinishBatch<true>(batch,keyCount);
        }
        losses.SetValue(GetBlockIdx()*16,total.hi); losses.SetValue(GetBlockIdx()*16+1,total.lo);
        DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(losses);
        if(t.cores>1) SyncAll();
        if(GetBlockIdx()==0) {
            SliPair sum(0);
            for(uint32_t i=0;i<t.cores;++i) sum=sum+SliPair(losses.GetValue(i*16),losses.GetValue(i*16+1));
            loss.SetValue(0,sum.Value());
            DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(loss);
        }
    }
    __aicore__ inline void Process() {
        if constexpr (AscendC::IsSameType<T,float>::value) { ProcessAccurate(); return; }
        float totalLoss = 0, lossCorrection = 0;
        for (uint32_t batch = GetBlockIdx()/t.rowWorkers; batch < t.batch; batch += t.cores/t.rowWorkers) {
            uint64_t keyCount = static_cast<uint64_t>(t.keyRows)*t.indexDim;
            Duplicate(a, 0.0f, 512);
            for (uint64_t offset = 0; offset < keyCount; offset += 512) {
                uint32_t n = keyCount-offset < 512 ? keyCount-offset : 512;
                Store(scratch, offset, a, n);
            }
            uint32_t qlen = t.empty ? 0 : Length(true,batch), klen = Length(false,batch);
            if (cachedMain && qlen) {
                LoadMatrix(mainKeys,k,static_cast<uint64_t>(batch)*t.keyRows*t.queryDim,t.keyRows*t.queryDim);
                LoadMatrix(mainKeyRopes,kr,static_cast<uint64_t>(batch)*t.keyRows*t.ropeDim,t.keyRows*t.ropeDim);
            }
            for (uint32_t row = GetBlockIdx()%t.rowWorkers; row < t.queryRows; row += t.rowWorkers) {
                Duplicate(dqCache, 0.0f, t.indexHeads*stride);
                Duplicate(weightOut, 0.0f, 64);
                PipeBarrier<PIPE_ALL>();
                uint64_t token = static_cast<uint64_t>(batch)*t.queryRows+row;
                if (row < qlen) {
                    if (cachedMain) {
                        Load(mainQuery,q,token*t.queryHeads*t.queryDim,t.queryHeads*t.queryDim);
                        Load(mainRope,qr,token*t.queryHeads*t.ropeDim,t.queryHeads*t.ropeDim);
                    }
                    for (uint32_t h=0; h<t.indexHeads; ++h)
                        Load(qiCache[h*stride], qi, (token*t.indexHeads+h)*t.indexDim, t.indexDim);
                    float headMax[128], headSum[128], score[128], weight[64], dwSum[64], dwCorr[64];
                    for (uint32_t h=0; h<t.indexHeads; ++h) {
                        if constexpr (AscendC::IsSameType<W,bfloat16_t>::value)
                            weight[h] = AscendC::ToFloat(w.GetValue(token*t.indexHeads+h));
                        else weight[h] = static_cast<float>(w.GetValue(token*t.indexHeads+h));
                        dwSum[h]=0; dwCorr[h]=0;
                    }
                    for (uint32_t h=0; h<t.queryHeads; ++h) {
                        uint64_t so = static_cast<uint64_t>(batch)*t.queryRows*t.queryHeads +
                            (t.statsLayout ? h*t.queryRows+row : row*t.queryHeads+h);
                        headMax[h] = maxima.GetValue(so); headSum[h] = sums.GetValue(so);
                        if constexpr (AscendC::IsSameType<T,float>::value) {
                            float mx = -3.4e38f;
                            int64_t visible = static_cast<int64_t>(row)+klen-qlen+1;
                            for (int64_t key=0; key<visible; ++key) {
                                float v=MainScore(batch,row,h,key); if(v>mx) mx=v;
                            }
                            float sm=0, correction=0;
                            for (int64_t key=0; key<visible; ++key)
                                SliSum(SliExp(MainScore(batch,row,h,key)-mx),sm,correction);
                            headMax[h]=mx; headSum[h]=sm;
                        }
                    }
                    float maxMass=-3.4e38f, maxLogit=-3.4e38f;
                    uint32_t validCount=0;
                    for (uint32_t j=0; j<t.sparseSize; ++j) {
                        int32_t key=indices.GetValue(token*t.sparseSize+j);
                        if (!Valid(key,row,qlen,klen)) continue;
                        ++validCount;
                        float mh=-3.4e38f;
                        MainScores(batch,row,key,score);
                        for(uint32_t h=0;h<t.queryHeads;++h) {
                            score[h]=score[h]-headMax[h]-SliLog(headSum[h]);
                            if(score[h]>mh) mh=score[h];
                        }
                        float mass=0, mc=0;
                        for(uint32_t h=0;h<t.queryHeads;++h) SliSum(SliExp(score[h]-mh),mass,mc);
                        float logmass=mh+SliLog(mass);
                        scratch.SetValue(massOffset+j,logmass);
                        if(logmass>maxMass) maxMass=logmass;
                        Load(keyCache,ki,(static_cast<uint64_t>(batch)*t.keyRows+key)*t.indexDim,t.indexDim);
                        float logit=0, lc=0;
                        for(uint32_t h=0;h<t.indexHeads;++h) {
                            Mul(product,qiCache[h*stride],keyCache,t.indexDim);
                            PipeBarrier<PIPE_V>();
                            ReduceSum(result,product,temporary,t.indexDim);
                            PipeBarrier<PIPE_ALL>();
                            float similarity=result.GetValue(0);
                            scratch.SetValue(simOffset+static_cast<uint64_t>(j)*t.indexHeads+h,similarity);
                            SliSum(weight[h]*(similarity>0?similarity:0),logit,lc);
                        }
                        scratch.SetValue(logitOffset+j,logit);
                        if(logit>maxLogit) maxLogit=logit;
                    }
                    if(validCount) {
                        float mt=0, mtc=0, ip=0, ipc=0;
                        for(uint32_t j=0;j<t.sparseSize;++j) {
                            int32_t key=indices.GetValue(token*t.sparseSize+j);
                            if(!Valid(key,row,qlen,klen)) continue;
                            SliSum(SliExp(scratch.GetValue(massOffset+j)-maxMass),mt,mtc);
                            SliSum(SliExp(scratch.GetValue(logitOffset+j)-maxLogit),ip,ipc);
                        }
                        float logmt=SliLog(mt), logip=SliLog(ip);
                        for(uint32_t j=0;j<t.sparseSize;++j) {
                            int32_t key=indices.GetValue(token*t.sparseSize+j);
                            if(!Valid(key,row,qlen,klen)) continue;
                            float lt=(scratch.GetValue(massOffset+j)-maxMass)-logmt;
                            float lp=(scratch.GetValue(logitOffset+j)-maxLogit)-logip;
                            float target=SliExp(lt), predicted=SliExp(lp), di=predicted-target;
                            SliSum(target*(lt-lp),totalLoss,lossCorrection);
                            Load(keyCache,ki,(static_cast<uint64_t>(batch)*t.keyRows+key)*t.indexDim,t.indexDim);
                            Load(dkCache,scratch,static_cast<uint64_t>(key)*t.indexDim,t.indexDim);
                            for(uint32_t h=0;h<t.indexHeads;++h) {
                                float similarity=scratch.GetValue(simOffset+static_cast<uint64_t>(j)*t.indexHeads+h);
                                if(similarity<=0) continue;
                                SliSum(di*similarity,dwSum[h],dwCorr[h]);
                                float ds=di*weight[h];
                                Muls(gradTemp,keyCache,ds,t.indexDim); PipeBarrier<PIPE_V>();
                                Add(dqCache[h*stride],dqCache[h*stride],gradTemp,t.indexDim); PipeBarrier<PIPE_V>();
                                Muls(gradTemp,qiCache[h*stride],ds,t.indexDim); PipeBarrier<PIPE_V>();
                                Add(dkCache,dkCache,gradTemp,t.indexDim); PipeBarrier<PIPE_V>();
                            }
                            Store(scratch,static_cast<uint64_t>(key)*t.indexDim,dkCache,t.indexDim);
                        }
                        for(uint32_t h=0;h<t.indexHeads;++h) weightOut.SetValue(h,dwSum[h]);
                    }
                }
                for(uint32_t h=0;h<t.indexHeads;++h)
                    Store(dq,(token*t.indexHeads+h)*t.indexDim,dqCache[h*stride],t.indexDim);
                Store(dw,token*t.indexHeads,weightOut,t.indexHeads);
            }
            FinishBatch<false>(batch,keyCount);
        }
        losses.SetValue(GetBlockIdx()*16,totalLoss);
        DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(losses);
        if (t.cores > 1) SyncAll();
        if(GetBlockIdx()==0) {
            float sum=0, correction=0;
            for(uint32_t i=0;i<t.cores;++i) SliSum(losses.GetValue(i*16),sum,correction);
            loss.SetValue(0,sum);
            DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(loss);
        }
    }
};

// Each worker keeps a bounded working set in UB and owns a contiguous query range.
template<class T, class W, class QR, class KR>
class SparseCompact {
    SparseLightningIndexerGradKlLossTilingData t;
    TBuf<TPosition::VECCALC> storage;
    LocalTensor<float> q,k,qi,ki,w,mx,sm,qr,kr;
    LocalTensor<float> rq,rk,rqi,rki,rw,rqr,rkr,dq,dk,dw,outLoss,rdq,rdk,rdw;
    LocalTensor<float> product,reduction,ha,hb,headMax,headSum,weight,similarity,ds,broadcast,simStore;
    LocalTensor<float> mass,logits,target,predicted,di,logTarget,logPred,aux;
    LocalTensor<uint8_t> mask;
    LocalTensor<int32_t> indices;
    GlobalTensor<int32_t> ql32,kl32;
    GlobalTensor<int64_t> ql64,kl64;
    GM_ADDR dqAddr;
    GM_ADDR dkAddr;
    GM_ADDR dwAddr;
    GM_ADDR lossAddr;
    GM_ADDR workAddr;
    uint32_t qCount,kCount,qiCount,kiCount,wCount,statsCount,qrCount,krCount;

    template<class U>
    __aicore__ inline void CopyIn(LocalTensor<float> converted,LocalTensor<float> raw,GM_ADDR address,uint32_t count) {
        GlobalTensor<U> source; source.SetGlobalBuffer((__gm__ U*)address);
        DataCopyExtParams params{1,static_cast<uint32_t>(count*sizeof(U)),0,0,0};
        if constexpr (AscendC::IsSameType<U,float>::value)
            DataCopyPad(converted,source,params,DataCopyPadExtParams<float>{false,0,0,0});
        else DataCopyPad(raw.ReinterpretCast<U>(),source,params,DataCopyPadExtParams<U>{false,0,0,0});
    }
    template<class U>
    __aicore__ inline void Decode(LocalTensor<float> converted,LocalTensor<float> raw,uint32_t count) {
        if constexpr (!AscendC::IsSameType<U,float>::value)
            Cast(converted,raw.ReinterpretCast<U>(),RoundMode::CAST_NONE,count);
    }
    template<class U>
    __aicore__ inline void Encode(LocalTensor<float> raw,LocalTensor<float> source,uint32_t count) {
        if constexpr (!AscendC::IsSameType<U,float>::value)
            Cast(raw.ReinterpretCast<U>(),source,RoundMode::CAST_RINT,count);
    }
    template<class U>
    __aicore__ inline void CopyOut(GM_ADDR address,LocalTensor<float> converted,LocalTensor<float> raw,uint32_t count) {
        GlobalTensor<U> destination; destination.SetGlobalBuffer((__gm__ U*)address);
        DataCopyExtParams params{1,static_cast<uint32_t>(count*sizeof(U)),0,0,0};
        if constexpr (AscendC::IsSameType<U,float>::value) DataCopyPad(destination,converted,params);
        else DataCopyPad(destination,raw.ReinterpretCast<U>(),params);
    }
    __aicore__ inline uint32_t Length(bool query,uint32_t batch) {
        uint32_t type=query?t.queryLengthType:t.keyLengthType;
        uint32_t bound=query?t.queryRows:t.keyRows;
        if(!type) return bound;
        int64_t value=type==1?(query?ql32.GetValue(batch):kl32.GetValue(batch)):
                              (query?ql64.GetValue(batch):kl64.GetValue(batch));
        return value<0?0:(value>bound?bound:static_cast<uint32_t>(value));
    }
    __aicore__ inline bool Valid(int32_t key,uint32_t row,uint32_t qlen,uint32_t klen) {
        return key>=0 && key<static_cast<int32_t>(klen) && key<=static_cast<int64_t>(row)+klen-qlen;
    }
    __aicore__ inline void MainDot(uint32_t token,uint32_t batch,uint32_t key) {
        uint8_t span=static_cast<uint8_t>(t.queryDim/8);
        Mul(product,q[token*t.queryHeads*t.queryDim],k[(batch*t.keyRows+key)*t.queryDim],
            static_cast<uint64_t>(t.queryDim),static_cast<uint8_t>(t.queryHeads),{1,1,1,span,span,0});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(ha,product,t.queryDim,t.queryHeads,1,1,span);
        PipeBarrier<PIPE_V>();
        span=static_cast<uint8_t>(t.ropeDim/8);
        Mul(product,qr[token*t.queryHeads*t.ropeDim],kr[(batch*t.keyRows+key)*t.ropeDim],
            static_cast<uint64_t>(t.ropeDim),static_cast<uint8_t>(t.queryHeads),{1,1,1,span,span,0});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(hb,product,t.ropeDim,t.queryHeads,1,1,span);
        PipeBarrier<PIPE_V>();
        Add(ha,ha,hb,t.queryHeads); PipeBarrier<PIPE_V>();
        Muls(ha,ha,t.scale,t.queryHeads); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void StableTarget(uint32_t token,uint32_t batch,uint32_t row,uint32_t qlen,uint32_t klen) {
        Ln(headSum,headSum,t.queryHeads); PipeBarrier<PIPE_V>();
        Duplicate(mass,-3.4e38f,64); PipeBarrier<PIPE_V>();
        for(uint32_t j=0;j<t.sparseSize;++j) {
            int32_t key=indices.GetValue(token*t.sparseSize+j);
            if(!Valid(key,row,qlen,klen)) continue;
            MainDot(token,batch,key);
            Sub(ha,ha,headMax,t.queryHeads); PipeBarrier<PIPE_V>();
            Sub(ha,ha,headSum,t.queryHeads); PipeBarrier<PIPE_V>();
            WholeReduceMax(reduction,ha,t.queryHeads,1,1,1,8); PipeBarrier<PIPE_ALL>();
            float peak=reduction.GetValue(0);
            Adds(ha,ha,-peak,t.queryHeads); PipeBarrier<PIPE_V>();
            Exp(ha,ha,t.queryHeads); PipeBarrier<PIPE_V>();
            WholeReduceSum(reduction,ha,t.queryHeads,1,1,1,8); PipeBarrier<PIPE_ALL>();
            mass.SetValue(j,peak+SliLog(reduction.GetValue(0)));
        }
        PipeBarrier<PIPE_ALL>();
        WholeReduceMax(reduction,mass,t.sparseSize,1,1,1,8); PipeBarrier<PIPE_ALL>();
        float peak=reduction.GetValue(0);
        Adds(target,mass,-peak,t.sparseSize); PipeBarrier<PIPE_V>();
        Exp(target,target,t.sparseSize); PipeBarrier<PIPE_V>();
        WholeReduceSum(reduction,target,t.sparseSize,1,1,1,8); PipeBarrier<PIPE_ALL>();
        Muls(target,target,1.0f/reduction.GetValue(0),t.sparseSize); PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ inline void Init(GM_ADDR query,GM_ADDR key,GM_ADDR queryIndex,GM_ADDR keyIndex,
        GM_ADDR weights,GM_ADDR sparseIndices,GM_ADDR softmaxMax,GM_ADDR softmaxSum,
        GM_ADDR queryRope,GM_ADDR keyRope,GM_ADDR actualQ,GM_ADDR actualK,
        GM_ADDR dQuery,GM_ADDR dKey,GM_ADDR dWeight,GM_ADDR outputLoss,GM_ADDR workspace,
        const SparseLightningIndexerGradKlLossTilingData &tiling,TPipe *pipe) {
        t=tiling;
        // Maximum live UB allocation is 132096 bytes; Host bounds every tensor.
        pipe->InitBuffer(storage,33024*sizeof(float));
        auto mem=storage.Get<float>();
        q=mem; k=mem[4096]; qi=mem[8192]; ki=mem[10240]; w=mem[12288];
        mx=mem[12416]; sm=mem[12544]; qr=mem[12672]; kr=mem[13696];
        rq=mem[14720]; rk=mem[16768]; rqi=mem[18816]; rki=mem[19840]; rw=mem[20864];
        rqr=mem[20992]; rkr=mem[21504]; dq=mem[22016]; dk=mem[24064]; dw=mem[26112];
        outLoss=mem[26240]; rdq=mem[26248]; rdk=mem[27272]; rdw=mem[28296];
        product=mem[28424]; reduction=mem[29448]; ha=mem[30472]; hb=mem[30600];
        headMax=mem[30728]; headSum=mem[30856]; weight=mem[30984]; similarity=mem[31048];
        ds=mem[31112]; broadcast=mem[31176]; simStore=mem[31688]; mass=mem[32200];
        logits=mem[32264]; target=mem[32328]; predicted=mem[32392]; di=mem[32456];
        logTarget=mem[32520]; logPred=mem[32584]; aux=mem[32648];
        mask=mem[32776].ReinterpretCast<uint8_t>(); indices=mem[32840].ReinterpretCast<int32_t>();
        dqAddr=dQuery; dkAddr=dKey; dwAddr=dWeight; lossAddr=outputLoss; workAddr=workspace;
        ql32.SetGlobalBuffer((__gm__ int32_t*)actualQ); kl32.SetGlobalBuffer((__gm__ int32_t*)actualK);
        ql64.SetGlobalBuffer((__gm__ int64_t*)actualQ); kl64.SetGlobalBuffer((__gm__ int64_t*)actualK);
        GlobalTensor<float> output; output.SetGlobalBuffer((__gm__ float*)outputLoss);
        DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(output);
        uint32_t tokens=t.batch*t.queryRows,keys=t.batch*t.keyRows;
        qCount=tokens*t.queryHeads*t.queryDim; kCount=keys*t.queryDim;
        qiCount=tokens*t.indexHeads*t.indexDim; kiCount=keys*t.indexDim;
        wCount=tokens*t.indexHeads; statsCount=tokens*t.queryHeads;
        qrCount=tokens*t.queryHeads*t.ropeDim; krCount=keys*t.ropeDim;
        CopyIn<T>(q,rq,query,qCount); CopyIn<T>(k,rk,key,kCount);
        CopyIn<T>(qi,rqi,queryIndex,qiCount); CopyIn<T>(ki,rki,keyIndex,kiCount);
        CopyIn<W>(w,rw,weights,wCount); CopyIn<QR>(qr,rqr,queryRope,qrCount); CopyIn<KR>(kr,rkr,keyRope,krCount);
        CopyIn<float>(mx,mx,softmaxMax,statsCount); CopyIn<float>(sm,sm,softmaxSum,statsCount);
        GlobalTensor<int32_t> indexGm; indexGm.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
        DataCopyPad(indices,indexGm,DataCopyExtParams{1,tokens*t.sparseSize*4,0,0,0},
                    DataCopyPadExtParams<int32_t>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();
        Decode<T>(q,rq,qCount); Decode<T>(k,rk,kCount); Decode<T>(qi,rqi,qiCount); Decode<T>(ki,rki,kiCount);
        Decode<W>(w,rw,wCount); Decode<QR>(qr,rqr,qrCount); Decode<KR>(kr,rkr,krCount);
        Duplicate(dq,0.0f,qiCount); Duplicate(dk,0.0f,kiCount); Duplicate(dw,0.0f,wCount);
        Duplicate(weight,0.0f,64); Duplicate(similarity,0.0f,64);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Process() {
        float total=0,correction=0;
        uint32_t simStride=(t.indexHeads+7)/8*8;
        uint32_t tokens=t.batch*t.queryRows;
        uint32_t begin=tokens*GetBlockIdx()/t.cores,end=tokens*(GetBlockIdx()+1)/t.cores;
        for(uint32_t batch=0;batch<t.batch;++batch) {
            uint32_t qlen=Length(true,batch),klen=Length(false,batch);
            for(uint32_t row=0;row<qlen;++row) {
                uint32_t token=batch*t.queryRows+row,validCount=0;
                if(token<begin || token>=end) continue;
                int32_t firstKey=-1;
                bool distinctKeys=false;
                for(uint32_t h=0;h<t.queryHeads;++h) {
                    uint32_t offset=batch*t.queryRows*t.queryHeads+(t.statsLayout?h*t.queryRows+row:row*t.queryHeads+h);
                    headMax.SetValue(h,mx.GetValue(offset)); headSum.SetValue(h,sm.GetValue(offset));
                }
                for(uint32_t h=0;h<t.indexHeads;++h) weight.SetValue(h,w.GetValue(token*t.indexHeads+h));
                Duplicate(mass,0.0f,64); Duplicate(logits,-3.4e38f,64); PipeBarrier<PIPE_ALL>();
                for(uint32_t j=0;j<t.sparseSize;++j) {
                    int32_t key=indices.GetValue(token*t.sparseSize+j);
                    if(!Valid(key,row,qlen,klen)) continue;
                    ++validCount;
                    if(firstKey<0) firstKey=key;
                    else if(key!=firstKey) distinctKeys=true;
                    MainDot(token,batch,key);
                    Sub(ha,ha,headMax,t.queryHeads); PipeBarrier<PIPE_V>();
                    Exp(ha,ha,t.queryHeads); PipeBarrier<PIPE_V>();
                    Div(ha,ha,headSum,t.queryHeads); PipeBarrier<PIPE_V>();
                    WholeReduceSum(mass[j],ha,t.queryHeads,1,1,1,8); PipeBarrier<PIPE_V>();
                    uint8_t span=static_cast<uint8_t>(t.indexDim/8);
                    Mul(product,qi[token*t.indexHeads*t.indexDim],ki[(batch*t.keyRows+key)*t.indexDim],
                        static_cast<uint64_t>(t.indexDim),static_cast<uint8_t>(t.indexHeads),{1,1,1,span,span,0});
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum(similarity,product,t.indexDim,t.indexHeads,1,1,span); PipeBarrier<PIPE_V>();
                    Relu(similarity,similarity,64); PipeBarrier<PIPE_V>();
                    DataCopy(simStore[j*simStride],similarity,simStride); PipeBarrier<PIPE_V>();
                    Mul(ds,similarity,weight,64); PipeBarrier<PIPE_V>();
                    WholeReduceSum(logits[j],ds,t.indexHeads,1,1,1,8); PipeBarrier<PIPE_V>();
                }
                // Repeated copies of a single key give identical uniform distributions.
                if(!distinctKeys) continue;
                WholeReduceSum(reduction,mass,t.sparseSize,1,1,1,8);
                WholeReduceMax(reduction[8],logits,t.sparseSize,1,1,1,8); PipeBarrier<PIPE_ALL>();
                float massSum=reduction.GetValue(0),peak=reduction.GetValue(8);
                if(massSum>1e-20f) { Muls(target,mass,1.0f/massSum,t.sparseSize); PipeBarrier<PIPE_V>(); }
                else StableTarget(token,batch,row,qlen,klen);
                Adds(logPred,logits,-peak,t.sparseSize); PipeBarrier<PIPE_V>();
                Exp(predicted,logPred,t.sparseSize); PipeBarrier<PIPE_V>();
                WholeReduceSum(reduction,predicted,t.sparseSize,1,1,1,8); PipeBarrier<PIPE_ALL>();
                float normalizer=reduction.GetValue(0);
                Muls(predicted,predicted,1.0f/normalizer,t.sparseSize); PipeBarrier<PIPE_V>();
                Sub(di,predicted,target,t.sparseSize); PipeBarrier<PIPE_V>();
                CompareScalar(mask,target,0.0f,CMPMODE::GT,64); PipeBarrier<PIPE_V>();
                Select(logTarget,mask,target,1.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,t.sparseSize); PipeBarrier<PIPE_V>();
                Ln(logTarget,logTarget,t.sparseSize); PipeBarrier<PIPE_V>();
                Adds(logPred,logPred,-SliLog(normalizer),t.sparseSize); PipeBarrier<PIPE_V>();
                Sub(aux,logTarget,logPred,t.sparseSize); PipeBarrier<PIPE_V>();
                Mul(aux,aux,target,t.sparseSize); PipeBarrier<PIPE_V>();
                WholeReduceSum(reduction,aux,t.sparseSize,1,1,1,8); PipeBarrier<PIPE_ALL>();
                SliSum(reduction.GetValue(0),total,correction);
                Duplicate(ha,0.0f,64); PipeBarrier<PIPE_V>();
                for(uint32_t j=0;j<t.sparseSize;++j) {
                    int32_t key=indices.GetValue(token*t.sparseSize+j);
                    if(!Valid(key,row,qlen,klen)) continue;
                    float gradient=di.GetValue(j);
                    Muls(ds,weight,gradient,64); PipeBarrier<PIPE_V>();
                    CompareScalar(mask,simStore[j*simStride],0.0f,CMPMODE::GT,64); PipeBarrier<PIPE_V>();
                    Select(ds,mask,ds,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,t.indexHeads); PipeBarrier<PIPE_V>();
                    Brcb(broadcast,ds,static_cast<uint8_t>((t.indexHeads+7)/8),{1,8}); PipeBarrier<PIPE_V>();
                    uint8_t span=static_cast<uint8_t>(t.indexDim/8);
                    uint32_t qo=token*t.indexHeads*t.indexDim,ko=(batch*t.keyRows+key)*t.indexDim;
                    Mul(product,ki[ko],broadcast,static_cast<uint64_t>(t.indexDim),static_cast<uint8_t>(t.indexHeads),
                        {1,1,0,span,0,1}); PipeBarrier<PIPE_V>();
                    Add(dq[qo],dq[qo],product,t.indexHeads*t.indexDim); PipeBarrier<PIPE_V>();
                    Mul(product,qi[qo],broadcast,static_cast<uint64_t>(t.indexDim),static_cast<uint8_t>(t.indexHeads),
                        {1,1,0,span,span,1}); PipeBarrier<PIPE_V>();
                    uint32_t heads=t.indexHeads;
                    while(heads>1) {
                        uint32_t half=(heads+1)/2;
                        Add(product,product,product[half*t.indexDim],(heads/2)*t.indexDim); PipeBarrier<PIPE_V>();
                        heads=half;
                    }
                    Add(dk[ko],dk[ko],product,t.indexDim); PipeBarrier<PIPE_V>();
                    Muls(hb,simStore[j*simStride],gradient,t.indexHeads); PipeBarrier<PIPE_V>();
                    Add(ha,ha,hb,t.indexHeads); PipeBarrier<PIPE_V>();
                }
                PipeBarrier<PIPE_ALL>();
                for(uint32_t h=0;h<t.indexHeads;++h) dw.SetValue(token*t.indexHeads+h,ha.GetValue(h));
            }
        }
        uint32_t localQi=(end-begin)*t.indexHeads*t.indexDim;
        uint32_t qiBegin=begin*t.indexHeads*t.indexDim;
        uint32_t localW=(end-begin)*t.indexHeads;
        for(uint32_t i=0;i<localW;++i) hb.SetValue(i,dw.GetValue(begin*t.indexHeads+i));
        outLoss.SetValue(0,total); PipeBarrier<PIPE_ALL>();
        Encode<T>(rdq,dq[qiBegin],localQi); Encode<W>(rdw,hb,localW);
        PipeBarrier<PIPE_ALL>();
        CopyOut<T>(dqAddr+qiBegin*sizeof(T),dq[qiBegin],rdq,localQi);
        CopyOut<W>(dwAddr+begin*t.indexHeads*sizeof(W),hb,rdw,localW);
        if(t.cores==1) {
            Encode<T>(rdk,dk,kiCount); PipeBarrier<PIPE_ALL>();
            CopyOut<T>(dkAddr,dk,rdk,kiCount); CopyOut<float>(lossAddr,outLoss,outLoss,1);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        uint64_t lossOffset=(kiCount+15)/16*16;
        uint64_t own=GetBlockIdx()*t.batchScratch;
        CopyOut<float>(workAddr+own*4,dk,dk,kiCount);
        CopyOut<float>(workAddr+(own+lossOffset)*4,outLoss,outLoss,1);
        SyncAll();
        GlobalTensor<float> partial; partial.SetGlobalBuffer((__gm__ float*)workAddr);
        uint32_t chunk=((kiCount+t.cores-1)/t.cores+7)/8*8;
        uint32_t keyBegin=GetBlockIdx()*chunk;
        if(keyBegin<kiCount) {
            uint32_t n=kiCount-keyBegin<chunk?kiCount-keyBegin:chunk;
            // One strided DMA gathers every worker's partial for this output range.
            DataCopyPad(product,partial[keyBegin],
                DataCopyExtParams{static_cast<uint16_t>(t.cores),n*4,static_cast<uint32_t>((t.batchScratch-n)*4),0,0},
                DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            for(uint32_t peer=1;peer<t.cores;++peer) {
                Add(product,product,product[peer*n],n); PipeBarrier<PIPE_V>();
            }
            Encode<T>(rdk,product,n); PipeBarrier<PIPE_ALL>();
            CopyOut<T>(dkAddr+keyBegin*sizeof(T),product,rdk,n);
        }
        if(GetBlockIdx()==0) {
            DataCopyPad(ha,partial[lossOffset],
                DataCopyExtParams{static_cast<uint16_t>(t.cores),4,static_cast<uint32_t>((t.batchScratch-1)*4),0,0},
                DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            float value=0,error=0;
            for(uint32_t peer=0;peer<t.cores;++peer) SliSum(ha.GetValue(peer*8),value,error);
            outLoss.SetValue(0,value); PipeBarrier<PIPE_ALL>();
            CopyOut<float>(lossAddr,outLoss,outLoss,1);
        }
        PipeBarrier<PIPE_ALL>();
    }
};

template<typename DT_QUERY, typename DT_WEIGHT, typename DT_QROPE, typename DT_KROPE>
__global__ __aicore__ void sparse_lightning_indexer_grad_kl_loss(
    GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex, GM_ADDR weights,
    GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR queryRope,
    GM_ADDR keyRope, GM_ADDR actualQ, GM_ADDR actualK, GM_ADDR dQuery, GM_ADDR dKey,
    GM_ADDR dWeight, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(SparseLightningIndexerGradKlLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseLightningIndexerGradKlLossTilingData,t,tiling);
    if constexpr (AscendC::IsSameType<DT_QUERY,half>::value) {
        if (t.compact) {
            TPipe compactPipe;
            SparseCompact<DT_QUERY,DT_WEIGHT,DT_QROPE,DT_KROPE> compactOp;
            compactOp.Init(query,key,queryIndex,keyIndex,weights,sparseIndices,softmaxMax,softmaxSum,
                queryRope,keyRope,actualQ,actualK,dQuery,dKey,dWeight,loss,GetUserWorkspace(workspace),t,&compactPipe);
            compactOp.Process();
            return;
        }
    }
    SparseGrad<DT_QUERY,DT_WEIGHT,DT_QROPE,DT_KROPE> op;
    op.Init(query,key,queryIndex,keyIndex,weights,sparseIndices,softmaxMax,softmaxSum,
        queryRope,keyRope,actualQ,actualK,dQuery,dKey,dWeight,loss,GetUserWorkspace(workspace),t);
    op.Process();
}

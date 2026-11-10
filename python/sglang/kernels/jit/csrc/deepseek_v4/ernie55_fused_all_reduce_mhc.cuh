// EB5 fused all-reduce + mHC: post residual, one fused W r' + Gram reduction,
// split mix, RMSNorm and g32 fp8 quant. x is already all-reduced, or (kFuseAR)
// this rank's attention partial, all-reduced in-kernel over the LL push plane;
// the MoE side (kHasShared) folds the shared-expert output into x first. Grid
// (16, ceil(T / kTPC)), cluster 16 along x; CTA c owns columns [256c, 256c +
// 256), one per thread of warps 0..7; warp 8 runs the comb sinkhorn. Semantics:
// test_internal/layers/_ll_mhc_ref.py fused_ar_mhc_ref_gram.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../../../../kernels/jit/csrc/distributed/ll_all_reduce_push.cuh"
#include "ernie55_g32_fp8_quant.cuh"

namespace sglang {
namespace eb5 {
namespace fused_ar_mhc {

constexpr int kHidden = 4096;
constexpr int kHc = 4;
constexpr int kHc3 = kHc * 2 + kHc * kHc;
// p[0..23] = W r', p[24..33] = Gram upper triangle G00 G01 G02 G03 G11 G12 G13
// G22 G23 G33.
constexpr int kNumP = kHc3 + 10;
// p padded to whole float4s for the v4 cluster exchange; p[34], p[35] are 0.
constexpr int kNumPPad = 36;
constexpr int kPV = kNumPPad / 4;
static_assert(kNumPPad % 4 == 0 && kNumPPad >= kNumP);
constexpr int kCS = 16;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
// Plus one warp that runs the comb sinkhorn beside phases 3 and 4.
constexpr int kBlockThreads = kThreads + 32;
constexpr int kPC = kHc + kHc * kHc;
// Push all-reduce (kFuseAR): TP world and 16B vectors per CTA token row.
constexpr int kWorld = 4;
constexpr int kRowVecs = kThreads / 8;
static_assert(kCS * kThreads == kHidden);

// LL push plane (push_protocol.txt); slot(dst, src, ph) = ws[dst] + src * SB +
// ph * kWorld * SB, payload token-major [T, 4096] bf16 like AllReducePushImpl.
struct PushArgs {
  uint8_t *ws[kWorld];
  device::distributed::Counter *counter;
  int64_t slot_bytes;
  int tp_rank;
  int num_blocks;
};

// kProf stamp ring (profiling builds only). Entry e = launch seq % ring_len is
// [kMaxClusters][kCS] CTAs x kProfWords u64: [0, 16) globaltimer ns and
// [16, 32) clock64 cycles of stamps S0..S12 = 0..12 (tid 0) and W0/W1 = 13/14
// (sinkhorn lane), slot 15 = the meta word (prof_meta) and seq; [32, 48)
// extras: kProfExtra + w = warp w's clock64 at the first CTA barrier, then the
// sinkhorn warp's kProfResetDone / kProfTotReady clock64 and the caller's
// kProfTag. seq [kMaxClusters * kCS] counts launches per CTA slot; every
// launch bumps all of them once, so they stay equal at kernel boundaries.
constexpr int kMaxClusters = 7;
constexpr int kProfStamps = 16;
constexpr int kProfWords = 48;
constexpr int kProfMeta = 15;
constexpr int kProfW0 = 13;
constexpr int kProfW1 = 14;
constexpr int kProfExtra = 32;
constexpr int kProfResetDone = kProfExtra + kWarps + 1;
constexpr int kProfTotReady = kProfResetDone + 1;
constexpr int kProfTag = kProfTotReady + 1;
// token_scalars sub-phases of warp 0 (clock64): rms, sigmoid, rn, broadcast.
constexpr int kProfScalRms = kProfTag + 1;
constexpr int kProfScalSig = kProfScalRms + 1;
constexpr int kProfScalRn = kProfScalSig + 1;
constexpr int kProfScalBcast = kProfScalRn + 1;
static_assert(kProfScalBcast < kProfWords);
struct ProfArgs {
  uint64_t *ring;
  uint32_t *seq;
  int ring_len;
  int64_t tag;
};

// Fields used only by the kFuseAR variant go last so the other variant keeps
// its layout and SASS.
struct Params {
  const bf16_t *__restrict__ x;
  const bf16_t *__restrict__ residual;
  const float *__restrict__ post;
  const float *__restrict__ comb;
  const float *__restrict__ fn;
  const float *__restrict__ hc_scale;
  const float *__restrict__ hc_base;
  const bf16_t *__restrict__ norm_weight;
  bf16_t *__restrict__ residual_out;
  bf16_t *__restrict__ layer_out;
  float *__restrict__ post_out;
  // comb_out [T, 16], or the comb logits M when write_logits.
  float *__restrict__ mix_out;
  uint8_t *__restrict__ q_out;
  uint8_t *__restrict__ sf_out;
  int num_tokens;
  int sinkhorn_repeat;
  int write_logits;
  float rms_eps;
  float hc_pre_eps;
  float hc_sinkhorn_eps;
  float hc_post_mult;
  float norm_eps;
  PushArgs push;
  // [T, 4096] bf16 under kHasShared, else null.
  const bf16_t *__restrict__ shared;
  ProfArgs prof;
};

template <int kTPC> struct Smem {
  alignas(16) bf16_t xs[kTPC][kThreads];
  alignas(16) bf16_t rs[kTPC][kHc][kThreads];
  alignas(16) float pc[kTPC][kPC];
  alignas(16) float red[kWarps][kTPC][kNumPPad];
  alignas(16) float cl[kCS][kTPC][kNumPPad];
  float tot[kTPC][kNumP];
  float hcs[3];
  float hcb[kHc3];
  alignas(8) uint64_t bar;
  // tot ready for the sinkhorn warp, which skips the compute warps' barriers.
  uint64_t tot_bar;
  // Shared-expert slice; filled only under kHasShared.
  alignas(16) bf16_t ss[kTPC][kThreads];
};

__device__ __forceinline__ uint32_t smem_u32(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(out)
               : "r"(addr), "r"(rank));
  return out;
}
__device__ __forceinline__ void st_async_v4(uint32_t addr, float4 v,
                                            uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
               "[%0], {%1, %2, %3, %4}, [%5];" ::"r"(addr),
               "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void mbar_init_expect(uint32_t bar,
                                                 uint32_t tx_bytes) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
      "r"(tx_bytes)
      : "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive_release(uint32_t bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(bar)
               : "memory");
}
__device__ __forceinline__ void mbar_wait_phase0(uint32_t bar) {
  asm volatile("{\n"
               ".reg .pred P;\n"
               "WAIT_%=:\n"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, "
               "[%0], 0;\n"
               "@!P bra WAIT_%=;\n"
               "}\n" ::"r"(bar)
               : "memory");
}
__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
}
// Barrier of the kThreads compute threads; the sinkhorn warp skips it.
__device__ __forceinline__ void compute_sync() {
  asm volatile("bar.sync 1, %0;" ::"n"(kThreads) : "memory");
}

// kProf helpers. The "memory" clobbers keep the stamps in program order with
// the loads and stores around them.
__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}
__device__ __forceinline__ uint64_t clock64_cycles() {
  uint64_t c;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)::"memory");
  return c;
}
__device__ __forceinline__ uint32_t sm_id() {
  uint32_t s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
// Gives v a real consumer in SASS, so a later stamp issues only once the load
// that produced v has returned. The store never fires for finite inputs.
__device__ __forceinline__ void prof_consume(float v, uint64_t *sink) {
  asm volatile("{\n.reg .pred p;\nsetp.eq.f32 p, %0, 0f7F7FFFFF;\n"
               "@p st.global.u32 [%1], 0;\n}" ::"f"(v),
               "l"(sink)
               : "memory");
}
// kProf stamps are staged in shared memory past Smem and flushed to the ring
// entry at exit; stamps slot 15 holds the meta word (ns) and seq (cycles).
struct ProfStage {
  uint64_t st[kProfWords];
};
__device__ __forceinline__ void prof_stamp(ProfStage &ps, int s) {
  const uint64_t ns = globaltimer_ns();
  ps.st[s] = ns;
  ps.st[kProfStamps + s] = clock64_cycles();
}
// tid 0 at entry: S0, and zeroes only the words no other thread writes before
// the first CTA barrier (S4 is off the non-push path, W0/W1 run only in CTAs
// with a sinkhorn token, the reset stamp only under kFuseAR, the tail is
// unused); the per-warp words are written unconditionally, so no race.
__device__ __forceinline__ void prof_open(ProfStage &ps) {
  for (const int s : {4, kProfW0, kProfW1}) {
    ps.st[s] = 0;
    ps.st[kProfStamps + s] = 0;
  }
  for (int k = kProfResetDone; k < kProfWords; ++k)
    if (k != kProfTotReady)
      ps.st[k] = 0;
  prof_stamp(ps, 0);
}
__device__ __forceinline__ void prof_cycles(ProfStage &ps, int k) {
  ps.st[k] = clock64_cycles();
}
template <int kN>
__device__ __forceinline__ void prof_consume_weights(const float (&w)[kN][kHc],
                                                     float wn, uint64_t *sink) {
  // Every weight load of this thread must land before S1.
  float acc = wn;
#pragma unroll
  for (int m = 0; m < kN; ++m)
#pragma unroll
    for (int j = 0; j < kHc; ++j)
      acc = acc + w[m][j];
  prof_consume(acc, sink);
}
// Exit, the sinkhorn warp after a CTA barrier: this CTA's staged stamps to
// ring entry seq % ring_len, then seq + 1 into its counter, and from CTA 0 into
// the counters past the grid (the push bumper's rule), all equal at boundaries.
__device__ __forceinline__ void prof_flush(const ProfArgs &a,
                                           const ProfStage &ps, int blk,
                                           int lane, uint32_t seq,
                                           uint64_t meta) {
  const int64_t entry = seq % static_cast<uint32_t>(a.ring_len);
  uint64_t *slot = a.ring + (entry * (kMaxClusters * kCS) + blk) * kProfWords;
  for (int k = lane; k < kProfWords; k += 32)
    slot[k] = k == kProfMeta                 ? meta
              : k == kProfStamps + kProfMeta ? uint64_t{seq}
              : k == kProfTag                ? static_cast<uint64_t>(a.tag)
                                             : ps.st[k];
  if (lane == 0)
    a.seq[blk] = seq + 1;
  if (blk == 0)
    for (int i = kCS * static_cast<int>(gridDim.y) + lane;
         i < kMaxClusters * kCS; i += 32)
      a.seq[i] = seq + 1;
}
// Meta word: T | grid.y << 8 | nt << 16 | kind << 24 | smid << 32, kind =
// kFuseAR | kHasShared << 1 | kEarlyTrigger << 2 | kCombLogits << 3 |
// kTPC << 4.
template <int kTPC, bool kFuseAR, bool kHasShared, bool kEarlyTrigger,
          bool kCombLogits>
__device__ __forceinline__ uint64_t prof_meta(int nt, int num_tokens) {
  constexpr uint64_t kKind = (kFuseAR ? 1 : 0) | (kHasShared ? 2 : 0) |
                             (kEarlyTrigger ? 4 : 0) | (kCombLogits ? 8 : 0) |
                             (uint64_t{kTPC} << 4);
  return (uint64_t{sm_id()} << 32) | (kKind << 24) | (uint64_t(nt) << 16) |
         (uint64_t{gridDim.y} << 8) | uint64_t(num_tokens);
}
// Named barrier between the push warps (arrive) and the loader warps (sync).
constexpr int kPushBarrier = 2;
__device__ __forceinline__ void named_arrive(int id, int count) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void named_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ float sigmoid_tl(float z) {
  return 1.0f / (1.0f + expf(0.0f - z));
}
template <bool kAlignFleet>
__device__ __forceinline__ float mix_rms(float sqrsum, float eps) {
  constexpr float kN = static_cast<float>(kHc * kHidden);
  if constexpr (kAlignFleet)
    return 1.0f / (sqrtf(sqrsum / kN) + eps);
  else
    return rsqrtf(sqrsum / kN + eps);
}

__device__ __forceinline__ float rcp_approx(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Ref (c0 + c2) + (c1 + c3) row sums, one approximate reciprocal per sum, then
// multiplies; the first row pass adds eps after the scaling, later ones before.
// _rn intrinsics forbid fma contraction, so every kernel inlining this rounds
// identically.
template <bool kFirst>
__device__ __forceinline__ void sinkhorn_row_pass(float (&c)[16], float eps) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float s = __fadd_rn(__fadd_rn(c[i * 4], c[i * 4 + 2]),
                              __fadd_rn(c[i * 4 + 1], c[i * 4 + 3]));
    const float inv = rcp_approx(kFirst ? s : __fadd_rn(s, eps));
#pragma unroll
    for (int j = 0; j < 4; ++j)
      c[i * 4 + j] =
          kFirst ? fmaf(c[i * 4 + j], inv, eps) : __fmul_rn(c[i * 4 + j], inv);
  }
}

__device__ __forceinline__ void sinkhorn_col_pass(float (&c)[16], float eps) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float s =
        __fadd_rn(__fadd_rn(c[j], c[8 + j]), __fadd_rn(c[4 + j], c[12 + j]));
    const float inv = rcp_approx(__fadd_rn(s, eps));
#pragma unroll
    for (int i = 0; i < 4; ++i)
      c[i * 4 + j] = __fmul_rn(c[i * 4 + j], inv);
  }
}

template <bool kAlignFleet>
__device__ __forceinline__ void sinkhorn_exp(float (&c)[16]) {
  constexpr float kLog2e = 1.4426950408889634f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float rmax =
        fmaxf(fmaxf(c[i * 4], c[i * 4 + 1]), fmaxf(c[i * 4 + 2], c[i * 4 + 3]));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if constexpr (kAlignFleet)
        c[i * 4 + j] = exp2f((c[i * 4 + j] - rmax) * kLog2e);
      else
        c[i * 4 + j] = expf(c[i * 4 + j] - rmax);
    }
  }
}

/// \brief Scalar sinkhorn16_fast: the arithmetic the f32x2 version must match
///        bit for bit (SinkhornCompareTest); not used by the kernels.
template <bool kAlignFleet>
__device__ __forceinline__ void sinkhorn16_scalar(float (&c)[16], float eps,
                                                  int repeat) {
  sinkhorn_exp<kAlignFleet>(c);
  sinkhorn_row_pass<true>(c, eps);
  sinkhorn_col_pass(c, eps);
  for (int iter = 1; iter < repeat; ++iter) {
    sinkhorn_row_pass<false>(c, eps);
    sinkhorn_col_pass(c, eps);
  }
}

// f32x2 passes on v[2i] = (c[i][0], c[i][1]), v[2i + 1] = (c[i][2], c[i][3]).
// Each lane does the scalar pass's operation on the same operands: row sum
// (c0 + c2) + (c1 + c3), column sum (r0 + r2) + (r1 + r3), then + eps.
template <bool kFirst>
__device__ __forceinline__ void sinkhorn_row_pass2(float2 (&v)[8], float eps) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 a = __fadd2_rn(v[2 * i], v[2 * i + 1]);
    const float s = __fadd_rn(a.x, a.y);
    const float inv = rcp_approx(kFirst ? s : __fadd_rn(s, eps));
    const float2 inv2 = make_float2(inv, inv);
#pragma unroll
    for (int h = 0; h < 2; ++h)
      v[2 * i + h] = kFirst
                         ? __ffma2_rn(v[2 * i + h], inv2, make_float2(eps, eps))
                         : __fmul2_rn(v[2 * i + h], inv2);
  }
}

__device__ __forceinline__ void sinkhorn_col_pass2(float2 (&v)[8], float eps) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float2 s =
        __fadd2_rn(__fadd2_rn(v[h], v[4 + h]), __fadd2_rn(v[2 + h], v[6 + h]));
    const float2 se = __fadd2_rn(s, make_float2(eps, eps));
    const float2 inv = make_float2(rcp_approx(se.x), rcp_approx(se.y));
#pragma unroll
    for (int i = 0; i < 4; ++i)
      v[2 * i + h] = __fmul2_rn(v[2 * i + h], inv);
  }
}

/// \brief One-thread sinkhorn of a [4, 4] logit matrix, in place: the ref up to
///        the approximate reciprocals. Shared by the main kernel and
///        fused_ar_mhc_sinkhorn_kernel, so both paths give bit-identical combs;
///        bit-identical to sinkhorn16_scalar.
template <bool kAlignFleet>
__device__ __forceinline__ void sinkhorn16_fast(float (&c)[16], float eps,
                                                int repeat) {
  sinkhorn_exp<kAlignFleet>(c);
  float2 v[8];
#pragma unroll
  for (int k = 0; k < 8; ++k)
    v[k] = make_float2(c[2 * k], c[2 * k + 1]);
  sinkhorn_row_pass2<true>(v, eps);
  sinkhorn_col_pass2(v, eps);
  // Not unrolled: the loop body is the whole cost, the extra copies only code.
#pragma unroll 1
  for (int iter = 1; iter < repeat; ++iter) {
    sinkhorn_row_pass2<false>(v, eps);
    sinkhorn_col_pass2(v, eps);
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    c[2 * k] = v[k].x;
    c[2 * k + 1] = v[k].y;
  }
}

/// \brief Bit-equality test of sinkhorn16_fast vs sinkhorn16_scalar: logits
///        [N, 16] f32 in, fast / scalar [N, 16] f32 out.
template <bool kAlignFleet>
__global__ void sinkhorn_compare_kernel(const float *__restrict__ logits,
                                        float *__restrict__ fast,
                                        float *__restrict__ scalar, int n,
                                        float eps, int repeat) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n)
    return;
  float a[16], b[16];
#pragma unroll
  for (int k = 0; k < 16; ++k)
    a[k] = b[k] = logits[t * 16 + k];
  sinkhorn16_fast<kAlignFleet>(a, eps, repeat);
  sinkhorn16_scalar<kAlignFleet>(b, eps, repeat);
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    fast[t * 16 + k] = a[k];
    scalar[t * 16 + k] = b[k];
  }
}

template <bool kAlignFleet> struct SinkhornCompareTest {
  static void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView fast,
                  tvm::ffi::TensorView scalar, double eps, int64_t repeat) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto N = SymbolicSize{"n"};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, int64_t{16}})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(logits)
        .verify(fast)
        .verify(scalar);
    const auto n = static_cast<int>(N.unwrap());
    if (n == 0)
      return;
    LaunchKernel(static_cast<uint32_t>((n + 255) / 256), 256, device.unwrap())(
        sinkhorn_compare_kernel<kAlignFleet>,
        static_cast<const float *>(logits.data_ptr()),
        static_cast<float *>(fast.data_ptr()),
        static_cast<float *>(scalar.data_ptr()), n, static_cast<float>(eps),
        static_cast<int>(repeat));
  }
};

// One reduce-scatter level: lanes with `hi` keep in[kOut..), the others
// in[0..kOut); in[] past kIn is a zero pad.
template <int kIn, int kMask>
__device__ __forceinline__ void rs_step(const float (&in)[kIn],
                                        float (&out)[(kIn + 1) / 2], int lane) {
  constexpr int kOut = (kIn + 1) / 2;
  const bool hi = (lane & kMask) != 0;
#pragma unroll
  for (int k = 0; k < kOut; ++k) {
    const float lo_v = in[k];
    const float hi_v = kOut + k < kIn ? in[kOut + k] : 0.0f;
    const float send = hi ? lo_v : hi_v;
    const float keep = hi ? hi_v : lo_v;
    out[k] = keep + __shfl_xor_sync(0xffffffffu, send, kMask);
  }
}

/// \brief Warp reduce-scatter of p[34] (36 shuffles): lane keeps 2 sums.
/// \param idx Receives the p index of out[k], or -1 where out[k] is padding.
__device__ __forceinline__ void warp_reduce_scatter34(const float (&p)[kNumP],
                                                      float (&out)[2],
                                                      int (&idx)[2], int lane) {
  static_assert(kNumP == 34);
  float a17[17], a9[9], a5[5], a3[3];
  rs_step<34, 16>(p, a17, lane);
  rs_step<17, 8>(a17, a9, lane);
  rs_step<9, 4>(a9, a5, lane);
  rs_step<5, 2>(a5, a3, lane);
  rs_step<3, 1>(a3, out, lane);
  const int b = (lane >> 4) & 1, c = (lane >> 3) & 1, d = (lane >> 2) & 1;
  const int e = (lane >> 1) & 1, f = lane & 1;
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int l4 = f * 2 + k;
    const int l3 = e * 3 + l4;
    const int l2 = d * 5 + l3;
    const int l1 = c * 9 + l2;
    const bool ok = l4 < 3 && l3 < 5 && l2 < 9 && l1 < 17;
    idx[k] = ok ? b * 17 + l1 : -1;
  }
}

// Reduce-scatter unit test: in [W * 32, 34] -> out [W, 34] warp sums.
__global__ void
fused_ar_mhc_reduce_scatter_test_kernel(const float *__restrict__ in,
                                        float *__restrict__ out) {
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  const int row = warp * 32 + lane;
  float p[kNumP];
#pragma unroll
  for (int v = 0; v < kNumP; ++v)
    p[v] = in[row * kNumP + v];
  float s[2];
  int idx[2];
  warp_reduce_scatter34(p, s, idx, lane);
#pragma unroll
  for (int k = 0; k < 2; ++k)
    if (idx[k] >= 0)
      out[warp * kNumP + idx[k]] = s[k];
}

struct ReduceScatterTest {
  static void run(tvm::ffi::TensorView in, tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto R = SymbolicSize{"rows"};
    auto W = SymbolicSize{"warps"};
    device.set_options<kDLCUDA>();
    TensorMatcher({R, int64_t{kNumP}})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(in);
    TensorMatcher({W, int64_t{kNumP}})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(out);
    RuntimeCheck(R.unwrap() == W.unwrap() * 32, "in must be [warps * 32, 34]");
    const auto warps = static_cast<uint32_t>(W.unwrap());
    LaunchKernel(warps, 32,
                 device.unwrap())(fused_ar_mhc_reduce_scatter_test_kernel,
                                  static_cast<const float *>(in.data_ptr()),
                                  static_cast<float *>(out.data_ptr()));
  }
};
// One 16B load per item over rows x, residual 0..3, shared. Without kWithX
// (push all-reduce) x comes from the poll.
// Items lt, lt + nthr, ... of the x / residual / shared rows and of post/comb.
// kCombLogits: comb comes from the sinkhorn warp, so only post is loaded.
template <int kTPC, bool kWithX, bool kHasShared, bool kCombLogits>
__device__ __forceinline__ void load_inputs(const Params &p, Smem<kTPC> &sm,
                                            int tok0, int nt, int col0, int lt,
                                            int nthr) {
  constexpr int kRow = kThreads / 8;
  constexpr int kFirst = kWithX ? 0 : 1;
  constexpr int kPerTok = kRow * (1 + kHc + (kHasShared ? 1 : 0) - kFirst);
  for (int i = lt; i < nt * kPerTok; i += nthr) {
    const int t = i / kPerTok, r = kFirst + (i % kPerTok) / kRow, k = i % kRow;
    const int64_t tok = tok0 + t;
    const bf16_t *src = r == 0     ? p.x + tok * kHidden
                        : r <= kHc ? p.residual + (tok * kHc + r - 1) * kHidden
                                   : p.shared + tok * kHidden;
    bf16_t *dst = r == 0 ? sm.xs[t] : r <= kHc ? sm.rs[t][r - 1] : sm.ss[t];
    reinterpret_cast<uint4 *>(dst)[k] =
        reinterpret_cast<const uint4 *>(src + col0)[k];
  }
  constexpr int kPCLoad = kCombLogits ? kHc : kPC;
  for (int i = lt; i < nt * kPCLoad; i += nthr) {
    const int t = i / kPCLoad, e = i % kPCLoad;
    const int tok = tok0 + t;
    sm.pc[t][e] =
        e < kHc ? p.post[tok * kHc + e] : p.comb[tok * kHc * kHc + (e - kHc)];
  }
}

// Weights of column h: fn[m][j][h] and the RMSNorm weight.
__device__ __forceinline__ void load_weights(const Params &p, int h,
                                             float (&w)[kHc3][kHc], float &wn) {
#pragma unroll
  for (int m = 0; m < kHc3; ++m)
#pragma unroll
    for (int j = 0; j < kHc; ++j)
      w[m][j] = __ldg(p.fn + (m * kHc + j) * kHidden + h);
  wn = __bfloat162float(p.norm_weight[h]);
}

// kCombLogits, sinkhorn warp: comb of this CTA's nt tokens from their logits
// (p.comb), the side kernel's sinkhorn16_fast, into sm.pc before the
// post-residual pass. Lane t takes token t.
template <int kTPC, bool kAlignFleet>
__device__ __forceinline__ void
comb_from_logits(const Params &p, Smem<kTPC> &sm, int tok0, int nt, int lane) {
  if (lane >= nt)
    return;
  const float *src = p.comb + (tok0 + lane) * kHc * kHc;
  float c[kHc * kHc];
#pragma unroll
  for (int k = 0; k < kHc * kHc; ++k)
    c[k] = src[k];
  sinkhorn16_fast<kAlignFleet>(c, p.hc_sinkhorn_eps, p.sinkhorn_repeat);
#pragma unroll
  for (int k = 0; k < kHc * kHc; ++k)
    sm.pc[lane][kHc + k] = c[k];
}

using push_vec_t = device::AlignedVector<bf16x2_t, 4>;

// 16B vector index of (token, column slice, k) in the token-major [T, 4096]
// payload, the AllReducePushImpl layout; producer and consumer agree on it.
__device__ __forceinline__ int64_t push_vid(int tok, int col0, int k) {
  return static_cast<int64_t>(tok) * (kHidden / 8) + col0 / 8 + k;
}

// This CTA's x_r slice to slot tp_rank of every rank (self included); +0
// lanes go out as -0 so they cannot be mistaken for the empty marker.
__device__ __forceinline__ void push_x(const Params &p, int tok0, int nt,
                                       int col0, int tid, int64_t phase_off) {
  if (tid >= nt * kRowVecs)
    return;
  const int64_t vid = push_vid(tok0 + tid / kRowVecs, col0, tid % kRowVecs);
  push_vec_t v;
  ld_global_16B(v, p.x, vid);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    clear_pos_zero(v[j].x);
    clear_pos_zero(v[j].y);
  }
  const int64_t src_off = p.push.tp_rank * p.push.slot_bytes + phase_off;
#pragma unroll
  for (int d = 0; d < kWorld; ++d)
    st_relaxed_16B(v, p.push.ws[d] + src_off, vid);
}

// Spin until all ranks' shares landed, then sum them exactly as
// AllReducePushImpl::poll_impl (rank order, fp32, one RN to bf16) into xs.
// Built without fast-math: the sum must not flush. The +0 markers are restored
// later by reset_slots, off the CTA barriers: a CTA barrier right behind those
// stores waits for them.
template <int kTPC>
__device__ __forceinline__ void poll_x(const Params &p, Smem<kTPC> &sm,
                                       int tok0, int nt, int col0, int tid,
                                       int64_t phase_off) {
  if (tid >= nt * kRowVecs)
    return;
  const int t = tid / kRowVecs, k = tid % kRowVecs;
  const int64_t vid = push_vid(tok0 + t, col0, k);
  const uint8_t *base = p.push.ws[p.push.tp_rank] + phase_off;
  push_vec_t v[kWorld];
  bool pending;
  do {
    pending = false;
#pragma unroll
    for (int s = 0; s < kWorld; ++s)
      ld_relaxed_16B(v[s], base + s * p.push.slot_bytes, vid);
#pragma unroll
    for (int s = 0; s < kWorld; ++s)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        pending |= is_pos_zero(v[s][j].x) || is_pos_zero(v[s][j].y);
  } while (pending);
  ::sglang::reduce(v).store(sm.xs[t], k);
}

// The sinkhorn warp, once the CTA barrier after poll_x is passed: +0 back into
// the kWorld slots of every vid this CTA polled, for the next same-phase round.
__device__ __forceinline__ void reset_slots(const Params &p, int tok0, int nt,
                                            int col0, int lane,
                                            int64_t phase_off) {
  push_vec_t zero;
  const bf16_t z = get_pos_zero<bf16_t>();
#pragma unroll
  for (int j = 0; j < 4; ++j)
    zero[j] = bf16x2_t{z, z};
  uint8_t *base = p.push.ws[p.push.tp_rank] + phase_off;
  for (int i = lane; i < nt * kRowVecs; i += 32) {
    const int64_t vid = push_vid(tok0 + i / kRowVecs, col0, i % kRowVecs);
#pragma unroll
    for (int s = 0; s < kWorld; ++s)
      st_global_16B(zero, base + s * p.push.slot_bytes, vid);
  }
}

// Gram entry G_jk from p[24..33] (upper triangle, row-major).
__device__ __forceinline__ float gram_at(const float *tot, int j, int k) {
  const int a = min(j, k), b = max(j, k);
  return tot[kHc3 + a * kHc - a * (a - 1) / 2 + (b - a)];
}

/// \brief Max |v| over the warp; NaN lanes are ignored like abs_max8.
__device__ __forceinline__ float warp_abs_max(float v) {
  float m = fabsf(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    m = g32_quant::detail::max_ftz(m, __shfl_xor_sync(0xffffffffu, m, o));
  return m;
}

/// \brief e4m3 byte of v * inv, same PTX as g32_quant::quant8_e4m3.
__device__ __forceinline__ uint8_t quant1_e4m3(float v, float inv) {
  const float s = g32_quant::detail::scale_clamp(v, inv);
  uint16_t packed;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
      : "=h"(packed)
      : "f"(0.0f), "f"(s));
  return static_cast<uint8_t>(packed & 0xffu);
}

template <int kTPC, bool kHasShared>
__device__ __forceinline__ void
post_residual_and_reduce(const Params &p, Smem<kTPC> &sm,
                         const float (&w)[kHc3][kHc], int t, int64_t tok, int h,
                         int tid, int lane, int warp, float (&rf)[kHc]) {
  const float xf = __bfloat162float(sm.xs[t][tid]);
  // mhc_fused_cluster's hidden_in at unit scales: one fp32 add, no bf16 round.
  const float hin =
      kHasShared ? __fadd_rn(xf, __bfloat162float(sm.ss[t][tid])) : xf;
  const float *pc = sm.pc[t];
  float r[kHc];
#pragma unroll
  for (int i = 0; i < kHc; ++i)
    r[i] = __bfloat162float(sm.rs[t][i][tid]);
  // f32x2 over output pairs: each lane runs the scalar op on the scalar
  // operands (_rn: no contraction), so every value keeps its bits.
  const auto pair = [](const float *v) {
    return *reinterpret_cast<const float2 *>(v);
  };
  const auto bc = [](float v) { return make_float2(v, v); };
#pragma unroll
  for (int j = 0; j < kHc; j += 2) {
    // Ref _post_residual contraction: fma(post, hin, comb_0j * r_0), then fma.
    float2 acc = __ffma2_rn(pair(pc + j), bc(hin),
                            __fmul2_rn(pair(pc + kHc + j), bc(r[0])));
#pragma unroll
    for (int i = 1; i < kHc; ++i)
      acc = __ffma2_rn(pair(pc + kHc + i * kHc + j), bc(r[i]), acc);
    const bf16_t rb[2] = {__float2bfloat16_rn(acc.x),
                          __float2bfloat16_rn(acc.y)};
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      p.residual_out[(tok * kHc + j + u) * kHidden + h] = rb[u];
      rf[j + u] = __bfloat162float(rb[u]);
    }
  }
  float pv[kNumP];
#pragma unroll
  for (int m = 0; m < kHc3; m += 2) {
    float2 a = __fmul2_rn(make_float2(w[m][0], w[m + 1][0]), bc(rf[0]));
#pragma unroll
    for (int j = 1; j < kHc; ++j)
      a = __ffma2_rn(make_float2(w[m][j], w[m + 1][j]), bc(rf[j]), a);
    pv[m] = a.x;
    pv[m + 1] = a.y;
  }
  // Gram upper triangle rf[j] * rf[k], j <= k, in p order, two per FMUL2.
  constexpr int kGj[10] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3};
  constexpr int kGk[10] = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
#pragma unroll
  for (int n = 0; n < 10; n += 2) {
    const float2 g = __fmul2_rn(make_float2(rf[kGj[n]], rf[kGj[n + 1]]),
                                make_float2(rf[kGk[n]], rf[kGk[n + 1]]));
    pv[kHc3 + n] = g.x;
    pv[kHc3 + n + 1] = g.y;
  }
  float s[2];
  int idx[2];
  warp_reduce_scatter34(pv, s, idx, lane);
#pragma unroll
  for (int k = 0; k < 2; ++k)
    if (idx[k] >= 0)
      sm.red[warp][t][idx[k]] = s[k];
}
// Mix rms of token t from the cluster totals.
template <int kTPC, bool kAlignFleet>
__device__ __forceinline__ float token_rms(const Params &p,
                                           const Smem<kTPC> &sm, int t) {
  const float *tot = sm.tot[t];
  const float zsq = (gram_at(tot, 0, 0) + gram_at(tot, 1, 1)) +
                    (gram_at(tot, 2, 2) + gram_at(tot, 3, 3));
  return mix_rms<kAlignFleet>(zsq, p.rms_eps);
}

// Comb logits M of token t: what logits_out receives and the sinkhorn consumes.
template <int kTPC>
__device__ __forceinline__ void comb_logits(const Smem<kTPC> &sm, int t,
                                            float rms, float (&c)[16]) {
#pragma unroll
  for (int k = 0; k < kHc * kHc; ++k)
    c[k] = fmaf(sm.tot[t][2 * kHc + k] * rms, sm.hcs[2], sm.hcb[2 * kHc + k]);
}

// Phase 3 in every compute warp, 8 lanes per token t (lanes past the last token
// mirror it): lane 8t + j computes pre_j and lane 8t + 4 + j post_j, one
// sigmoid each, by the formulas of the one-thread version; every lane then runs
// the |y|^2 = pre^T G pre fmaf chain (j outer, k inner) of its token on the
// gathered pre, so every output keeps its bits. pre / rn of every token reach
// every lane by shuffles: no smem round trip and no CTA barrier. With `store`
// (warp 0 of cluster rank 0) it also writes post_out and the comb logits.
template <int kTPC, bool kAlignFleet, bool kProf>
__device__ __forceinline__ void
token_scalars(const Params &p, const Smem<kTPC> &sm, int tok0, int nt, int lane,
              bool store, float (&pre_out)[kTPC][kHc], float (&rn_out)[kTPC],
              ProfStage *ps, bool stamps) {
  static_assert(kTPC * 8 <= 32, "one warp holds 8 lanes per token");
  // kProf sub-phase stamps of warp 0; the __syncwarp keeps the warp converged.
  const auto cyc = [&](int k) {
    if constexpr (kProf)
      if (stamps) {
        if (lane == 0)
          prof_cycles(*ps, k);
        __syncwarp();
      }
  };
  const int t = min(lane >> 3, nt - 1), q = lane & 7, j = q & 3;
  const bool is_post = q >= kHc;
  const float *tot = sm.tot[t];
  // Every smem operand first, so their latency overlaps the rms chain.
  const int e = is_post ? kHc + j : j;
  const float tot_e = tot[e];
  const float hcs = sm.hcs[is_post ? 1 : 0], hcb = sm.hcb[e];
  float g[kHc][kHc];
#pragma unroll
  for (int a = 0; a < kHc; ++a)
#pragma unroll
    for (int b = 0; b < kHc; ++b)
      g[a][b] = gram_at(tot, a, b);
  const float rms = token_rms<kTPC, kAlignFleet>(p, sm, t);
  cyc(kProfScalRms);
  const float s = sigmoid_tl(fmaf(tot_e * rms, hcs, hcb));
  const float pre_j = s + p.hc_pre_eps;
  cyc(kProfScalSig);
  float pre[kHc];
#pragma unroll
  for (int k = 0; k < kHc; ++k)
    pre[k] = __shfl_sync(0xffffffffu, pre_j, (lane & ~7) + k);
  float ysq = 0.0f;
#pragma unroll
  for (int a = 0; a < kHc; ++a)
#pragma unroll
    for (int b = 0; b < kHc; ++b)
      ysq = fmaf(pre[a] * g[a][b], pre[b], ysq);
  const float rn = rsqrtf(ysq / static_cast<float>(kHidden) + p.norm_eps);
  cyc(kProfScalRn);
#pragma unroll
  for (int u = 0; u < kTPC; ++u) {
    // kTPC == 1: every lane group already holds token 0.
#pragma unroll
    for (int k = 0; k < kHc; ++k)
      pre_out[u][k] =
          kTPC == 1 ? pre[k] : __shfl_sync(0xffffffffu, pre_j, 8 * u + k);
    rn_out[u] = kTPC == 1 ? rn : __shfl_sync(0xffffffffu, rn, 8 * u);
  }
  cyc(kProfScalBcast);
  if (!store || (lane >> 3) >= nt)
    return;
  const int64_t tok = tok0 + t;
  if (is_post)
    p.post_out[tok * kHc + j] = s * p.hc_post_mult;
  if (p.write_logits) {
    // comb_logits, two of the 16 per lane.
#pragma unroll
    for (int k = 2 * q; k < 2 * q + 2; ++k)
      p.mix_out[tok * kHc * kHc + k] =
          fmaf(tot[2 * kHc + k] * rms, sm.hcs[2], sm.hcb[2 * kHc + k]);
  }
}

// Sinkhorn warp, one lane per token: comb straight from the cluster totals,
// concurrent with phases 3 and 4 of the compute warps.
template <int kTPC, bool kAlignFleet>
__device__ __forceinline__ void
token_comb(const Params &p, const Smem<kTPC> &sm, int t, int64_t tok) {
  float c[kHc * kHc];
  comb_logits<kTPC>(sm, t, token_rms<kTPC, kAlignFleet>(p, sm, t), c);
  sinkhorn16_fast<kAlignFleet>(c, p.hc_sinkhorn_eps, p.sinkhorn_repeat);
#pragma unroll
  for (int k = 0; k < kHc * kHc; ++k)
    p.mix_out[tok * kHc * kHc + k] = c[k];
}

// Phase 4 for token t: collapse, RMSNorm, layer_out and its g32 fp8 quant;
// pre / rn of the token come from token_scalars.
__device__ __forceinline__ void
norm_quant_out(const Params &p, int64_t tok, int h, int lane, float wn,
               const float (&rf)[kHc], const float (&pre)[kHc], float rn) {
  float y = pre[0] * rf[0];
#pragma unroll
  for (int j = 1; j < kHc; ++j)
    y = fmaf(pre[j], rf[j], y);
  const float yb = __bfloat162float(__float2bfloat16_rn(y));
  const bf16_t ob = __float2bfloat16_rn((yb * rn) * wn);
  p.layer_out[tok * kHidden + h] = ob;
  const float o = __bfloat162float(ob);
  uint8_t be;
  const float inv = g32_quant::group_inv_scale(warp_abs_max(o), be);
  p.q_out[tok * kHidden + h] = quant1_e4m3(o, inv);
  if (lane == 0)
    p.sf_out[tok * (kHidden / g32_quant::kGroupSize) +
             h / g32_quant::kGroupSize] = be;
}
// Cluster all-to-all of this CTA's per-token partials, 9 float4 per (dst,
// token): each is the warp sum over warps 0..7 in that order, lane by lane, as
// the scalar exchange added them; p[34], p[35] carry the zero pad.
template <int kTPC>
__device__ __forceinline__ void
exchange_partials(Smem<kTPC> &sm, int nt, int rank, int tid, uint32_t bar) {
  for (int i = tid; i < kCS * nt * kPV; i += kThreads) {
    const int dst = i % kCS, t = (i / kCS) / kPV, k = (i / kCS) % kPV;
    const float4 *src = reinterpret_cast<const float4 *>(&sm.red[0][t][4 * k]);
    float4 r = src[0];
    float2 lo = make_float2(r.x, r.y), hi = make_float2(r.z, r.w);
#pragma unroll
    for (int ww = 1; ww < kWarps; ++ww) {
      r = *reinterpret_cast<const float4 *>(&sm.red[ww][t][4 * k]);
      lo = __fadd2_rn(lo, make_float2(r.x, r.y));
      hi = __fadd2_rn(hi, make_float2(r.z, r.w));
    }
    st_async_v4(mapa(smem_u32(&sm.cl[rank][t][4 * k]), dst),
                make_float4(lo.x, lo.y, hi.x, hi.y), mapa(bar, dst));
  }
}

template <int kTPC, bool kAlignFleet, bool kUsePDL, bool kFuseAR,
          bool kHasShared, bool kProf, bool kEarlyTrigger, bool kCombLogits>
__global__ void __launch_bounds__(kBlockThreads, 1)
    fused_ar_mhc_kernel(const Params __grid_constant__ p) {
  // The shared fold is MoE-side only, where x needs no all-reduce; the early
  // trigger sits right after the push.
  static_assert(!(kFuseAR && kHasShared) && (kFuseAR || !kEarlyTrigger));
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Smem<kTPC> &sm = *reinterpret_cast<Smem<kTPC> *>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // Warps 0..7 own the columns. Warp kWarps runs the comb sinkhorns and the
  // kFuseAR marker reset, and past the first CTA barrier skips the compute
  // warps' barriers.
  const bool compute = warp < kWarps;
  const int rank = static_cast<int>(cluster_ctarank());
  const int col0 = rank * kThreads;
  const int h = col0 + tid;
  // Push counter and kProf slot of this CTA (push_protocol.txt R3).
  [[maybe_unused]] const int blk = static_cast<int>(blockIdx.y) * kCS + rank;
  const int tok0 = static_cast<int>(blockIdx.y) * kTPC;
  const int nt = min(kTPC, p.num_tokens - tok0);
  const uint32_t bar = smem_u32(&sm.bar);
  const uint32_t tot_bar = smem_u32(&sm.tot_bar);
  [[maybe_unused]] ProfStage *ps = nullptr;
  [[maybe_unused]] uint32_t prof_seq = 0;
  // kFuseAR: warp kWarps's push phase (0 / 1) for the marker reset.
  [[maybe_unused]] uint32_t reset_phase = 0;
  // kProf writes: lane 0 of a whole warp, then __syncwarp. A tid-0-only branch
  // lets nvcc jump-thread tid 0 into the compute body, so warp 0 runs split
  // through its 36 shuffles (measured +4.5 us at T=6).
  const auto lane0 = [&](auto &&f) {
    if constexpr (kProf) {
      if (lane == 0)
        f();
      __syncwarp();
    }
  };
  const auto stamp = [&](int s) {
    if constexpr (kProf)
      if (warp == 0)
        lane0([&] { prof_stamp(*ps, s); });
  };
  if constexpr (kProf) {
    ps = reinterpret_cast<ProfStage *>(smem_raw + sizeof(Smem<kTPC>));
    if (warp == 0)
      lane0([&] { prof_open(*ps); });
  }

  float w[kHc3][kHc];
  float wn = 0.0f;
  // Under kFuseAR nothing precedes this kernel's start (wo_b never triggers
  // early), so the weights wait until the push is out; otherwise they load
  // here, overlapping the previous kernel.
  if constexpr (!kFuseAR) {
    if (compute)
      load_weights(p, h, w, wn);
    if constexpr (kProf)
      if (warp == 0)
        lane0([&] { prof_consume_weights(w, wn, p.prof.ring); });
    if (tid < 3)
      sm.hcs[tid] = p.hc_scale[tid];
    else if (tid < 3 + kHc3)
      sm.hcb[tid - 3] = p.hc_base[tid - 3];
  }
  stamp(1);
  // The pad p[34], p[35] of every warp partial; the reduce-scatter never
  // writes them.
  if (tid < kWarps * kTPC * (kNumPPad - kNumP)) {
    constexpr int kPad = kNumPPad - kNumP;
    sm.red[tid / (kTPC * kPad)][tid / kPad % kTPC][kNumP + tid % kPad] = 0.0f;
  }
  if (tid == 0) {
    mbar_init_expect(bar, static_cast<uint32_t>(kCS * nt * kNumPPad * 4));
    mbar_init(tot_bar, 1);
    fence_mbarrier_init();
  }
  cluster_arrive_relaxed();
  device::PDLWaitPrimary<kUsePDL>();
  stamp(2);

  if (compute) {
    if constexpr (kFuseAR) {
      // Warps 0..nt-1 push and poll; the others load everything else once the
      // push is out, so the x load and the push go first in the memory pipe.
      const int pushers = nt * kRowVecs;
      if (tid < pushers) {
        const int64_t phase_off =
            (p.push.counter[blk].get() & 1) * kWorld * p.push.slot_bytes;
        push_x(p, tok0, nt, col0, tid, phase_off);
        stamp(3);
        // Dependents launch before the poll; their griddepcontrol.wait
        // still waits for this grid's completion.
        if constexpr (kEarlyTrigger)
          device::PDLTriggerSecondary<kUsePDL>();
        named_arrive(kPushBarrier, kThreads);
        load_weights(p, h, w, wn);
        poll_x<kTPC>(p, sm, tok0, nt, col0, tid, phase_off);
        stamp(4);
      } else {
        named_sync(kPushBarrier, kThreads);
        const int lt = tid - pushers, nl = kThreads - pushers;
        // Counters past the grid flip too, so all NB keep one parity.
        if (blk == 0)
          for (int i = kCS * static_cast<int>(gridDim.y) + lt;
               i < p.push.num_blocks; i += nl)
            p.push.counter[i].inc(1);
        load_weights(p, h, w, wn);
        load_inputs<kTPC, false, false, kCombLogits>(p, sm, tok0, nt, col0, lt,
                                                     nl);
        if (lt < 3)
          sm.hcs[lt] = p.hc_scale[lt];
        else if (lt < 3 + kHc3)
          sm.hcb[lt - 3] = p.hc_base[lt - 3];
      }
    } else {
      load_inputs<kTPC, true, kHasShared, kCombLogits>(p, sm, tok0, nt, col0,
                                                       tid, kThreads);
      stamp(3);
    }
  } else {
    if constexpr (kProf)
      prof_seq = p.prof.seq[blk];
    // This CTA's last read of counter[blk]; the flip below follows the first
    // CTA barrier, behind the pushers' reads (push_protocol.txt R3).
    if constexpr (kFuseAR)
      reset_phase = p.push.counter[blk].get() & 1;
    if constexpr (kCombLogits)
      comb_from_logits<kTPC, kAlignFleet>(p, sm, tok0, nt, lane);
  }
  lane0([&] { prof_cycles(*ps, kProfExtra + warp); });
  __syncthreads();
  stamp(5);

  // Past here warp kWarps skips the compute warps' barriers, so its stores
  // (the marker reset) and its sinkhorn stay off them.
  if (compute) {
    float rf[kTPC][kHc] = {};
#pragma unroll
    for (int t = 0; t < kTPC; ++t)
      if (t < nt)
        post_residual_and_reduce<kTPC, kHasShared>(p, sm, w, t, tok0 + t, h,
                                                   tid, lane, warp, rf[t]);
    compute_sync();
    stamp(6);
    // Every CTA sums ranks 0..15 in the same order: bit-identical totals.
    cluster_wait();
    exchange_partials<kTPC>(sm, nt, rank, tid, bar);
    stamp(7);
    mbar_wait_phase0(bar);
    stamp(8);
    if (tid < nt * kNumP) {
      const int t = tid / kNumP, v = tid % kNumP;
      float s = sm.cl[0][t][v];
#pragma unroll
      for (int r = 1; r < kCS; ++r)
        s = s + sm.cl[r][t][v];
      sm.tot[t][v] = s;
    }
    // No DSMEM traffic targets this CTA past its mbarrier; pairs with the exit
    // wait.
    cluster_arrive_relaxed();
    compute_sync();
    stamp(9);
    if (tid == 0)
      mbar_arrive_release(tot_bar);
    // Early launch only; the dependents' griddepcontrol.wait still covers
    // every output written below.
    if constexpr (!kEarlyTrigger)
      device::PDLTriggerSecondary<kUsePDL>();
    // Every compute warp derives pre / rn itself: no barrier before phase 4.
    float pre[kTPC][kHc], rn[kTPC];
    token_scalars<kTPC, kAlignFleet, kProf>(
        p, sm, tok0, nt, lane, warp == 0 && rank == 0, pre, rn, ps, warp == 0);
    stamp(10);
#pragma unroll
    for (int t = 0; t < kTPC; ++t)
      if (t < nt)
        norm_quant_out(p, tok0 + t, h, lane, wn, rf[t], pre[t], rn[t]);
    stamp(11);
  } else {
    if constexpr (kFuseAR) {
      // Flipped here rather than at exit: the next push kernel reads it only
      // after its griddepcontrol.wait, i.e. after this grid completes, and
      // the atomic leaves the exit path.
      if (lane == 0)
        p.push.counter[blk].inc(1);
      reset_slots(p, tok0, nt, col0, lane,
                  reset_phase * kWorld * p.push.slot_bytes);
      lane0([&] { prof_cycles(*ps, kProfResetDone); });
    }
    cluster_wait();
    cluster_arrive_relaxed();
    mbar_wait_phase0(tot_bar);
    lane0([&] { prof_cycles(*ps, kProfTotReady); });
    // CTA c of the cluster takes token c of the group.
    if (lane == 0 && rank < nt && !p.write_logits) {
      if constexpr (kProf)
        prof_stamp(*ps, kProfW0);
      token_comb<kTPC, kAlignFleet>(p, sm, rank, tok0 + rank);
      if constexpr (kProf)
        prof_stamp(*ps, kProfW1);
    }
  }
  stamp(12);
  cluster_wait();
  if constexpr (kProf) {
    __syncthreads();
    if (!compute)
      prof_flush(
          p.prof, *ps, blk, lane, prof_seq,
          prof_meta<kTPC, kFuseAR, kHasShared, kEarlyTrigger, kCombLogits>(
              nt, p.num_tokens));
  }
}

// Body of fused_ar_mhc_sinkhorn_kernel, shared with its L2-prefetch variant.
template <bool kAlignFleet>
__device__ __forceinline__ void
side_sinkhorn(const float *__restrict__ logits, float *__restrict__ comb_out,
              int num_tokens, float eps, int repeat) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= num_tokens)
    return;
  float c[kHc * kHc];
#pragma unroll
  for (int k = 0; k < kHc * kHc; ++k)
    c[k] = logits[t * kHc * kHc + k];
  sinkhorn16_fast<kAlignFleet>(c, eps, repeat);
#pragma unroll
  for (int k = 0; k < kHc * kHc; ++k)
    comb_out[t * kHc * kHc + k] = c[k];
}

// One warp per CTA, one lane per token, lanes past T exit. It runs on a side
// stream beside fc1 and the router, which fill every SM: the minimum-blocks
// bound caps it at 64 registers per thread, so one 32-thread CTA (2K regs)
// fits next to them.
template <bool kAlignFleet>
__global__ void __launch_bounds__(32, 32)
    fused_ar_mhc_sinkhorn_kernel(const float *__restrict__ logits,
                                 float *__restrict__ comb_out, int num_tokens,
                                 float eps, int repeat) {
  side_sinkhorn<kAlignFleet>(logits, comb_out, num_tokens, eps, repeat);
}

// L2 prefetch of what a later fused mHC kernel reads: up to kMaxPrefetchRanges
// byte ranges as cp.async.bulk.prefetch.L2 hints that nothing waits for.
constexpr int kMaxPrefetchRanges = 8;

struct L2PrefetchArgs {
  // 16B-aligned starts; byte counts are multiples of 16, 0 for unused ranges.
  uint64_t addr[kMaxPrefetchRanges];
  uint32_t bytes[kMaxPrefetchRanges];
  // Bytes per prefetch instruction (multiple of 16); chunks are dealt to the
  // issuing threads round-robin across all ranges.
  uint32_t chunk;
};

// Thread w of nw issues global chunks w, w + nw, ... One UBLKPF per chunk; a
// warp serializes its lanes' chunks, so only more warps issue faster.
__device__ __forceinline__ void l2_prefetch(const L2PrefetchArgs &a, uint32_t w,
                                            uint32_t nw) {
  uint32_t first = 0;
#pragma unroll 1
  for (int r = 0; r < kMaxPrefetchRanges; ++r) {
    const uint32_t bytes = a.bytes[r];
    const uint32_t n = (bytes + a.chunk - 1) / a.chunk;
    for (uint32_t c = (w + nw - first % nw) % nw; c < n; c += nw) {
      const uint32_t off = c * a.chunk;
      asm volatile(
          "cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a.addr[r] + off),
          "r"(min(a.chunk, bytes - off))
          : "memory");
    }
    first += n;
  }
}

// fused_ar_mhc_sinkhorn_kernel that first issues the L2 prefetch from every
// lane; the sinkhorn reads none of the prefetched bytes.
template <bool kAlignFleet>
__global__ void __launch_bounds__(32, 32) fused_ar_mhc_sinkhorn_prefetch_kernel(
    const float *__restrict__ logits, float *__restrict__ comb_out,
    int num_tokens, float eps, int repeat,
    const L2PrefetchArgs __grid_constant__ pf) {
  l2_prefetch(pf, blockIdx.x * blockDim.x + threadIdx.x,
              gridDim.x * blockDim.x);
  side_sinkhorn<kAlignFleet>(logits, comb_out, num_tokens, eps, repeat);
}

// The L2 prefetch alone, for a side stream without a deferred sinkhorn.
__global__ void __launch_bounds__(32, 32)
    mhc_l2_prefetch_kernel(const L2PrefetchArgs __grid_constant__ pf) {
  l2_prefetch(pf, blockIdx.x * blockDim.x + threadIdx.x,
              gridDim.x * blockDim.x);
}

// plan [S, 2] int64 host tensor of (address, bytes) rows, S <=
// kMaxPrefetchRanges; rows past S stay empty.
inline L2PrefetchArgs l2_prefetch_args(tvm::ffi::TensorView plan,
                                       int64_t chunk) {
  using namespace host;
  auto S = SymbolicSize{"num_ranges"};
  TensorMatcher({S, int64_t{2}})
      .with_dtype<int64_t>()
      .with_device<kDLCPU>()
      .verify(plan);
  const int64_t n = S.unwrap();
  RuntimeCheck(n <= kMaxPrefetchRanges, "at most ", kMaxPrefetchRanges,
               " L2 prefetch ranges, got ", n);
  // The 1 GiB bounds keep every chunk offset inside uint32.
  constexpr int64_t kMaxBytes = int64_t{1} << 30;
  RuntimeCheck(
      chunk >= 16 && chunk % 16 == 0 && chunk <= kMaxBytes,
      "L2 prefetch chunk must be a multiple of 16 in [16, 1 GiB], got ", chunk);
  const auto *row = static_cast<const int64_t *>(plan.data_ptr());
  L2PrefetchArgs a{};
  a.chunk = static_cast<uint32_t>(chunk);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t addr = row[2 * i], bytes = row[2 * i + 1];
    RuntimeCheck(addr > 0 && addr % 16 == 0 && bytes >= 0 && bytes % 16 == 0 &&
                     bytes <= kMaxBytes,
                 "L2 prefetch range ", i, " needs a 16B-aligned address and ",
                 "a multiple-of-16 size up to 1 GiB, got ", addr, " / ", bytes);
    a.addr[i] = static_cast<uint64_t>(addr);
    a.bytes[i] = static_cast<uint32_t>(bytes);
  }
  return a;
}

template <bool kAlignFleet> struct FusedArMhcSinkhorn {
  static constexpr int kBlock = 32;

  /// \brief comb_out[t] = sinkhorn16_fast(logits[t]), bit-identical to the
  ///        in-kernel comb; logits, comb_out [T, 16] f32. Launched without
  ///        PDL: ceil(T / 32) one-warp CTAs, so one CTA for decode.
  static void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView comb_out,
                  double eps, int64_t repeat) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, int64_t{kHc * kHc}})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(logits)
        .verify(comb_out);
    RuntimeCheck(repeat >= 1, "sinkhorn repeat must be >= 1");
    const int64_t num_tokens = M.unwrap();
    if (num_tokens == 0)
      return;
    const auto blocks =
        static_cast<uint32_t>((num_tokens + kBlock - 1) / kBlock);
    LaunchKernel(blocks, kBlock, device.unwrap())(
        fused_ar_mhc_sinkhorn_kernel<kAlignFleet>,
        static_cast<const float *>(logits.data_ptr()),
        static_cast<float *>(comb_out.data_ptr()), static_cast<int>(num_tokens),
        static_cast<float>(eps), static_cast<int>(repeat));
  }

  /// \brief run() whose lanes first issue the L2 prefetch of `plan`.
  /// \param plan [S, 2] int64 host tensor of (16B-aligned address, bytes)
  ///        ranges, sizes multiples of 16, S <= kMaxPrefetchRanges.
  /// \param chunk Bytes per prefetch instruction, a multiple of 16.
  /// \param ctas Minimum CTA count; CTAs past the tokens only prefetch.
  static void run_prefetch(tvm::ffi::TensorView logits,
                           tvm::ffi::TensorView comb_out,
                           tvm::ffi::TensorView plan, double eps,
                           int64_t repeat, int64_t chunk, int64_t ctas) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, int64_t{kHc * kHc}})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(logits)
        .verify(comb_out);
    RuntimeCheck(repeat >= 1, "sinkhorn repeat must be >= 1");
    RuntimeCheck(ctas >= 1 && ctas <= 1024, "ctas must be in [1, 1024], got ",
                 ctas);
    const L2PrefetchArgs pf = l2_prefetch_args(plan, chunk);
    const int64_t num_tokens = M.unwrap();
    if (num_tokens == 0)
      return;
    const auto blocks = static_cast<uint32_t>(
        std::max<int64_t>((num_tokens + kBlock - 1) / kBlock, ctas));
    LaunchKernel(blocks, kBlock, device.unwrap())(
        fused_ar_mhc_sinkhorn_prefetch_kernel<kAlignFleet>,
        static_cast<const float *>(logits.data_ptr()),
        static_cast<float *>(comb_out.data_ptr()), static_cast<int>(num_tokens),
        static_cast<float>(eps), static_cast<int>(repeat), pf);
  }
};

struct MhcL2Prefetch {
  /// \brief cp.async.bulk.prefetch.L2 of the ranges in `plan` (as in
  ///        FusedArMhcSinkhorn::run_prefetch) from `ctas` one-warp CTAs on
  ///        the current stream of `anchor`'s device; no PDL.
  /// \param anchor Any CUDA tensor of the target device; not read.
  static void run(tvm::ffi::TensorView anchor, tvm::ffi::TensorView plan,
                  int64_t ctas, int64_t chunk) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    device.verify(anchor.device());
    RuntimeCheck(ctas >= 1 && ctas <= 1024, "ctas must be in [1, 1024], got ",
                 ctas);
    const L2PrefetchArgs pf = l2_prefetch_args(plan, chunk);
    LaunchKernel(static_cast<uint32_t>(ctas), 32,
                 device.unwrap())(mhc_l2_prefetch_kernel, pf);
  }
};
template <int64_t kTPC, bool kAlignFleet, bool kUsePDL, bool kFuseAR = false,
          bool kHasShared = false, bool kProf = false,
          bool kEarlyTrigger = false, bool kCombLogits = false>
struct FusedArMhc {
  // One thread per (token, p) for the cluster totals.
  static_assert(kTPC >= 1 && kTPC * kNumP <= kThreads);
  using SmemT = Smem<static_cast<int>(kTPC)>;
  // Smem::ss is last, so variants without the shared fold launch without it;
  // the kProf staging sits past the whole Smem.
  static constexpr int kSmemBytes = static_cast<int>(
      kProf ? sizeof(SmemT) + sizeof(ProfStage)
            : (kHasShared ? sizeof(SmemT) : offsetof(SmemT, ss)));
  static constexpr auto kernel =
      fused_ar_mhc_kernel<static_cast<int>(kTPC), kAlignFleet, kUsePDL, kFuseAR,
                          kHasShared, kProf, kEarlyTrigger, kCombLogits>;

  static void prepare() {
    static const bool done = [] {
      host::RuntimeDeviceCheck(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      host::RuntimeDeviceCheck(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      return true;
    }();
    (void)done;
  }

  /// \brief cudaOccupancyMaxActiveClusters for the (16, 1, 1) cluster.
  static int64_t max_active_clusters() {
    prepare();
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(kCS, 64, 1);
    cfg.blockDim = dim3(kBlockThreads, 1, 1);
    cfg.dynamicSmemBytes = kSmemBytes;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = kCS;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int n = 0;
    host::RuntimeDeviceCheck(cudaOccupancyMaxActiveClusters(&n, kernel, &cfg));
    return n;
  }
  /// \brief Fused mHC on an all-reduced x; mix_out is comb_out [T, 16], or the
  ///        comb logits [T, 16] when write_logits (comb is then left to the
  ///        caller).
  /// \param shared [T, 4096] bf16 added to x in fp32 under kHasShared (MoE
  ///        side); otherwise any [T, 4096] bf16 tensor, unread.
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView shared,
                  tvm::ffi::TensorView residual, tvm::ffi::TensorView post,
                  tvm::ffi::TensorView comb, tvm::ffi::TensorView fn,
                  tvm::ffi::TensorView hc_scale, tvm::ffi::TensorView hc_base,
                  tvm::ffi::TensorView norm_weight,
                  tvm::ffi::TensorView residual_out,
                  tvm::ffi::TensorView layer_out, tvm::ffi::TensorView post_out,
                  tvm::ffi::TensorView mix_out, tvm::ffi::TensorView q_out,
                  tvm::ffi::TensorView sf_out, double rms_eps,
                  double hc_pre_eps, double hc_sinkhorn_eps,
                  double hc_post_mult, int64_t sinkhorn_repeat, double norm_eps,
                  bool write_logits) {
    static_assert(!kFuseAR && !kProf);
    launch(x, shared, residual, post, comb, fn, hc_scale, hc_base, norm_weight,
           residual_out, layer_out, post_out, mix_out, q_out, sf_out, rms_eps,
           hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat, norm_eps,
           write_logits, PushArgs{}, ProfArgs{});
  }

  /// \brief run() with x = this rank's partial, all-reduced in-kernel over
  ///        the LL push plane of `ref` (TP world 4) before the fused mHC.
  static void
  run_push(host::distributed::CommunicatorRef ref, tvm::ffi::TensorView x,
           tvm::ffi::TensorView residual, tvm::ffi::TensorView post,
           tvm::ffi::TensorView comb, tvm::ffi::TensorView fn,
           tvm::ffi::TensorView hc_scale, tvm::ffi::TensorView hc_base,
           tvm::ffi::TensorView norm_weight, tvm::ffi::TensorView residual_out,
           tvm::ffi::TensorView layer_out, tvm::ffi::TensorView post_out,
           tvm::ffi::TensorView mix_out, tvm::ffi::TensorView q_out,
           tvm::ffi::TensorView sf_out, double rms_eps, double hc_pre_eps,
           double hc_sinkhorn_eps, double hc_post_mult, int64_t sinkhorn_repeat,
           double norm_eps, bool write_logits) {
    static_assert(kFuseAR && !kProf);
    launch(x, x, residual, post, comb, fn, hc_scale, hc_base, norm_weight,
           residual_out, layer_out, post_out, mix_out, q_out, sf_out, rms_eps,
           hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat, norm_eps,
           write_logits, push_args(ref), ProfArgs{});
  }

  /// \brief run() of a kProf build (profiling only).
  /// \param prof_ring [L, kMaxClusters, 16, kProfWords] int64 stamps, see
  ///        ProfArgs.
  /// \param prof_seq [kMaxClusters * 16] int32 launch counters, zeroed once.
  /// \param prof_tag Caller's id of this call site, stored with each entry.
  static void
  run_prof(tvm::ffi::TensorView prof_ring, tvm::ffi::TensorView prof_seq,
           int64_t prof_tag, tvm::ffi::TensorView x,
           tvm::ffi::TensorView shared, tvm::ffi::TensorView residual,
           tvm::ffi::TensorView post, tvm::ffi::TensorView comb,
           tvm::ffi::TensorView fn, tvm::ffi::TensorView hc_scale,
           tvm::ffi::TensorView hc_base, tvm::ffi::TensorView norm_weight,
           tvm::ffi::TensorView residual_out, tvm::ffi::TensorView layer_out,
           tvm::ffi::TensorView post_out, tvm::ffi::TensorView mix_out,
           tvm::ffi::TensorView q_out, tvm::ffi::TensorView sf_out,
           double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
           double hc_post_mult, int64_t sinkhorn_repeat, double norm_eps,
           bool write_logits) {
    static_assert(!kFuseAR && kProf);
    launch(x, shared, residual, post, comb, fn, hc_scale, hc_base, norm_weight,
           residual_out, layer_out, post_out, mix_out, q_out, sf_out, rms_eps,
           hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat, norm_eps,
           write_logits, PushArgs{}, prof_args(prof_ring, prof_seq, prof_tag));
  }

  /// \brief run_push() of a kProf build; prof tensors as in run_prof().
  static void
  run_push_prof(tvm::ffi::TensorView prof_ring, tvm::ffi::TensorView prof_seq,
                int64_t prof_tag, host::distributed::CommunicatorRef ref,
                tvm::ffi::TensorView x, tvm::ffi::TensorView residual,
                tvm::ffi::TensorView post, tvm::ffi::TensorView comb,
                tvm::ffi::TensorView fn, tvm::ffi::TensorView hc_scale,
                tvm::ffi::TensorView hc_base, tvm::ffi::TensorView norm_weight,
                tvm::ffi::TensorView residual_out,
                tvm::ffi::TensorView layer_out, tvm::ffi::TensorView post_out,
                tvm::ffi::TensorView mix_out, tvm::ffi::TensorView q_out,
                tvm::ffi::TensorView sf_out, double rms_eps, double hc_pre_eps,
                double hc_sinkhorn_eps, double hc_post_mult,
                int64_t sinkhorn_repeat, double norm_eps, bool write_logits) {
    static_assert(kFuseAR && kProf);
    launch(x, x, residual, post, comb, fn, hc_scale, hc_base, norm_weight,
           residual_out, layer_out, post_out, mix_out, q_out, sf_out, rms_eps,
           hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat, norm_eps,
           write_logits, push_args(ref),
           prof_args(prof_ring, prof_seq, prof_tag));
  }

private:
  static PushArgs push_args(host::distributed::CommunicatorRef ref) {
    const auto &data = *ref.get();
    const auto &plane = data.get_push_obj();
    host::RuntimeCheck(data.get_world_size() == kWorld,
                       "push all-reduce needs TP world size 4");
    PushArgs push{};
    for (int i = 0; i < kWorld; ++i)
      push.ws[i] = plane.workspaces[i];
    push.counter = plane.counter;
    push.slot_bytes = plane.slot_bytes;
    push.tp_rank = static_cast<int>(data.get_rank());
    push.num_blocks = static_cast<int>(plane.num_blocks);
    return push;
  }

  static ProfArgs prof_args(tvm::ffi::TensorView ring, tvm::ffi::TensorView seq,
                            int64_t tag) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto L = SymbolicSize{"ring_len"};
    device.set_options<kDLCUDA>();
    TensorMatcher({L, int64_t{kMaxClusters}, int64_t{kCS}, int64_t{kProfWords}})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(ring);
    TensorMatcher({int64_t{kMaxClusters * kCS}})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq);
    return ProfArgs{.ring = static_cast<uint64_t *>(ring.data_ptr()),
                    .seq = static_cast<uint32_t *>(seq.data_ptr()),
                    .ring_len = static_cast<int>(L.unwrap()),
                    .tag = static_cast<int64_t>(tag)};
  }

  static void
  launch(tvm::ffi::TensorView x, tvm::ffi::TensorView shared,
         tvm::ffi::TensorView residual, tvm::ffi::TensorView post,
         tvm::ffi::TensorView comb, tvm::ffi::TensorView fn,
         tvm::ffi::TensorView hc_scale, tvm::ffi::TensorView hc_base,
         tvm::ffi::TensorView norm_weight, tvm::ffi::TensorView residual_out,
         tvm::ffi::TensorView layer_out, tvm::ffi::TensorView post_out,
         tvm::ffi::TensorView mix_out, tvm::ffi::TensorView q_out,
         tvm::ffi::TensorView sf_out, double rms_eps, double hc_pre_eps,
         double hc_sinkhorn_eps, double hc_post_mult, int64_t sinkhorn_repeat,
         double norm_eps, bool write_logits, const PushArgs &push,
         const ProfArgs &prof) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    device.set_options<kDLCUDA>();
    const int64_t H = kHidden, HC = kHc;
    TensorMatcher({M, H})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(x)
        .verify(shared)
        .verify(layer_out);
    TensorMatcher({M, HC, H})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual)
        .verify(residual_out);
    TensorMatcher({M, HC})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(post)
        .verify(post_out);
    TensorMatcher({M, HC * HC})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(comb)
        .verify(mix_out);
    TensorMatcher({int64_t{kHc3}, HC, H})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(fn);
    TensorMatcher({int64_t{3}})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(hc_scale);
    TensorMatcher({int64_t{kHc3}})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(hc_base);
    TensorMatcher({H}).with_dtype<bf16_t>().with_device(device).verify(
        norm_weight);
    TensorMatcher({M, H})
        .with_dtype<fp8_e4m3_t, uint8_t>()
        .with_device(device)
        .verify(q_out);
    TensorMatcher({M, H / g32_quant::kGroupSize})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(sf_out);

    const int64_t num_tokens = M.unwrap();
    if (num_tokens == 0)
      return;
    RuntimeCheck(sinkhorn_repeat >= 1, "sinkhorn_repeat must be >= 1");
    const auto aligned16 = [](tvm::ffi::TensorView t) {
      return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
    };
    RuntimeCheck(aligned16(x) && aligned16(shared) && aligned16(residual),
                 "x, shared and residual must be 16B aligned");
    prepare();
    const Params params{
        .x = static_cast<const bf16_t *>(x.data_ptr()),
        .residual = static_cast<const bf16_t *>(residual.data_ptr()),
        .post = static_cast<const float *>(post.data_ptr()),
        .comb = static_cast<const float *>(comb.data_ptr()),
        .fn = static_cast<const float *>(fn.data_ptr()),
        .hc_scale = static_cast<const float *>(hc_scale.data_ptr()),
        .hc_base = static_cast<const float *>(hc_base.data_ptr()),
        .norm_weight = static_cast<const bf16_t *>(norm_weight.data_ptr()),
        .residual_out = static_cast<bf16_t *>(residual_out.data_ptr()),
        .layer_out = static_cast<bf16_t *>(layer_out.data_ptr()),
        .post_out = static_cast<float *>(post_out.data_ptr()),
        .mix_out = static_cast<float *>(mix_out.data_ptr()),
        .q_out = static_cast<uint8_t *>(q_out.data_ptr()),
        .sf_out = static_cast<uint8_t *>(sf_out.data_ptr()),
        .num_tokens = static_cast<int>(num_tokens),
        .sinkhorn_repeat = static_cast<int>(sinkhorn_repeat),
        .write_logits = write_logits ? 1 : 0,
        .rms_eps = static_cast<float>(rms_eps),
        .hc_pre_eps = static_cast<float>(hc_pre_eps),
        .hc_sinkhorn_eps = static_cast<float>(hc_sinkhorn_eps),
        .hc_post_mult = static_cast<float>(hc_post_mult),
        .norm_eps = static_cast<float>(norm_eps),
        .push = push,
        .shared = kHasShared ? static_cast<const bf16_t *>(shared.data_ptr())
                             : nullptr,
        .prof = prof,
    };
    const auto num_clusters =
        static_cast<uint32_t>((num_tokens + kTPC - 1) / kTPC);
    if constexpr (kFuseAR) {
      RuntimeCheck(kCS * num_clusters <= static_cast<uint32_t>(push.num_blocks),
                   "grid exceeds the push counters (push_protocol.txt R1)");
      RuntimeCheck(num_tokens * kHidden * int64_t{sizeof(bf16_t)} <=
                       push.slot_bytes,
                   "x exceeds the push slot");
    }
    if constexpr (kProf)
      RuntimeCheck(num_clusters <= kMaxClusters && prof.ring_len > 0,
                   "kProf ring holds at most kMaxClusters clusters");
    LaunchKernel(dim3(kCS, num_clusters, 1), kBlockThreads, device.unwrap(),
                 kSmemBytes)
        .config({.use_pdl = kUsePDL, .cluster_dim = dim3(kCS, 1, 1)})(kernel,
                                                                      params);
  }
};

} // namespace fused_ar_mhc
} // namespace eb5
} // namespace sglang

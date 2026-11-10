// Group-32 fp8 e4m3 + UE8M0 quant, bit-exact with HALOps
// group_wise_quant(x, mode="w8a8-mxfp8", use_swizzle=False) on bf16 input.
//
// The HALOps kernel is fast-math code; every op below is pinned in PTX to
// its SASS (group_wise_quant_kernel<__nv_bfloat16, 32, false>), so the bits do
// not depend on our compile flags:
//   be = clamp(ceil(lg2(max(amax, 1e-10) * rn(1/448))) + 127, 0, 254)
//   q  = e4m3_satfinite(min(max(v * ex2(127 - be), -448), 448))
// with FTZ everywhere and MUFU lg2 / ex2. MUFU lg2(2^k) is not exact for k < 0,
// so amax = 448 * 2^k there gives be = 128 + k; an exact log2f is off by one.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>

#include <cstdint>

namespace sglang {
namespace eb5 {
namespace g32_quant {

constexpr int kGroupSize = 32;
constexpr int kElemsPerLane = 8;
constexpr int kLanesPerGroup = kGroupSize / kElemsPerLane;

namespace detail {

__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// PTX max/min return the non-NaN operand, like fmaxf/fminf.
__device__ __forceinline__ float max_ftz(float a, float b) {
  float r;
  asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float min_ftz(float a, float b) {
  float r;
  asm("min.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// Order matters for NaN: max first sends it to -448 (0xFE), as HALOps does.
__device__ __forceinline__ float scale_clamp(float v, float inv) {
  return min_ftz(max_ftz(mul_ftz(v, inv), -448.0f), 448.0f);
}

} // namespace detail

/// \brief max |v[i]|; NaN entries are ignored, like the reference amax.
__device__ __forceinline__ float abs_max8(const float (&v)[8]) {
  float m = fabsf(v[0]);
#pragma unroll
  for (int i = 1; i < 8; ++i)
    m = detail::max_ftz(m, fabsf(v[i]));
  return m;
}

/// \brief Max of `local_amax` over the 4 lanes 4g..4g+3 that share group g.
///        All 32 lanes of the warp must call it.
__device__ __forceinline__ float group_amax_4lanes(float local_amax) {
  local_amax =
      detail::max_ftz(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, 1));
  local_amax =
      detail::max_ftz(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, 2));
  return local_amax;
}

/// \brief UE8M0 scale of one 32-element group and its fp8 multiplier.
/// \param amax Group max |v| from a NaN-ignoring max; 0 or NaN (all-NaN group)
///        hits the 1e-10 floor, be = 85.
/// \param scale_byte Receives the biased exponent be (scale = 2^(be - 127)).
/// \return 2^(127 - be), the multiplier quant8_e4m3 expects. An Inf in the
///         group gives be = 254 and a zero multiplier (2^-127 is flushed).
__device__ __forceinline__ float group_inv_scale(float amax,
                                                 uint8_t &scale_byte) {
  float s, lg, be, e, inv;
  // 0x2EDBE6FF = 1e-10f; 0x3B124925 = rn(1/448), fast-math's amax / 448.
  asm("max.ftz.f32 %0, %1, 0f2EDBE6FF;" : "=f"(s) : "f"(amax));
  asm("mul.rn.ftz.f32 %0, %1, 0f3B124925;" : "=f"(s) : "f"(s));
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(lg) : "f"(s));
  asm("cvt.rpi.ftz.f32.f32 %0, %1;" : "=f"(be) : "f"(lg));
  asm("add.rn.ftz.f32 %0, %1, 0f42FE0000;" : "=f"(be) : "f"(be));
  be = detail::min_ftz(detail::max_ftz(be, 0.0f), 254.0f);
  asm("sub.rn.ftz.f32 %0, 0f42FE0000, %1;" : "=f"(e) : "f"(be));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(e));
  scale_byte = static_cast<uint8_t>(__float2uint_rz(be));
  return inv;
}

/// \brief e4m3 bytes of v[i] * inv; byte i (little-endian) holds v[i].
///        NaN, and Inf (its group multiplier is 0), become 0xFE (-448).
/// \param v Values already rounded to bf16, as group_wise_quant's input is.
/// \param inv Multiplier from group_inv_scale.
__device__ __forceinline__ uint2 quant8_e4m3(const float (&v)[8], float inv) {
  uint32_t pair[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float lo = detail::scale_clamp(v[2 * i], inv);
    const float hi = detail::scale_clamp(v[2 * i + 1], inv);
    uint16_t packed;
    // The first source lands in the high byte.
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(packed)
        : "f"(hi), "f"(lo));
    pair[i] = packed;
  }
  return make_uint2(pair[0] | (pair[1] << 16), pair[2] | (pair[3] << 16));
}

// Standalone [T, H] quant for the bit-exactness test; one lane per 8 bf16,
// 4 lanes per group, the layout the fused kernels use.
template <int kThreads>
__global__ void g32_quant_test_kernel(const bf16_t *__restrict__ x,
                                      uint8_t *__restrict__ q,
                                      uint8_t *__restrict__ sf, int hidden) {
  const int col = (blockIdx.x * kThreads + threadIdx.x) * kElemsPerLane;
  // hidden % 256 == 0 keeps this exit warp-uniform for the full-mask shuffles.
  if (col >= hidden)
    return;
  const int64_t off = static_cast<int64_t>(blockIdx.y) * hidden + col;
  const uint4 raw = *reinterpret_cast<const uint4 *>(x + off);
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
  float v[kElemsPerLane];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[2 * i] = __uint_as_float(words[i] << 16);
    v[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
  uint8_t scale_byte;
  const float inv = group_inv_scale(group_amax_4lanes(abs_max8(v)), scale_byte);
  *reinterpret_cast<uint2 *>(q + off) = quant8_e4m3(v, inv);
  if (threadIdx.x % kLanesPerGroup == 0)
    sf[off / kGroupSize] = scale_byte;
}

template <int kThreads = 128> struct G32QuantTest {
  static_assert(kThreads % 32 == 0);

  /// \brief group_wise_quant(x, is_fp4=False, use_swizzle=False) clone.
  /// \param x [T, H] bf16, H % 256 == 0.
  /// \param q_out [T, H] fp8 e4m3 (or uint8 bytes).
  /// \param sf_out [T, H / 32] uint8 UE8M0, row-major.
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView q_out,
                  tvm::ffi::TensorView sf_out) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto T = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden"};
    auto G = SymbolicSize{"num_groups"};
    device.set_options<kDLCUDA>();
    TensorMatcher({T, H}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({T, H})
        .with_dtype<fp8_e4m3_t, uint8_t>()
        .with_device(device)
        .verify(q_out);
    TensorMatcher({T, G}).with_dtype<uint8_t>().with_device(device).verify(
        sf_out);

    const int64_t num_tokens = T.unwrap(), hidden = H.unwrap();
    RuntimeCheck(hidden % 256 == 0, "hidden must be a multiple of 256");
    RuntimeCheck(G.unwrap() * kGroupSize == hidden,
                 "sf_out must be [num_tokens, hidden / 32]");
    RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(q_out.data_ptr()) % 8 == 0,
                 "x must be 16B and q_out 8B aligned");
    if (num_tokens == 0)
      return;

    constexpr int64_t kElemsPerBlock = kThreads * kElemsPerLane;
    const dim3 grid(
        static_cast<uint32_t>((hidden + kElemsPerBlock - 1) / kElemsPerBlock),
        static_cast<uint32_t>(num_tokens));
    LaunchKernel(grid, kThreads, device.unwrap())(
        g32_quant_test_kernel<kThreads>,
        static_cast<const bf16_t *>(x.data_ptr()),
        static_cast<uint8_t *>(q_out.data_ptr()),
        static_cast<uint8_t *>(sf_out.data_ptr()), static_cast<int>(hidden));
  }
};

} // namespace g32_quant
} // namespace eb5
} // namespace sglang

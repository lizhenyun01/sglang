
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::pack_fp8;

struct AddRmsNormQuantParams {
  const bf16_t* __restrict__ x;
  const bf16_t* __restrict__ residual;
  const bf16_t* __restrict__ weight;
  bf16_t* __restrict__ residual_out;
  // Optional bf16 rms_norm(x) * weight; nullptr = not written.
  bf16_t* __restrict__ norm_out;
  fp8_e4m3_t* __restrict__ q;
  int32_t* __restrict__ sf;
  float eps;
  uint32_t hidden;
  uint32_t num_groups;
  uint32_t scale_row_stride_int32;
  // Row stride of x in elements; lets x be a strided slice such as qkv_a[:, :q_lora].
  uint32_t x_row_stride;
};

template <uint32_t kGroupSize, bool kHasResidual, bool kUsePDL>
__global__ __launch_bounds__(1024, 1) void add_rmsnorm_group_quant_ue8m0_kernel(
    const AddRmsNormQuantParams __grid_constant__ params) {
  using namespace device;
  constexpr uint32_t kVecElems = 8u;
  constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;
  using Vec = AlignedVector<bf16x2_t, kVecElems / 2>;
  using OutVec = AlignedVector<fp8x2_e4m3_t, kVecElems / 2>;

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const auto base = static_cast<uint64_t>(token) * params.hidden;
  const auto xbase = static_cast<uint64_t>(token) * params.x_row_stride;

  PDLWaitPrimary<kUsePDL>();
  Vec xv, rv, wv;
  xv.load(params.x + xbase, tid);
  if constexpr (kHasResidual) rv.load(params.residual + base, tid);
  wv.load(params.weight, tid);

  float s[kVecElems];
  float local_sumsq = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kVecElems / 2; ++i) {
    const auto [x0, x1] = cast<fp32x2_t>(xv[i]);
    if constexpr (kHasResidual) {
      const auto [r0, r1] = cast<fp32x2_t>(rv[i]);
      s[2 * i + 0] = x0 + r0;
      s[2 * i + 1] = x1 + r1;
    } else {
      s[2 * i + 0] = x0;
      s[2 * i + 1] = x1;
    }
    local_sumsq += s[2 * i + 0] * s[2 * i + 0] + s[2 * i + 1] * s[2 * i + 1];
  }
  if constexpr (kHasResidual) {
    Vec ro;
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i)
      ro[i] = cast<bf16x2_t>(fp32x2_t{s[2 * i + 0], s[2 * i + 1]});
    ro.store(params.residual_out + base, tid);
  }

  local_sumsq = warp::reduce_sum(local_sumsq);
  __shared__ float s_warp[32];
  const uint32_t lane = tid % kWarpThreads;
  const uint32_t warp = tid / kWarpThreads;
  const uint32_t num_warps = blockDim.x / kWarpThreads;
  if (lane == 0) s_warp[warp] = local_sumsq;
  __syncthreads();
  float total = 0.0f;
  if (warp == 0) {
    float v = lane < num_warps ? s_warp[lane] : 0.0f;
    v = warp::reduce_sum(v);
    if (lane == 0) s_warp[0] = v;
  }
  __syncthreads();
  const float rms = rsqrtf(s_warp[0] / static_cast<float>(params.hidden) + params.eps);

  float n[kVecElems];
  float local_max = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kVecElems / 2; ++i) {
    const auto [w0, w1] = cast<fp32x2_t>(wv[i]);
    n[2 * i + 0] = s[2 * i + 0] * rms * w0;
    n[2 * i + 1] = s[2 * i + 1] * rms * w1;
    if constexpr (kGroupSize == 32) {
      n[2 * i + 0] = cast<fp32_t>(cast<bf16_t>(n[2 * i + 0]));
      n[2 * i + 1] = cast<fp32_t>(cast<bf16_t>(n[2 * i + 1]));
    }
    local_max = fmaxf(local_max, fmaxf(fabsf(n[2 * i + 0]), fabsf(n[2 * i + 1])));
  }
  if (params.norm_out != nullptr) {
    Vec no;
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i)
      no[i] = cast<bf16x2_t>(fp32x2_t{n[2 * i + 0], n[2 * i + 1]});
    no.store(params.norm_out + base, tid);
  }
  local_max = warp::reduce_max<kThreadsPerGroup>(local_max);
  const uint32_t group_id = tid / kThreadsPerGroup;

  if constexpr (kGroupSize == 32) {
    const float amax = fmaxf(1e-10f, local_max);
    const float scale = amax * (1.0f / 448.0f);
    const float biased_exp = fminf(fmaxf(ceilf(__log2f(scale)) + 127.0f, 0.0f), 254.0f);
    const float inv_scale = exp2f(127.0f - biased_exp);

    OutVec qv;
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i)
      qv[i] = fp8x2_e4m3_t{fp32x2_t{
          fminf(fmaxf(n[2 * i + 0] * inv_scale, -448.0f), 448.0f),
          fminf(fmaxf(n[2 * i + 1] * inv_scale, -448.0f), 448.0f)}};
    PDLTriggerSecondary<kUsePDL>();
    qv.store(params.q + base, tid);
    if (tid % kThreadsPerGroup == 0)
      reinterpret_cast<uint8_t*>(params.sf)[token * params.num_groups + group_id] = static_cast<uint8_t>(biased_exp);
    return;
  }

  const uint32_t ue8m0_exp = cast_to_ue8m0(fmaxf(local_max, 1e-10f) / math::FP8_E4M3_MAX);
  const float inv_scale = __uint_as_float((127u + 127u - ue8m0_exp) << 23);

  OutVec qv;
#pragma unroll
  for (uint32_t i = 0; i < kVecElems / 2; ++i)
    qv[i] = pack_fp8(n[2 * i + 0] * inv_scale, n[2 * i + 1] * inv_scale);

  PDLTriggerSecondary<kUsePDL>();
  qv.store(params.q + base, tid);

  if (tid % kThreadsPerGroup == 0 && group_id < params.num_groups) {
    uint8_t* base = reinterpret_cast<uint8_t*>(params.sf);
    base[(group_id / 4u) * (params.scale_row_stride_int32 * 4u) + token * 4u + (group_id % 4u)] =
        static_cast<uint8_t>(ue8m0_exp);
  }
}

template <int64_t kGroupSize, bool kHasResidual, bool kUsePDL>
static void _launch_add_rmsnorm_group_quant_ue8m0(
    tvm::ffi::TensorView x,
    const bf16_t* residual,
    const bf16_t* weight,
    bf16_t* residual_out,
    bf16_t* norm_out,
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView sf,
    double eps) {
  using namespace host;
  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto H = SymbolicSize{"hidden"};
  auto G4 = SymbolicSize{"num_groups_div_4"};
  auto M_pad = SymbolicSize{"M padded"};
  device.set_options<kDLCUDA>();
  auto Xs = SymbolicSize{"x_row_stride"};
  TensorMatcher({M, H}).with_strides({Xs, int64_t{1}}).with_dtype<bf16_t>().with_device(device).verify(x);
  TensorMatcher({M, H}).with_dtype<fp8_e4m3_t>().with_device(device).verify(q);

  const uint32_t num_tokens = static_cast<uint32_t>(M.unwrap());
  const uint32_t hidden = static_cast<uint32_t>(H.unwrap());
  RuntimeCheck(hidden % kGroupSize == 0, "hidden not divisible by group_size");
  const uint32_t num_groups = hidden / static_cast<uint32_t>(kGroupSize);
  uint32_t scale_row_stride_int32 = 0;
  if constexpr (kGroupSize == 32) {
    auto G = SymbolicSize{"num_groups"};
    G.set_value(num_groups);
    TensorMatcher({M, G}).with_dtype<uint8_t>().with_device(device).verify(sf);
  } else {
    TensorMatcher({M, G4}).with_strides({int64_t{1}, M_pad}).with_dtype<int32_t>().with_device(device).verify(sf);
    RuntimeCheck(num_groups % 4 == 0, "num_groups must be a multiple of 4 for int32 packing");
    scale_row_stride_int32 = static_cast<uint32_t>(M_pad.unwrap());
  }
  const uint32_t block = hidden / 8u;
  RuntimeCheck(block > 0 && block <= 1024, "hidden/8 must be in (0, 1024]");
  if (num_tokens == 0) return;

  const auto params = AddRmsNormQuantParams{
      .x = static_cast<const bf16_t*>(x.data_ptr()),
      .residual = residual,
      .weight = weight,
      .residual_out = residual_out,
      .norm_out = norm_out,
      .q = static_cast<fp8_e4m3_t*>(q.data_ptr()),
      .sf = static_cast<int32_t*>(sf.data_ptr()),
      .eps = static_cast<float>(eps),
      .hidden = hidden,
      .num_groups = num_groups,
      .scale_row_stride_int32 = scale_row_stride_int32,
      .x_row_stride = static_cast<uint32_t>(Xs.unwrap()),
  };
  LaunchKernel(num_tokens, block, device.unwrap())
      .enable_pdl(kUsePDL)(add_rmsnorm_group_quant_ue8m0_kernel<kGroupSize, kHasResidual, kUsePDL>, params);
}

template <int64_t kGroupSize, bool kUsePDL>
void add_rmsnorm_group_quant_ue8m0(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView residual_out,
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView sf,
    double eps) {
  using namespace host;
  auto Mr = SymbolicSize{"m"};
  auto Hr = SymbolicSize{"h"};
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({Mr, Hr}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(residual_out);
  _launch_add_rmsnorm_group_quant_ue8m0<kGroupSize, true, kUsePDL>(
      x,
      static_cast<const bf16_t*>(residual.data_ptr()),
      static_cast<const bf16_t*>(weight.data_ptr()),
      static_cast<bf16_t*>(residual_out.data_ptr()),
      nullptr,
      q,
      sf,
      eps);
}

template <int64_t kGroupSize, bool kUsePDL>
void rmsnorm_group_quant_ue8m0(
    tvm::ffi::TensorView x, tvm::ffi::TensorView weight, tvm::ffi::TensorView q, tvm::ffi::TensorView sf, double eps) {
  _launch_add_rmsnorm_group_quant_ue8m0<kGroupSize, false, kUsePDL>(
      x, nullptr, static_cast<const bf16_t*>(weight.data_ptr()), nullptr, nullptr, q, sf, eps);
}

// Same as rmsnorm_group_quant_ue8m0 plus the bf16 normed output, for callers that
// still need rms_norm(x) * weight in bf16 next to the (q, sf) pair.
template <int64_t kGroupSize, bool kUsePDL>
void rmsnorm_group_quant_ue8m0_with_norm(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView norm_out,
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView sf,
    double eps) {
  using namespace host;
  auto Mn = SymbolicSize{"m_norm"};
  auto Hn = SymbolicSize{"h_norm"};
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({Mn, Hn}).with_dtype<bf16_t>().with_device(dev).verify(norm_out);
  _launch_add_rmsnorm_group_quant_ue8m0<kGroupSize, false, kUsePDL>(
      x,
      nullptr,
      static_cast<const bf16_t*>(weight.data_ptr()),
      nullptr,
      static_cast<bf16_t*>(norm_out.data_ptr()),
      q,
      sf,
      eps);
}

}  // namespace sglang

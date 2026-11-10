
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

struct SigmoidGateMulQuantParams {
  const bf16_t* __restrict__ o;
  const bf16_t* __restrict__ gate;
  fp8_e4m3_t* __restrict__ q;
  int32_t* __restrict__ sf;
  uint32_t hidden;
  uint32_t num_groups;
  uint32_t scale_row_stride_int32;
  // Gate row stride in elements: hidden when contiguous, the merged [q_up | gate]
  // row width when gate is the strided slice of the q_b output.
  uint32_t gate_row_stride;
};

template <uint32_t kGroupSize, bool kUsePDL>
__global__ __launch_bounds__(1024, 1) void sigmoid_gate_mul_group_quant_kernel(
    const SigmoidGateMulQuantParams __grid_constant__ params) {
  using namespace device;
  constexpr uint32_t kVecElems = 8u;
  constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;
  using Vec = AlignedVector<bf16x2_t, kVecElems / 2>;
  using OutVec = AlignedVector<fp8x2_e4m3_t, kVecElems / 2>;

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const auto base = static_cast<uint64_t>(token) * params.hidden;

  PDLTriggerSecondary<kUsePDL>();
  PDLWaitPrimary<kUsePDL>();
  Vec ov, gv;
  ov.load(params.o + base, tid);
  gv.load(params.gate + static_cast<uint64_t>(token) * params.gate_row_stride, tid);

  float n[kVecElems];
  float local_max = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kVecElems / 2; ++i) {
    const auto [o0, o1] = cast<fp32x2_t>(ov[i]);
    const auto [g0, g1] = cast<fp32x2_t>(gv[i]);
    const float s0 = cast<fp32_t>(cast<bf16_t>(1.0f / (1.0f + expf(-g0))));
    const float s1 = cast<fp32_t>(cast<bf16_t>(1.0f / (1.0f + expf(-g1))));
    n[2 * i + 0] = cast<fp32_t>(cast<bf16_t>(o0 * s0));
    n[2 * i + 1] = cast<fp32_t>(cast<bf16_t>(o1 * s1));
    local_max = fmaxf(local_max, fmaxf(fabsf(n[2 * i + 0]), fabsf(n[2 * i + 1])));
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

  qv.store(params.q + base, tid);

  if (tid % kThreadsPerGroup == 0 && group_id < params.num_groups) {
    uint8_t* base_sf = reinterpret_cast<uint8_t*>(params.sf);
    base_sf[(group_id / 4u) * (params.scale_row_stride_int32 * 4u) + token * 4u + (group_id % 4u)] =
        static_cast<uint8_t>(ue8m0_exp);
  }
}

template <int64_t kGroupSize, bool kUsePDL>
void sigmoid_gate_mul_group_quant(
    tvm::ffi::TensorView o, tvm::ffi::TensorView gate, tvm::ffi::TensorView q, tvm::ffi::TensorView sf) {
  using namespace host;
  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto H = SymbolicSize{"hidden"};
  auto G4 = SymbolicSize{"num_groups_div_4"};
  auto M_pad = SymbolicSize{"M padded"};
  device.set_options<kDLCUDA>();
  auto GS = SymbolicSize{"gate_row_stride"};
  TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device).verify(o);
  // Row-strided gate (unit inner stride); 16B-aligned rows for the 8 x bf16 loads.
  TensorMatcher({M, H})
      .with_strides({GS, int64_t{1}})
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(gate);
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
  // A single token row never reads its stride (torch may report any value there).
  const uint32_t gate_row_stride = num_tokens > 1 ? static_cast<uint32_t>(GS.unwrap()) : hidden;
  RuntimeCheck(gate_row_stride >= hidden, "gate rows overlap");

  const auto params = SigmoidGateMulQuantParams{
      .o = static_cast<const bf16_t*>(o.data_ptr()),
      .gate = static_cast<const bf16_t*>(gate.data_ptr()),
      .q = static_cast<fp8_e4m3_t*>(q.data_ptr()),
      .sf = static_cast<int32_t*>(sf.data_ptr()),
      .hidden = hidden,
      .num_groups = num_groups,
      .scale_row_stride_int32 = scale_row_stride_int32,
      .gate_row_stride = gate_row_stride,
  };
  LaunchKernel(num_tokens, block, device.unwrap())
      .enable_pdl(kUsePDL)(sigmoid_gate_mul_group_quant_kernel<kGroupSize, kUsePDL>, params);
}

}  // namespace sglang

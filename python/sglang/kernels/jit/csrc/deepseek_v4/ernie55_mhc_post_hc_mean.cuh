
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

template <uint32_t kHcMult>
struct MhcPostHcMeanParams {
  const bf16_t* __restrict__ x;
  const bf16_t* __restrict__ residual;
  const fp32_t* __restrict__ post;
  const fp32_t* __restrict__ comb;
  bf16_t* __restrict__ out;
  uint32_t hidden;
};

template <uint32_t kHcMult, bool kUsePDL>
__global__
__launch_bounds__(256, 1) void mhc_post_hc_mean_kernel(const MhcPostHcMeanParams<kHcMult> __grid_constant__ params) {
  using namespace device;
  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t nthreads = blockDim.x;
  const uint32_t H = params.hidden;

  __shared__ float s_post[kHcMult];
  __shared__ float s_comb[kHcMult * kHcMult];

  PDLTriggerSecondary<kUsePDL>();
  PDLWaitPrimary<kUsePDL>();
  if (tid < kHcMult) s_post[tid] = params.post[token * kHcMult + tid];
  if (tid < kHcMult * kHcMult) s_comb[tid] = params.comb[token * kHcMult * kHcMult + tid];
  __syncthreads();

  const uint64_t res_base = static_cast<uint64_t>(token) * kHcMult * H;
  const uint64_t row_base = static_cast<uint64_t>(token) * H;
  constexpr float kInvHc = 1.0f / static_cast<float>(kHcMult);

  for (uint32_t h = tid; h < H; h += nthreads) {
    const float xh = cast<fp32_t>(params.x[row_base + h]);
    float res[kHcMult];
#pragma unroll
    for (uint32_t hci = 0; hci < kHcMult; ++hci)
      res[hci] = cast<fp32_t>(params.residual[res_base + static_cast<uint64_t>(hci) * H + h]);
    float mean = 0.0f;
#pragma unroll
    for (uint32_t hco = 0; hco < kHcMult; ++hco) {
      float acc = s_post[hco] * xh;
#pragma unroll
      for (uint32_t hci = 0; hci < kHcMult; ++hci)
        acc += s_comb[hci * kHcMult + hco] * res[hci];
      mean += cast<fp32_t>(cast<bf16_t>(acc));
    }
    params.out[row_base + h] = cast<bf16_t>(mean * kInvHc);
  }
}

template <int64_t kHcMult, bool kUsePDL>
void mhc_post_hc_mean(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView post,
    tvm::ffi::TensorView comb,
    tvm::ffi::TensorView out) {
  using namespace host;
  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto H = SymbolicSize{"hidden"};
  device.set_options<kDLCUDA>();
  TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device).verify(x).verify(out);
  TensorMatcher({M, kHcMult, H}).with_dtype<bf16_t>().with_device(device).verify(residual);
  TensorMatcher({M, kHcMult}).with_dtype<fp32_t>().with_device(device).verify(post);
  TensorMatcher({M, kHcMult, kHcMult}).with_dtype<fp32_t>().with_device(device).verify(comb);

  const uint32_t num_tokens = static_cast<uint32_t>(M.unwrap());
  const uint32_t hidden = static_cast<uint32_t>(H.unwrap());
  if (num_tokens == 0) return;

  const auto params = MhcPostHcMeanParams<kHcMult>{
      .x = static_cast<const bf16_t*>(x.data_ptr()),
      .residual = static_cast<const bf16_t*>(residual.data_ptr()),
      .post = static_cast<const fp32_t*>(post.data_ptr()),
      .comb = static_cast<const fp32_t*>(comb.data_ptr()),
      .out = static_cast<bf16_t*>(out.data_ptr()),
      .hidden = hidden,
  };
  LaunchKernel(num_tokens, 256u, device.unwrap())
      .enable_pdl(kUsePDL)(mhc_post_hc_mean_kernel<kHcMult, kUsePDL>, params);
}

}  // namespace sglang

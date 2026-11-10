#pragma once

// EB5 HCA decode: q_norm + group-32 quant and kv_norm + RoPE + FP8 SWA-cache
// store in one launch, so both read qkv_a behind the wqkv_a GEMM's PDL edge and
// the kv work leaves the side stream. Each role is the stock kernel's code
// (bitwise).
#include "main_norm_rope.cuh"
#include "ernie55_add_rmsnorm_group_quant_ue8m0.cuh"

namespace sglang {

struct Eb5HcaQkvPrepParams {
  AddRmsNormQuantParams q;
  FusedKNormRopeFlashMLAParams k;
  uint32_t num_tokens;
};

// add_rmsnorm_group_quant_ue8m0_kernel<32, false, kUsePDL> for one token on the
// first hidden/8 threads; the rest of the 256-thread block only joins the
// barriers.
template <bool kUsePDL>
SGL_DEVICE void eb5_hca_q_role(const AddRmsNormQuantParams &params,
                               uint32_t token) {
  using namespace device;
  constexpr uint32_t kVecElems = 8u;
  constexpr uint32_t kThreadsPerGroup = 32u / kVecElems;
  using Vec = AlignedVector<bf16x2_t, kVecElems / 2>;
  using OutVec = AlignedVector<fp8x2_e4m3_t, kVecElems / 2>;

  const uint32_t tid = threadIdx.x;
  const uint32_t q_threads = params.hidden / kVecElems;
  const bool active = tid < q_threads;
  const auto base = static_cast<uint64_t>(token) * params.hidden;
  const auto xbase = static_cast<uint64_t>(token) * params.x_row_stride;

  PDLWaitPrimary<kUsePDL>();
  Vec xv, wv;
  float s[kVecElems];
  float local_sumsq = 0.0f;
  if (active) {
    xv.load(params.x + xbase, tid);
    wv.load(params.weight, tid);
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i) {
      const auto [x0, x1] = cast<fp32x2_t>(xv[i]);
      s[2 * i + 0] = x0;
      s[2 * i + 1] = x1;
      local_sumsq += s[2 * i + 0] * s[2 * i + 0] + s[2 * i + 1] * s[2 * i + 1];
    }
  }

  local_sumsq = warp::reduce_sum(local_sumsq);
  __shared__ float s_warp[32];
  const uint32_t lane = tid % kWarpThreads;
  const uint32_t warp = tid / kWarpThreads;
  const uint32_t num_warps = q_threads / kWarpThreads;
  if (lane == 0 && warp < num_warps)
    s_warp[warp] = local_sumsq;
  __syncthreads();
  if (warp == 0) {
    float v = lane < num_warps ? s_warp[lane] : 0.0f;
    v = warp::reduce_sum(v);
    if (lane == 0)
      s_warp[0] = v;
  }
  __syncthreads();
  if (!active)
    return;
  const float rms =
      rsqrtf(s_warp[0] / static_cast<float>(params.hidden) + params.eps);

  float n[kVecElems];
  float local_max = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kVecElems / 2; ++i) {
    const auto [w0, w1] = cast<fp32x2_t>(wv[i]);
    n[2 * i + 0] = s[2 * i + 0] * rms * w0;
    n[2 * i + 1] = s[2 * i + 1] * rms * w1;
    n[2 * i + 0] = cast<fp32_t>(cast<bf16_t>(n[2 * i + 0]));
    n[2 * i + 1] = cast<fp32_t>(cast<bf16_t>(n[2 * i + 1]));
    local_max =
        fmaxf(local_max, fmaxf(fabsf(n[2 * i + 0]), fabsf(n[2 * i + 1])));
  }
  local_max = warp::reduce_max<kThreadsPerGroup>(local_max);
  const uint32_t group_id = tid / kThreadsPerGroup;

  const float amax = fmaxf(1e-10f, local_max);
  const float scale = amax * (1.0f / 448.0f);
  const float biased_exp =
      fminf(fmaxf(ceilf(__log2f(scale)) + 127.0f, 0.0f), 254.0f);
  const float inv_scale = exp2f(127.0f - biased_exp);

  OutVec qv;
#pragma unroll
  for (uint32_t i = 0; i < kVecElems / 2; ++i)
    qv[i] = fp8x2_e4m3_t{
        fp32x2_t{fminf(fmaxf(n[2 * i + 0] * inv_scale, -448.0f), 448.0f),
                 fminf(fmaxf(n[2 * i + 1] * inv_scale, -448.0f), 448.0f)}};
  PDLTriggerSecondary<kUsePDL>();
  qv.store(params.q + base, tid);
  if (tid % kThreadsPerGroup == 0)
    reinterpret_cast<uint8_t *>(
        params.sf)[token * params.num_groups + group_id] =
        static_cast<uint8_t>(biased_exp);
}

// fused_k_norm_rope_flashmla<DType=bf16, 512, 64, PosT, kPageBits, kUsePDL> for
// one token.
template <typename PosT, int32_t kPageBits, bool kUsePDL>
SGL_DEVICE void eb5_hca_k_role(const FusedKNormRopeFlashMLAParams &params,
                               uint32_t work_id) {
  using namespace device;
  using DType = bf16_t;
  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 2;
  constexpr uint32_t kRopeWarp = kFusedKNumWarps - 1;
  constexpr int64_t kPageBytes = host::div_ceil(584ll << kPageBits, 576) * 576;
  static_assert(kHeadDim == kFusedKBlockSize * kVecSize);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float2 = AlignedVector<float, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;

  const auto input_ptr =
      static_cast<const DType *>(params.kv) + work_id * params.kv_stride_batch;
  const auto position = static_cast<int32_t>(
      static_cast<const PosT *>(params.positions)[work_id]);
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  const auto out_loc = params.out_loc[work_id];
  Float2 data, freq;
  {
    __shared__ float partial_sums[kFusedKNumWarps];

    Storage input_vec, weight_vec;
    input_vec.load(input_ptr, tx);
    weight_vec.load(params.kv_weight, tx);
    if (warp_id == kRopeWarp)
      freq.load(freqs_cis, lane_id);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      sum_of_squares += x * x;
    }
    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0)
      partial_sums[warp_id] = warp_sum;
    __syncthreads();
    sum_of_squares = warp::reduce_sum<kFusedKNumWarps>(
        partial_sums[lane_id % kFusedKNumWarps]);
    const auto norm_factor =
        math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      const auto w = cast<float>(weight_vec[i]);
      data[i] = x * norm_factor * w;
    }
  }
  if (out_loc < 0)
    return;

  const int32_t page = out_loc >> kPageBits;
  const int32_t offset = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * 576;

  PDLTriggerSecondary<kUsePDL>();
  if (warp_id == kRopeWarp) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto freq_real = freq[0];
    const auto freq_imag = freq[1];
    data[0] = x_real * freq_real - x_imag * freq_imag;
    data[1] = x_real * freq_imag + x_imag * freq_real;
    const auto result = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
    reinterpret_cast<bf16x2_t *>(value_ptr + 448)[lane_id] = result;
  } else {
    const auto x = data[0];
    const auto y = data[1];
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_raw = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const auto result = pack_fp8(x * inv_scale, y * inv_scale);
    const auto scale_ptr = page_ptr + (576 << kPageBits) + offset * 8;
    reinterpret_cast<fp8x2_e4m3_t *>(value_ptr)[tx] = result;
    if (lane_id == 0)
      static_cast<uint8_t *>(scale_ptr)[warp_id] = scale_ue8m0;
  }
}

// Blocks [0, T) run the q role, [T, 2T) the k role.
template <typename PosT, int32_t kPageBits, bool kUsePDL>
__global__ __launch_bounds__(kFusedKBlockSize, 8) void eb5_hca_qkv_prep_kernel(
    const __grid_constant__ Eb5HcaQkvPrepParams params) {
  // The dependent's griddepcontrol.wait still waits for this grid to finish, so
  // triggering first only lets wq_b become resident and prefetch its weights.
  device::PDLTriggerSecondary<kUsePDL>();
  if (blockIdx.x < params.num_tokens) {
    eb5_hca_q_role<kUsePDL>(params.q, blockIdx.x);
  } else {
    eb5_hca_k_role<PosT, kPageBits, kUsePDL>(params.k,
                                             blockIdx.x - params.num_tokens);
  }
}

template <uint32_t kPageSize, bool kUsePDL> struct Eb5HcaQkvPrepKernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes =
      host::div_ceil(584 * kPageSize, 576) * 576;
  static_assert(std::has_single_bit(kPageSize),
                "kPageSize must be a power of 2");

  static void run(tvm::ffi::TensorView qkv_a, tvm::ffi::TensorView q_weight,
                  tvm::ffi::TensorView q_out, tvm::ffi::TensorView sf_out,
                  double q_eps, tvm::ffi::TensorView kv_weight,
                  tvm::ffi::TensorView freqs_cis,
                  tvm::ffi::TensorView positions, tvm::ffi::TensorView out_loc,
                  tvm::ffi::TensorView kvcache, double kv_eps,
                  int64_t q_lora_rank) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto M = SymbolicSize{"num_tokens"};
    auto W = SymbolicSize{"qkv_width"};
    auto S = SymbolicSize{"qkv_row_stride"};
    TensorMatcher({M, W})
        .with_strides({S, int64_t{1}})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(qkv_a);
    RuntimeCheck(W.unwrap() == q_lora_rank + 512,
                 "qkv_a must be [T, q_lora_rank + 512]");
    RuntimeCheck(q_lora_rank % 32 == 0 && q_lora_rank / 8 <= kFusedKBlockSize,
                 "q_lora_rank/8 must fit one block");
    TensorMatcher({q_lora_rank})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_weight);
    TensorMatcher({M, q_lora_rank})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(q_out);
    TensorMatcher({M, q_lora_rank / 32})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(sf_out);
    TensorMatcher({int64_t{512}})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(kv_weight);
    TensorMatcher({-1, int64_t{64}})
        .with_dtype<float>()
        .with_device(device)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({M})
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device)
        .verify(positions);
    TensorMatcher({M}).with_dtype<int32_t>().with_device(device).verify(
        out_loc);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(kvcache);

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    if (num_tokens == 0)
      return;
    const auto base = static_cast<const bf16_t *>(qkv_a.data_ptr());
    const auto hidden = static_cast<uint32_t>(q_lora_rank);
    const auto params = Eb5HcaQkvPrepParams{
        .q =
            AddRmsNormQuantParams{
                .x = base,
                .residual = nullptr,
                .weight = static_cast<const bf16_t *>(q_weight.data_ptr()),
                .residual_out = nullptr,
                .norm_out = nullptr,
                .q = static_cast<fp8_e4m3_t *>(q_out.data_ptr()),
                .sf = static_cast<int32_t *>(sf_out.data_ptr()),
                .eps = static_cast<float>(q_eps),
                .hidden = hidden,
                .num_groups = hidden / 32u,
                .scale_row_stride_int32 = 0,
                .x_row_stride = static_cast<uint32_t>(S.unwrap()),
            },
        .k =
            FusedKNormRopeFlashMLAParams{
                .kv = base + q_lora_rank,
                .kv_weight = kv_weight.data_ptr(),
                .freqs_cis = static_cast<const float *>(freqs_cis.data_ptr()),
                .positions = positions.data_ptr(),
                .out_loc = static_cast<const int32_t *>(out_loc.data_ptr()),
                .kvcache = static_cast<uint8_t *>(kvcache.data_ptr()),
                .kv_stride_batch = S.unwrap(),
                .batch_size = num_tokens,
                .eps = static_cast<float>(kv_eps),
            },
        .num_tokens = num_tokens,
    };
    const auto k =
        pos_dtype.is_type<int32_t>()
            ? eb5_hca_qkv_prep_kernel<int32_t, kLogPageSize, kUsePDL>
            : eb5_hca_qkv_prep_kernel<int64_t, kLogPageSize, kUsePDL>;
    LaunchKernel(2 * num_tokens, kFusedKBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(k, params);
  }
};

} // namespace sglang

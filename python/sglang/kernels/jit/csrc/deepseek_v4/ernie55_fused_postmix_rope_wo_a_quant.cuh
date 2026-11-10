#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

constexpr float LOCAL_ABSMAX_FLOOR = 1e-10f;
constexpr uint32_t GROUP_SIZE = 128;
constexpr uint32_t HEADS = 64;
constexpr uint32_t HEAD_DIM = 512;
constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t GROUPS_PER_HEAD = HEAD_DIM / GROUP_SIZE;
constexpr uint32_t THREADS_PER_HEAD = 8;
constexpr uint32_t COLS_PER_LANE = GROUP_SIZE / THREADS_PER_HEAD;
constexpr uint32_t HEAD_SPLIT = 1;
constexpr uint32_t HEADS_PER_CTA = HEADS / HEAD_SPLIT;
constexpr uint32_t HEADS_PER_THREAD = 4;
constexpr uint32_t NUM_SUBWARPS = HEADS_PER_CTA / HEADS_PER_THREAD;
constexpr uint32_t BLOCK_THREADS = NUM_SUBWARPS * THREADS_PER_HEAD;

template <int THREADS_PER_SUBWARP> SGL_DEVICE float GroupReduceMax(float val) {
  static_assert((THREADS_PER_SUBWARP & (THREADS_PER_SUBWARP - 1)) == 0 &&
                    THREADS_PER_SUBWARP <= 16 && THREADS_PER_SUBWARP >= 1,
                "THREADS_PER_SUBWARP must be 1, 2, 4, 8, or 16");
  constexpr device::warp::mask_t kSub =
      (device::warp::mask_t{1} << THREADS_PER_SUBWARP) - 1;
  const device::warp::mask_t mask =
      kSub << (THREADS_PER_SUBWARP *
               ((threadIdx.x % 32) / THREADS_PER_SUBWARP));
  return device::warp::reduce_max<THREADS_PER_SUBWARP>(val, mask);
}

template <typename T>
SGL_DEVICE void hmma_m16n8k16(float (&d)[4], const uint32_t (&A)[4],
                              const uint32_t (&B)[2]);

template <>
SGL_DEVICE void hmma_m16n8k16<bf16_t>(float (&d)[4], const uint32_t (&A)[4],
                                      const uint32_t (&B)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                 "r"(B[1]));
}

template <>
SGL_DEVICE void hmma_m16n8k16<fp16_t>(float (&d)[4], const uint32_t (&A)[4],
                                      const uint32_t (&B)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                 "r"(B[1]));
}

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *);

SGL_DEVICE void ldsm_x4(const void *smem, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(__nvvm_get_smem_pointer(const_cast<void *>(smem))));
}
SGL_DEVICE void ldsm_x2_trans(const void *smem, uint32_t (&r)[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(__nvvm_get_smem_pointer(const_cast<void *>(smem))));
}

template <typename T, bool kUsePDL>
__global__ void __launch_bounds__(BLOCK_THREADS)
    fp8_wo_a_fused_postmix_rope_group_major_quant_ue8m0_kernel(
        const T *__restrict__ input, const T *__restrict__ postmix,
        fp8_e4m3_t *__restrict__ output_q, int32_t *__restrict__ output_s,
        const float *__restrict__ freqs_cis,
        const int64_t *__restrict__ positions, int out_head_offset, int n_out,
        int scale_S, int scale_A, int scale_P) {
  constexpr uint32_t SMEM_PAD = 8;
  __shared__ alignas(16) T sIn[HEADS][GROUP_SIZE + SMEM_PAD];
  __shared__ alignas(16) T sM[HEADS_PER_CTA][HEADS + SMEM_PAD];
  __shared__ alignas(16) T sC[HEADS_PER_CTA][GROUP_SIZE + SMEM_PAD];

  const uint32_t token = blockIdx.x;
  const uint32_t cblk = blockIdx.y;
  const uint32_t h_base = blockIdx.z * HEADS_PER_CTA;
  const uint32_t tid = threadIdx.x;

  constexpr uint32_t VECM = sizeof(int4) / sizeof(T);
#pragma unroll
  for (uint32_t v = tid; v < (HEADS_PER_CTA * HEADS) / VECM;
       v += BLOCK_THREADS) {
    const uint32_t row = (v * VECM) / HEADS;
    const uint32_t col = (v * VECM) % HEADS;
    *reinterpret_cast<int4 *>(&sM[row][col]) =
        *reinterpret_cast<const int4 *>(postmix + (h_base + row) * HEADS + col);
  }

  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();

  constexpr uint32_t VEC = sizeof(int4) / sizeof(T);
  const int64_t in_base = static_cast<int64_t>(token) * (HEADS * HEAD_DIM) +
                          static_cast<int64_t>(cblk) * GROUP_SIZE;
#pragma unroll
  for (uint32_t v = tid; v < (HEADS * GROUP_SIZE) / VEC; v += BLOCK_THREADS) {
    const uint32_t h = (v * VEC) / GROUP_SIZE;
    const uint32_t c = (v * VEC) % GROUP_SIZE;
    const int64_t goff = in_base + static_cast<int64_t>(h) * HEAD_DIM + c;
    *reinterpret_cast<int4 *>(&sIn[h][c]) =
        *reinterpret_cast<const int4 *>(input + goff);
  }
  __syncthreads();

  if (cblk == GROUPS_PER_HEAD - 1) {
    constexpr uint32_t ROPE_LOCAL = GROUP_SIZE - ROPE_DIM;
    const int32_t pos = static_cast<int32_t>(positions[token]);
    const float *fr = freqs_cis + static_cast<int64_t>(pos) * ROPE_DIM;
#pragma unroll
    for (uint32_t i = tid; i < HEADS * (ROPE_DIM / 2); i += BLOCK_THREADS) {
      const uint32_t h = i / (ROPE_DIM / 2);
      const uint32_t p = i % (ROPE_DIM / 2);
      const uint32_t col = ROPE_LOCAL + 2 * p;
      const float x_r = static_cast<float>(sIn[h][col]);
      const float x_i = static_cast<float>(sIn[h][col + 1]);
      const float c0 = fr[2 * p], s0 = fr[2 * p + 1];
      sIn[h][col] = static_cast<T>(x_r * c0 + x_i * s0);
      sIn[h][col + 1] = static_cast<T>(x_i * c0 - x_r * s0);
    }
    __syncthreads();
  }

  static_assert(HEADS_PER_THREAD == 4 && GROUP_SIZE == 128 && HEADS % 16 == 0 &&
                    HEADS_PER_CTA % 16 == 0 &&
                    BLOCK_THREADS / 32 == HEADS_PER_CTA / 16,
                "MMA path expects 4 warps == HEADS_PER_CTA/16 M-tiles");
  {
    const uint32_t warp = tid / 32;
    const uint32_t lane = tid % 32;
    const uint32_t gr = lane / 4;
    const uint32_t mrow = warp * 16;
    uint32_t a_reg[HEADS / 16][4];
#pragma unroll
    for (uint32_t kt = 0; kt < HEADS / 16; ++kt)
      ldsm_x4(&sM[mrow + (lane % 16)][kt * 16 + (lane / 16) * 8], a_reg[kt]);
#pragma unroll
    for (uint32_t nt = 0; nt < GROUP_SIZE / 8; ++nt) {
      const uint32_t ncol = nt * 8;
      float cacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (uint32_t kt = 0; kt < HEADS / 16; ++kt) {
        uint32_t b_reg[2];
        ldsm_x2_trans(&sIn[kt * 16 + (lane % 16)][ncol], b_reg);
        hmma_m16n8k16<T>(cacc, a_reg[kt], b_reg);
      }
      const uint32_t cc = (lane % 4) * 2;
      device::AlignedVector<T, 2> p_lo, p_hi;
      p_lo[0] = static_cast<T>(cacc[0]);
      p_lo[1] = static_cast<T>(cacc[1]);
      p_hi[0] = static_cast<T>(cacc[2]);
      p_hi[1] = static_cast<T>(cacc[3]);
      p_lo.store(&sC[mrow + gr][ncol + cc], 0);
      p_hi.store(&sC[mrow + gr + 8][ncol + cc], 0);
    }
  }
  __syncthreads();

  const uint32_t sg = tid / THREADS_PER_HEAD;
  const uint32_t lane = tid % THREADS_PER_HEAD;
  const uint32_t col0 = lane * COLS_PER_LANE;

  float acc[HEADS_PER_THREAD][COLS_PER_LANE];
  using vec_t = device::AlignedVector<T, COLS_PER_LANE / 2>;
#pragma unroll
  for (uint32_t r = 0; r < HEADS_PER_THREAD; ++r) {
    const uint32_t h_local = sg + r * NUM_SUBWARPS;
    vec_t v0, v1;
    v0.load(&sC[h_local][col0], 0);
    v1.load(&sC[h_local][col0 + COLS_PER_LANE / 2], 0);
#pragma unroll
    for (uint32_t k = 0; k < COLS_PER_LANE / 2; ++k) {
      acc[r][k] = static_cast<float>(v0[k]);
      acc[r][k + COLS_PER_LANE / 2] = static_cast<float>(v1[k]);
    }
  }

  constexpr float kFp8MaxInv = 1.0f / kFP8E4M3Max;
#pragma unroll
  for (uint32_t r = 0; r < HEADS_PER_THREAD; ++r) {
    const uint32_t h_out = h_base + sg + r * NUM_SUBWARPS;
    const int h_local = static_cast<int>(h_out) - out_head_offset;
    if (h_local < 0 || h_local >= n_out)
      continue;
#pragma unroll
    for (uint32_t k = 0; k < COLS_PER_LANE; ++k)
      acc[r][k] = static_cast<float>(static_cast<T>(acc[r][k]));

    float local_absmax = LOCAL_ABSMAX_FLOOR;
#pragma unroll
    for (uint32_t k = 0; k < COLS_PER_LANE; ++k)
      local_absmax = fmaxf(local_absmax, fabsf(acc[r][k]));
    local_absmax = GroupReduceMax<THREADS_PER_HEAD>(local_absmax);

    const int32_t scale_ue8m0 = cast_to_ue8m0(local_absmax * kFp8MaxInv);
    const float y_scale = inv_scale_ue8m0(scale_ue8m0);

    int4 out_buf;
    auto *out_ptr = reinterpret_cast<fp8x2_e4m3_t *>(&out_buf);
#pragma unroll
    for (uint32_t k = 0; k < COLS_PER_LANE; k += 2)
      out_ptr[k / 2] = pack_fp8(acc[r][k] * y_scale, acc[r][k + 1] * y_scale);
    const int64_t oq_off =
        static_cast<int64_t>(token) * (static_cast<int64_t>(n_out) * HEAD_DIM) +
        static_cast<int64_t>(h_local) * HEAD_DIM +
        static_cast<int64_t>(cblk) * GROUP_SIZE + col0;
    *reinterpret_cast<int4 *>(output_q + oq_off) = out_buf;

    if (lane == 0) {
      const uint32_t gg =
          static_cast<uint32_t>(h_local) * GROUPS_PER_HEAD + cblk;
      const uint32_t g = gg / scale_S;
      const uint32_t s = gg % scale_S;
      uint8_t *s_bytes = reinterpret_cast<uint8_t *>(output_s);
      const int64_t elem =
          (static_cast<int64_t>(g) * scale_P + (s >> 2)) * scale_A + token;
      s_bytes[elem * 4 + (s & 3)] = static_cast<uint8_t>(scale_ue8m0 & 0xFF);
    }
  }
}

template <typename T, bool kUsePDL>
struct FP8WoAFusedPostmixRopeGroupMajorQuantUE8M0Kernel {
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView postmix,
                  tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s,
                  tvm::ffi::TensorView freqs_cis,
                  tvm::ffi::TensorView positions, int num_outer_groups,
                  int out_head_offset, int n_out) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto TSize = SymbolicSize{"num_tokens"};
    auto RLSize = SymbolicSize{"max_seq_len"};

    TensorMatcher(
        {TSize, static_cast<int64_t>(HEADS), static_cast<int64_t>(HEAD_DIM)})
        .with_dtype<T>()
        .with_device(device)
        .verify(input);
    TensorMatcher({static_cast<int64_t>(HEADS), static_cast<int64_t>(HEADS)})
        .with_dtype<T>()
        .with_device(device)
        .verify(postmix);
    RuntimeCheck(n_out > 0 && n_out <= static_cast<int>(HEADS) &&
                     out_head_offset >= 0 &&
                     out_head_offset + n_out <= static_cast<int>(HEADS),
                 "out_head_offset/n_out must select a sub-range of HEADS");
    TensorMatcher({TSize, static_cast<int64_t>(n_out) * HEAD_DIM})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output_q);
    TensorMatcher({RLSize, static_cast<int64_t>(ROPE_DIM)})
        .with_dtype<float>()
        .with_device(device)
        .verify(freqs_cis);
    TensorMatcher({TSize}).with_dtype<int64_t>().with_device(device).verify(
        positions);

    const auto num_tokens = TSize.unwrap();

    RuntimeCheck(num_outer_groups > 0 &&
                     (n_out * HEAD_DIM) % num_outer_groups == 0,
                 "num_outer_groups must divide n_out*HEAD_DIM");
    const int D = (n_out * HEAD_DIM) / num_outer_groups;
    RuntimeCheck(D % GROUP_SIZE == 0,
                 "D (=HEADS*HEAD_DIM/G) must be divisible by 128");
    const int S = D / GROUP_SIZE;
    RuntimeCheck(S % 4 == 0,
                 "S (=D/128) must be a multiple of 4 for int32-packed UE8M0");
    const int A = (static_cast<int>(num_tokens) + 3) / 4 * 4;
    const int P = (S + 3) / 4;

    TensorMatcher({static_cast<int64_t>(num_outer_groups),
                   static_cast<int64_t>(P), static_cast<int64_t>(A)})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(output_s);

    RuntimeCheck(reinterpret_cast<uintptr_t>(input.data_ptr()) % sizeof(int4) ==
                     0,
                 "input base pointer must be 16-byte aligned");

    if (num_tokens == 0)
      return;

    const dim3 grid(static_cast<uint32_t>(num_tokens), GROUPS_PER_HEAD,
                    HEAD_SPLIT);
    const dim3 block(BLOCK_THREADS);
    host::LaunchKernel(grid, block, device.unwrap())
        .enable_pdl(kUsePDL)(
            fp8_wo_a_fused_postmix_rope_group_major_quant_ue8m0_kernel<T,
                                                                       kUsePDL>,
            static_cast<const T *>(input.data_ptr()),
            static_cast<const T *>(postmix.data_ptr()),
            static_cast<fp8_e4m3_t *>(output_q.data_ptr()),
            static_cast<int32_t *>(output_s.data_ptr()),
            static_cast<const float *>(freqs_cis.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()), out_head_offset,
            n_out, S, A, P);
  }
};
} // namespace sglang

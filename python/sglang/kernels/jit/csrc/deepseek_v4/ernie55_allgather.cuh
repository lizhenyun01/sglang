
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <utility>

#include "../../../../kernels/jit/csrc/distributed/ll_all_reduce_push.cuh"

namespace sglang {

using device::distributed::Counter;

template <uint32_t kWorldSize> struct Eb5AllGatherPushParams {
  bf16_t *__restrict__ local;
  bf16_t *__restrict__ out;
  uint8_t *push_workspaces[kWorldSize];
  Counter *push_counter;
  int64_t push_buffer_stride;
  uint32_t rank;
  uint32_t num_vecs;
};

template <uint32_t kWorldSize, bool kUsePDL>
__global__ __launch_bounds__(256, 1) void eb5_allgather_push_kernel(
    const __grid_constant__ Eb5AllGatherPushParams<kWorldSize> params) {
  using namespace device;
  using vec_t = AlignedVector<bf16x2_t, 4>;

  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  const auto global_tid = bx * blockDim.x + tx;
  const auto num_threads = blockDim.x * gridDim.x;
  const auto num_vecs = params.num_vecs;
  const auto stride = params.push_buffer_stride;
  const auto r = params.rank;

  const auto phase = params.push_counter[bx].get() % 2;
  const auto phase_stride = static_cast<int64_t>(phase) * stride * kWorldSize;
  const int64_t my_slot_off = static_cast<int64_t>(r) * stride + phase_stride;
  void *push_buf[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    push_buf[i] = params.push_workspaces[i] + my_slot_off;
  }

  PDLWaitPrimary<kUsePDL>();
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    ld_global_16B(vec, params.local, vid);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      clear_pos_zero(vec[j].x);
      clear_pos_zero(vec[j].y);
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      st_relaxed_16B(vec, push_buf[i], vid);
    }
  }
  PDLTriggerSecondary<kUsePDL>();

  vec_t zero_vec;
  {
    const auto z = get_pos_zero<bf16_t>();
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j)
      zero_vec[j] = bf16x2_t{z, z};
  }
#pragma unroll
  for (uint32_t s = 0; s < kWorldSize; ++s) {
    void *poll_buf = params.push_workspaces[r] +
                     static_cast<int64_t>(s) * stride + phase_stride;
    bf16_t *out_slot = params.out + static_cast<int64_t>(s) * num_vecs * 8;
    for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
      vec_t vec;
      do {
        bool has_zero = false;
        ld_relaxed_16B(vec, poll_buf, vid);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          has_zero |= is_pos_zero(vec[j].x);
          has_zero |= is_pos_zero(vec[j].y);
        }
        if (!has_zero)
          break;
      } while (true);
      st_global_16B(vec, out_slot, vid);
      st_global_16B(zero_vec, poll_buf, vid);
    }
  }

  __syncthreads();
  if (tx == 0)
    params.push_counter[bx].inc(1);
}

template <uint32_t kWorldSize, bool kUsePDL> struct Eb5AllGatherPushKernel {
  using CommunicatorRef = host::distributed::CommunicatorRef;

  static void run(CommunicatorRef ref, const tvm::ffi::TensorView local,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    const auto &comm = *ref.get();
    const auto &push = comm.get_push_obj();
    RuntimeCheck(comm.get_world_size() == kWorldSize, "world size mismatch");
    RuntimeCheck(local.IsContiguous(), "local must be contiguous");
    RuntimeCheck(out.IsContiguous(), "out must be contiguous");
    RuntimeCheck(is_type<bf16_t>(local.dtype()), "local must be bf16");
    RuntimeCheck(is_type<bf16_t>(out.dtype()), "out must be bf16");
    RuntimeCheck(local.device().device_type == kDLCUDA,
                 "local must be on CUDA");
    RuntimeCheck(out.device().device_type == kDLCUDA, "out must be on CUDA");
    RuntimeCheck(std::bit_cast<intptr_t>(local.data_ptr()) % 16 == 0,
                 "local must be 16B aligned");
    RuntimeCheck(std::bit_cast<intptr_t>(out.data_ptr()) % 16 == 0,
                 "out must be 16B aligned");

    const int64_t numel = local.numel();
    if (numel == 0)
      return;
    RuntimeCheck(out.numel() == numel * static_cast<int64_t>(kWorldSize),
                 "out.numel() must be world_size * local.numel()");
    const int64_t nbytes = numel * static_cast<int64_t>(sizeof(bf16_t));
    RuntimeCheck(nbytes % 16 == 0, "shard bytes must be a multiple of 16");
    RuntimeCheck(nbytes <= push.slot_bytes,
                 "shard exceeds push workspace slot size");

    Eb5AllGatherPushParams<kWorldSize> params{};
    params.local = static_cast<bf16_t *>(local.data_ptr());
    params.out = static_cast<bf16_t *>(out.data_ptr());
    params.push_counter = push.counter;
    params.push_buffer_stride = push.slot_bytes;
    params.rank = comm.get_rank();
    params.num_vecs = static_cast<uint32_t>(numel / 8);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      params.push_workspaces[i] = push.workspaces[i];
    }

    const auto stream = LaunchKernel::resolve_device(local.device());
    const uint32_t num_blocks = push.num_blocks;
    const uint32_t block_size =
        std::min<uint32_t>(choose_block_size(params.num_vecs), 256u);
    LaunchKernel(num_blocks, block_size, stream)
        .enable_pdl(kUsePDL)(eb5_allgather_push_kernel<kWorldSize, kUsePDL>,
                             params);
  }
};

template <uint32_t kWorldSize> struct Eb5AllGatherPushGlobalParams {
  bf16_t *__restrict__ local;
  bf16_t *__restrict__ out;
  uint8_t *push_workspaces[kWorldSize];
  Counter *push_counter;
  int64_t push_buffer_stride;
  uint32_t rank;
  uint32_t num_vecs;
  uint32_t slot_row_vecs;
  uint32_t src_row_vecs;
  uint32_t hd_vecs;
  uint32_t n_local;
  uint32_t num_heads;
};

template <uint32_t kWorldSize, bool kUsePDL>
__global__ __launch_bounds__(256, 1) void eb5_allgather_push_global_kernel(
    const __grid_constant__ Eb5AllGatherPushGlobalParams<kWorldSize> params) {
  using namespace device;
  using vec_t = AlignedVector<bf16x2_t, 4>;

  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  const auto global_tid = bx * blockDim.x + tx;
  const auto num_threads = blockDim.x * gridDim.x;
  const auto num_vecs = params.num_vecs;
  const auto stride = params.push_buffer_stride;
  const auto r = params.rank;
  const auto slot_row_vecs = params.slot_row_vecs;
  const auto src_row_vecs = params.src_row_vecs;
  const auto hd_vecs = params.hd_vecs;
  const auto n_local = params.n_local;
  const auto num_heads = params.num_heads;

  const auto phase = params.push_counter[bx].get() % 2;
  const auto phase_stride = static_cast<int64_t>(phase) * stride * kWorldSize;
  const int64_t my_slot_off = static_cast<int64_t>(r) * stride + phase_stride;
  void *push_buf[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    push_buf[i] = params.push_workspaces[i] + my_slot_off;
  }

  PDLWaitPrimary<kUsePDL>();
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    const uint32_t token = vid / slot_row_vecs;
    const uint32_t within = vid - token * slot_row_vecs;
    const int64_t src_vid = static_cast<int64_t>(token) * src_row_vecs + within;
    vec_t vec;
    ld_global_16B(vec, params.local, src_vid);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      clear_pos_zero(vec[j].x);
      clear_pos_zero(vec[j].y);
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      st_relaxed_16B(vec, push_buf[i], vid);
    }
  }
  PDLTriggerSecondary<kUsePDL>();
  vec_t zero_vec;
  {
    const auto z = get_pos_zero<bf16_t>();
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j)
      zero_vec[j] = bf16x2_t{z, z};
  }
#pragma unroll
  for (uint32_t s = 0; s < kWorldSize; ++s) {
    void *poll_buf = params.push_workspaces[r] +
                     static_cast<int64_t>(s) * stride + phase_stride;
    const uint32_t head_base = s * n_local;
    for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
      vec_t vec;
      do {
        bool has_zero = false;
        ld_relaxed_16B(vec, poll_buf, vid);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          has_zero |= is_pos_zero(vec[j].x);
          has_zero |= is_pos_zero(vec[j].y);
        }
        if (!has_zero)
          break;
      } while (true);
      const uint32_t token = vid / slot_row_vecs;
      const uint32_t within = vid - token * slot_row_vecs;
      const uint32_t h = within / hd_vecs;
      const uint32_t dvec = within - h * hd_vecs;
      const int64_t out_vid =
          (static_cast<int64_t>(token) * num_heads + head_base + h) * hd_vecs +
          dvec;
      st_global_16B(vec, params.out, out_vid);
      st_global_16B(zero_vec, poll_buf, vid);
    }
  }

  __syncthreads();
  if (tx == 0)
    params.push_counter[bx].inc(1);
}
template <uint32_t kWorldSize, bool kUsePDL>
struct Eb5AllGatherPushGlobalKernel {
  using CommunicatorRef = host::distributed::CommunicatorRef;

  static void run(CommunicatorRef ref, const tvm::ffi::TensorView local,
                  const tvm::ffi::TensorView out, int64_t num_tokens,
                  int64_t n_local, int64_t head_dim, int64_t num_heads,
                  int64_t src_row_stride_elems) {
    using namespace host;
    const auto &comm = *ref.get();
    const auto &push = comm.get_push_obj();
    RuntimeCheck(comm.get_world_size() == kWorldSize, "world size mismatch");
    RuntimeCheck(is_type<bf16_t>(local.dtype()), "local must be bf16");
    RuntimeCheck(is_type<bf16_t>(out.dtype()), "out must be bf16");
    RuntimeCheck(out.IsContiguous(), "out must be contiguous");
    RuntimeCheck(local.device().device_type == kDLCUDA,
                 "local must be on CUDA");
    RuntimeCheck(out.device().device_type == kDLCUDA, "out must be on CUDA");
    RuntimeCheck(std::bit_cast<intptr_t>(local.data_ptr()) % 16 == 0,
                 "local must be 16B aligned");
    RuntimeCheck(std::bit_cast<intptr_t>(out.data_ptr()) % 16 == 0,
                 "out must be 16B aligned");
    RuntimeCheck(head_dim % 8 == 0, "head_dim must be a multiple of 8");
    RuntimeCheck(src_row_stride_elems % 8 == 0,
                 "src row stride must be a multiple of 8");
    RuntimeCheck(src_row_stride_elems >= n_local * head_dim,
                 "src row stride too small");
    RuntimeCheck(n_local * kWorldSize == num_heads,
                 "n_local * world_size must equal num_heads");
    RuntimeCheck(out.numel() == num_tokens * num_heads * head_dim,
                 "out.numel() must be num_tokens * num_heads * head_dim");

    const int64_t shard_numel = num_tokens * n_local * head_dim;
    if (shard_numel == 0)
      return;
    const int64_t nbytes = shard_numel * static_cast<int64_t>(sizeof(bf16_t));
    RuntimeCheck(nbytes % 16 == 0, "shard bytes must be a multiple of 16");
    RuntimeCheck(nbytes <= push.slot_bytes,
                 "shard exceeds push workspace slot size");

    Eb5AllGatherPushGlobalParams<kWorldSize> params{};
    params.local = static_cast<bf16_t *>(local.data_ptr());
    params.out = static_cast<bf16_t *>(out.data_ptr());
    params.push_counter = push.counter;
    params.push_buffer_stride = push.slot_bytes;
    params.rank = comm.get_rank();
    const int64_t hd_vecs = head_dim / 8;
    const int64_t slot_row_vecs = n_local * hd_vecs;
    params.num_vecs = static_cast<uint32_t>(num_tokens * slot_row_vecs);
    params.slot_row_vecs = static_cast<uint32_t>(slot_row_vecs);
    params.src_row_vecs = static_cast<uint32_t>(src_row_stride_elems / 8);
    params.hd_vecs = static_cast<uint32_t>(hd_vecs);
    params.n_local = static_cast<uint32_t>(n_local);
    params.num_heads = static_cast<uint32_t>(num_heads);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      params.push_workspaces[i] = push.workspaces[i];
    }

    const auto stream = LaunchKernel::resolve_device(local.device());
    const uint32_t num_blocks = push.num_blocks;
    const uint32_t block_size =
        std::min<uint32_t>(choose_block_size(params.num_vecs), 256u);
    LaunchKernel(num_blocks, block_size, stream)
        .enable_pdl(kUsePDL)(
            eb5_allgather_push_global_kernel<kWorldSize, kUsePDL>, params);
  }
};

template <uint32_t kWorldSize> struct Eb5CombinePushGlobalParams {
  const float *__restrict__ part;
  const float *__restrict__ maxv;
  const float *__restrict__ sumv;
  const float *__restrict__ sink;
  bf16_t *__restrict__ out;
  uint8_t *push_workspaces[kWorldSize];
  Counter *push_counter;
  int64_t push_buffer_stride;
  uint32_t rank;
  uint32_t num_vecs;
  uint32_t slot_row_vecs;
  uint32_t hd_vecs;
  uint32_t nt;
  uint32_t n_local;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t split;
  // HCA late-tile decoupling (swapab_decode split partial), off when sync is
  // null: a token whose last c128 tile is late (same test as partial_gluon
  // TILE_MODE) waits for sync[8 + token] == seq + 1, released by the
  // side-stream mode-2 partial. seq is the u64 at sync + 2 shifted right by 20:
  // every finished block adds to it, the adds of one launch sum to 1 << 20 and
  // any subset of them to less, so seq stays put until the whole grid is done.
  int32_t *sync;
  const int32_t *el;
  const int64_t *pos;
  uint32_t tokens_per_req;
  uint32_t c_ratio;
};

__device__ __forceinline__ int32_t eb5_ld_acquire(const int32_t *p) {
  int32_t v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ int32_t eb5_late_seq(const int32_t *sync) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
               : "=l"(v)
               : "l"(sync + 2)
               : "memory");
  return static_cast<int32_t>(v >> 20);
}

template <uint32_t kWorldSize, bool kUsePDL, bool kHasSink>
__global__ __launch_bounds__(768, 1) void eb5_combine_push_global_kernel(
    const __grid_constant__ Eb5CombinePushGlobalParams<kWorldSize> params) {
  using namespace device;
  using vec_t = AlignedVector<bf16x2_t, 4>;
  const auto tx = threadIdx.x, bx = blockIdx.x;
  const auto global_tid = bx * blockDim.x + tx;
  const auto num_threads = blockDim.x * gridDim.x;
  const auto num_vecs = params.num_vecs;
  const auto stride = params.push_buffer_stride;
  const auto r = params.rank, slot_row_vecs = params.slot_row_vecs;
  const auto hd_vecs = params.hd_vecs, nt = params.nt, n_local = params.n_local;
  const auto num_heads = params.num_heads, head_dim = params.head_dim;

  const auto phase = params.push_counter[bx].get() % 2;
  const auto phase_stride = static_cast<int64_t>(phase) * stride * kWorldSize;
  const int64_t my_slot_off = static_cast<int64_t>(r) * stride + phase_stride;
  void *push_buf[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i)
    push_buf[i] = params.push_workspaces[i] + my_slot_off;

  PDLWaitPrimary<kUsePDL>();
  int32_t *const sync = params.sync;
  constexpr uint32_t kLaneTiles = 8;
  const uint32_t split = params.split;
  const uint32_t half = global_tid & 1u;
  const uint32_t sub = (global_tid >> 1) & (split - 1);
  const uint32_t lanes_per_vec = 2 * split;
  const uint32_t vecs_per_iter = num_threads / lanes_per_vec;
  const uint32_t chunk = kLaneTiles * split;
  const uint32_t tile_stride4 = n_local * head_dim / 4;
  for (uint32_t vbase = 0; vbase < num_vecs; vbase += vecs_per_iter) {
    const uint32_t vid = vbase + global_tid / lanes_per_vec;
    const bool valid = vid < num_vecs;
    const uint32_t tend = valid ? nt : 0u;
    const uint32_t token = vid / slot_row_vecs;
    const uint32_t within = vid - token * slot_row_vecs;
    const uint32_t h = within / hd_vecs;
    const uint32_t dvec = within - h * hd_vecs;
    const uint32_t hbase = token * nt * n_local + h;
    // A late token's tile is written while this grid runs, so once it waited
    // its loads must not come from the non-coherent path (a neighbour token may
    // have cached the line).
    bool late = false;
#define EB5_LD(p) (late ? __ldcg(p) : __ldg(p))
    const float4 *pbase =
        reinterpret_cast<const float4 *>(
            params.part +
            (static_cast<int64_t>(token) * nt * n_local + h) * head_dim +
            static_cast<int64_t>(dvec) * 8) +
        half;
    float4 pa[kLaneTiles];
    float m;
    const auto first_pass = [&]() {
      m = -INFINITY;
      for (uint32_t c = 0; c < tend; c += chunk) {
#pragma unroll
        for (uint32_t j = 0; j < kLaneTiles; ++j) {
          const uint32_t t = c + sub + j * split;
          if (t < tend) {
            const uint32_t hidx = hbase + t * n_local;
            const float s = EB5_LD(params.sumv + hidx);
            const float mx = EB5_LD(params.maxv + hidx);
            m = fmaxf(m, (s > 0.f) ? mx : -INFINITY);
            if (c == 0)
              pa[j] = EB5_LD(pbase + static_cast<int64_t>(t) * tile_stride4);
          }
        }
      }
    };
    // Speculate the token is not late so the el/pos test overlaps the loads; a
    // late token (rare) waits for its flag and reloads.
    first_pass();
    if (sync != nullptr && valid) {
      const int64_t el = params.el[token];
      if (params.pos[token] + 1 - el * params.c_ratio <=
          static_cast<int64_t>(token % params.tokens_per_req)) {
        const int32_t want = eb5_late_seq(sync) + 1;
        while (eb5_ld_acquire(sync + 8 + token) != want) {
        }
        late = true;
        first_pass();
      }
    }
    for (uint32_t o = 2; o < lanes_per_vec; o <<= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float sink = (kHasSink && valid) ? params.sink[h] : -INFINITY;
    m = fmaxf(m, sink);
    if (isinf(m))
      m = 0.f;
    float denom = (kHasSink && sub == 0) ? __expf(sink - m) : 0.f;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t c = 0; c < tend; c += chunk) {
      if (c != 0) {
#pragma unroll
        for (uint32_t j = 0; j < kLaneTiles; ++j) {
          const uint32_t t = c + sub + j * split;
          if (t < tend)
            pa[j] = EB5_LD(pbase + static_cast<int64_t>(t) * tile_stride4);
        }
      }
#pragma unroll
      for (uint32_t j = 0; j < kLaneTiles; ++j) {
        const uint32_t t = c + sub + j * split;
        if (t < tend) {
          const uint32_t hidx = hbase + t * n_local;
          const float s = EB5_LD(params.sumv + hidx);
          const float mx = (s > 0.f) ? EB5_LD(params.maxv + hidx) : -INFINITY;
          const float f = __expf(mx - m);
          denom += s * f;
          acc[0] += pa[j].x * f;
          acc[1] += pa[j].y * f;
          acc[2] += pa[j].z * f;
          acc[3] += pa[j].w * f;
        }
      }
    }
    for (uint32_t o = 2; o < lanes_per_vec; o <<= 1) {
      denom += __shfl_xor_sync(0xffffffffu, denom, o);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o);
    }
    float hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      hi[j] = __shfl_xor_sync(0xffffffffu, acc[j], 1);
    if (!valid || sub != 0 || half != 0)
      continue;
    const bool keep = (denom > 0.f) && !(kHasSink && isinf(sink) && sink > 0.f);
    const float inv = keep ? (1.f / denom) : 0.f;
    vec_t vec;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      vec[j] = bf16x2_t{__float2bfloat16(acc[2 * j] * inv),
                        __float2bfloat16(acc[2 * j + 1] * inv)};
      vec[j + 2] = bf16x2_t{__float2bfloat16(hi[2 * j] * inv),
                            __float2bfloat16(hi[2 * j + 1] * inv)};
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      clear_pos_zero(vec[j].x);
      clear_pos_zero(vec[j].y);
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i)
      st_relaxed_16B(vec, push_buf[i], vid);
#undef EB5_LD
  }
  PDLTriggerSecondary<kUsePDL>();
  vec_t zero_vec;
  {
    const auto z = get_pos_zero<bf16_t>();
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j)
      zero_vec[j] = bf16x2_t{z, z};
  }
  void *poll_buf[kWorldSize];
#pragma unroll
  for (uint32_t s = 0; s < kWorldSize; ++s)
    poll_buf[s] = params.push_workspaces[r] + static_cast<int64_t>(s) * stride +
                  phase_stride;
  const auto not_ready = [](const vec_t &v) {
    bool has_zero = false;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      has_zero |= is_pos_zero(v[j].x);
      has_zero |= is_pos_zero(v[j].y);
    }
    return has_zero;
  };
  const uint32_t poll_start = ((tx / 32) * gridDim.x + bx) * 32 + (tx & 31);
  for (uint32_t vid = poll_start; vid < num_vecs; vid += num_threads) {
    vec_t vec[kWorldSize];
#pragma unroll
    for (uint32_t s = 0; s < kWorldSize; ++s)
      ld_relaxed_16B(vec[s], poll_buf[s], vid);
    uint32_t pending = 0;
#pragma unroll
    for (uint32_t s = 0; s < kWorldSize; ++s)
      pending |= static_cast<uint32_t>(not_ready(vec[s])) << s;
    while (pending != 0) {
#pragma unroll
      for (uint32_t s = 0; s < kWorldSize; ++s) {
        if (pending & (1u << s)) {
          ld_relaxed_16B(vec[s], poll_buf[s], vid);
          if (!not_ready(vec[s]))
            pending &= ~(1u << s);
        }
      }
    }
    const uint32_t token = vid / slot_row_vecs;
    const uint32_t within = vid - token * slot_row_vecs;
    const uint32_t h = within / hd_vecs;
    const uint32_t dvec = within - h * hd_vecs;
    const int64_t out_vid0 =
        (static_cast<int64_t>(token) * num_heads + h) * hd_vecs + dvec;
#pragma unroll
    for (uint32_t s = 0; s < kWorldSize; ++s) {
      st_global_16B(vec[s], params.out,
                    out_vid0 + static_cast<int64_t>(s) * n_local * hd_vecs);
      st_global_16B(zero_vec, poll_buf[s], vid);
    }
  }
  __syncthreads();
  if (tx == 0) {
    params.push_counter[bx].inc(1);
    if (sync != nullptr) {
      const uint64_t add = bx == 0 ? (1ull << 20) - (gridDim.x - 1) : 1ull;
      asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" ::"l"(sync + 2),
                   "l"(add)
                   : "memory");
    }
  }
}
inline std::pair<uint32_t, uint32_t>
combine_push_launch_shape(uint32_t num_vecs, uint32_t nt, uint32_t num_blocks) {
  constexpr uint32_t kMaxBlock = 768;
  uint32_t split = 1;
  while (split < 16 && (nt + split - 1) / split > 8)
    split *= 2;
  while (split > 1 && static_cast<uint64_t>(num_vecs) * 2 * split >
                          static_cast<uint64_t>(num_blocks) * kMaxBlock)
    split /= 2;
  const uint64_t lanes = static_cast<uint64_t>(num_vecs) * 2 * split;
  uint32_t block_size =
      static_cast<uint32_t>((lanes + num_blocks - 1) / num_blocks);
  block_size =
      std::clamp<uint32_t>((block_size + 31u) / 32u * 32u, 32u, kMaxBlock);
  return {split, block_size};
}

template <uint32_t kWorldSize, bool kUsePDL> struct Eb5CombinePushGlobalKernel {
  using CommunicatorRef = host::distributed::CommunicatorRef;

  static void run(CommunicatorRef ref, const tvm::ffi::TensorView part,
                  const tvm::ffi::TensorView maxv,
                  const tvm::ffi::TensorView sumv,
                  const tvm::ffi::TensorView sink,
                  const tvm::ffi::TensorView out, int64_t num_tokens,
                  int64_t n_local, int64_t head_dim, int64_t num_heads,
                  int64_t nt, const tvm::ffi::TensorView sync,
                  const tvm::ffi::TensorView el, const tvm::ffi::TensorView pos,
                  int64_t tokens_per_req, int64_t c_ratio) {
    using namespace host;
    const auto &comm = *ref.get();
    const auto &push = comm.get_push_obj();
    RuntimeCheck(comm.get_world_size() == kWorldSize, "world size mismatch");
    RuntimeCheck(is_type<float>(part.dtype()), "part must be fp32");
    RuntimeCheck(is_type<float>(maxv.dtype()) && is_type<float>(sumv.dtype()),
                 "max/sum must be fp32");
    RuntimeCheck(is_type<bf16_t>(out.dtype()), "out must be bf16");
    RuntimeCheck(out.IsContiguous(), "out must be contiguous");
    RuntimeCheck(part.IsContiguous(), "part must be contiguous");
    RuntimeCheck(head_dim % 8 == 0, "head_dim must be a multiple of 8");
    RuntimeCheck(n_local * kWorldSize == num_heads,
                 "n_local * world_size must equal num_heads");
    RuntimeCheck(out.numel() == num_tokens * num_heads * head_dim,
                 "out.numel() must be num_tokens * num_heads * head_dim");
    RuntimeCheck(std::bit_cast<intptr_t>(out.data_ptr()) % 16 == 0,
                 "out must be 16B aligned");

    const int64_t shard_numel = num_tokens * n_local * head_dim;
    if (shard_numel == 0)
      return;
    const int64_t nbytes = shard_numel * static_cast<int64_t>(sizeof(bf16_t));
    RuntimeCheck(nbytes % 16 == 0, "shard bytes must be a multiple of 16");
    RuntimeCheck(nbytes <= push.slot_bytes,
                 "shard exceeds push workspace slot size");

    Eb5CombinePushGlobalParams<kWorldSize> params{};
    params.part = static_cast<const float *>(part.data_ptr());
    params.maxv = static_cast<const float *>(maxv.data_ptr());
    params.sumv = static_cast<const float *>(sumv.data_ptr());
    const bool has_sink = sink.numel() > 0;
    params.sink =
        has_sink ? static_cast<const float *>(sink.data_ptr()) : nullptr;
    params.out = static_cast<bf16_t *>(out.data_ptr());
    params.push_counter = push.counter;
    params.push_buffer_stride = push.slot_bytes;
    params.rank = comm.get_rank();
    const int64_t hd_vecs = head_dim / 8;
    params.num_vecs = static_cast<uint32_t>(num_tokens * n_local * hd_vecs);
    params.slot_row_vecs = static_cast<uint32_t>(n_local * hd_vecs);
    params.hd_vecs = static_cast<uint32_t>(hd_vecs);
    params.nt = static_cast<uint32_t>(nt);
    params.n_local = static_cast<uint32_t>(n_local);
    params.num_heads = static_cast<uint32_t>(num_heads);
    params.head_dim = static_cast<uint32_t>(head_dim);
    if (sync.numel() > 0) {
      RuntimeCheck(is_type<int32_t>(sync.dtype()) &&
                       sync.numel() >= 8 + num_tokens &&
                       std::bit_cast<intptr_t>(sync.data_ptr()) % 8 == 0,
                   "sync must be 8-byte aligned int32 [8 + num_tokens]");
      RuntimeCheck(is_type<int32_t>(el.dtype()) && el.numel() >= num_tokens,
                   "el must be int32 [num_tokens]");
      RuntimeCheck(is_type<int64_t>(pos.dtype()) && pos.numel() >= num_tokens,
                   "pos must be int64 [num_tokens]");
      RuntimeCheck(tokens_per_req > 0 && c_ratio > 0,
                   "bad tokens_per_req / c_ratio");
      params.sync = static_cast<int32_t *>(sync.data_ptr());
      params.el = static_cast<const int32_t *>(el.data_ptr());
      params.pos = static_cast<const int64_t *>(pos.data_ptr());
      params.tokens_per_req = static_cast<uint32_t>(tokens_per_req);
      params.c_ratio = static_cast<uint32_t>(c_ratio);
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i)
      params.push_workspaces[i] = push.workspaces[i];

    const auto stream = LaunchKernel::resolve_device(out.device());
    const uint32_t num_blocks = push.num_blocks;
    const auto [split, block_size] =
        combine_push_launch_shape(params.num_vecs, params.nt, num_blocks);
    params.split = split;
    if (has_sink)
      LaunchKernel(num_blocks, block_size, stream)
          .enable_pdl(kUsePDL)(
              eb5_combine_push_global_kernel<kWorldSize, kUsePDL, true>,
              params);
    else
      LaunchKernel(num_blocks, block_size, stream)
          .enable_pdl(kUsePDL)(
              eb5_combine_push_global_kernel<kWorldSize, kUsePDL, false>,
              params);
  }
};
} // namespace sglang

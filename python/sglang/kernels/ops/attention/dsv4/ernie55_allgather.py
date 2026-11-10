"""Open-tree port of the internal EB5 push all-gather kernels
(sglang/srt/internal/kernels/eb5/jit_kernel.py), the low-latency TP all-gather /
fused swapab-combine entries used by the ERNIE5.5 HCA decode.

Each entry pushes over the CustomAllReduceV2 push workspace (``comm_obj`` is the
``comm.obj`` Communicator held by ``get_tp_group().ca_comm``), replacing the NCCL
ring all-gather on the decode critical path. The CUDA source is the verbatim
internal ``eb5_allgather.cuh`` (copied to ``deepseek_v4/ernie55_allgather.cuh``);
only the JIT source path and module names are open-tree local. It compiles
against open's identical ``sgl_kernel/distributed/communicator.cuh`` ABI and the
ported ``csrc/distributed/ll_all_reduce_push.cuh`` device helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CUDA_FILE = "deepseek_v4/ernie55_allgather.cuh"


@cache_once
def _jit_eb5_allgather_push_module(world_size: int, use_pdl: bool) -> "Module":
    """Build (once per world_size/PDL) the fused NVLink push all-gather."""
    args = make_cpp_args(world_size, use_pdl)
    return load_jit(
        "ernie55_allgather_push",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[("allgather_push", f"Eb5AllGatherPushKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


def eb5_allgather_push(comm_obj, local: torch.Tensor, out: torch.Tensor) -> None:
    """Low-latency TP all-gather over the push workspace, replacing the NCCL
    ring all-gather. Every rank pushes its shard into slot ``rank`` of all peers,
    then polls all slots into ``out``. Caller guarantees eligibility (bf16,
    contiguous, 16B aligned, shard nbytes <= push slot size,
    out.numel() == world_size * local.numel())."""
    module = _jit_eb5_allgather_push_module(
        int(comm_obj.world_size), is_arch_support_pdl()
    )
    module.allgather_push(comm_obj, local, out)


@cache_once
def _jit_eb5_allgather_push_global_module(world_size: int, use_pdl: bool) -> "Module":
    """Build (once per world_size/PDL) the fused NVLink push all-gather that
    reads a token-strided source and writes global-head-order out."""
    args = make_cpp_args(world_size, use_pdl)
    return load_jit(
        "ernie55_allgather_push_global",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("allgather_push_global", f"Eb5AllGatherPushGlobalKernel<{args}>::run")
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def eb5_allgather_push_global(
    comm_obj,
    local: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
    n_local: int,
    head_dim: int,
    num_heads: int,
    src_row_stride_elems: int,
) -> None:
    """Fused low-latency TP all-gather: read this rank's head-sliced shard
    straight from the (token-strided) attention output and scatter every rank's
    heads into global-head-order ``out``, folding the pre-``contiguous()`` and
    post-``permute/reshape`` copies into the push/poll phases."""
    module = _jit_eb5_allgather_push_global_module(
        int(comm_obj.world_size), is_arch_support_pdl()
    )
    module.allgather_push_global(
        comm_obj,
        local,
        out,
        int(num_tokens),
        int(n_local),
        int(head_dim),
        int(num_heads),
        int(src_row_stride_elems),
    )


@cache_once
def _jit_eb5_combine_push_global_module(world_size: int, use_pdl: bool) -> "Module":
    """Build (once per world_size/PDL) the fused swapab combine + push all-gather
    (online-softmax merge from partials, then push/poll into global-head order)."""
    args = make_cpp_args(world_size, use_pdl)
    return load_jit(
        "ernie55_combine_push_global",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("combine_push_global", f"Eb5CombinePushGlobalKernel<{args}>::run")
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def eb5_combine_push_global(
    comm_obj,
    part: torch.Tensor,
    maxv: torch.Tensor,
    sumv: torch.Tensor,
    sink: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
    n_local: int,
    head_dim: int,
    num_heads: int,
    nt: int,
    late_sync=None,
) -> None:
    """Fused swapab combine + TP all-gather: merge this rank's per-tile attention
    partials (online softmax) into its n_local-head shard, then push/poll into
    global-head-order ``out``. ``late_sync``:
    ``(sync, el, pos, tokens_per_req, c_ratio)`` makes late tokens wait on the
    side-stream mode-2 partial's flag instead of a graph edge."""
    module = _jit_eb5_combine_push_global_module(
        int(comm_obj.world_size), is_arch_support_pdl()
    )
    if late_sync is None:
        empty = part.new_empty(0, dtype=torch.int32)
        late_sync = (empty, empty, empty.long(), 1, 1)
    sync, el, pos, tpr, c_ratio = late_sync
    module.combine_push_global(
        comm_obj,
        part,
        maxv,
        sumv,
        sink,
        out,
        int(num_tokens),
        int(n_local),
        int(head_dim),
        int(num_heads),
        int(nt),
        sync,
        el,
        pos,
        int(tpr),
        int(c_ratio),
    )

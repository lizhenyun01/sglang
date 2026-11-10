"""Open-tree port of the internal EB5 low-latency TP all-gather for the
attention-TP VHA postmix (sglang/srt/internal/layers/moe/eb5_fast_allgather.py).

``try_fast_combine_push_global`` fuses the swapab online-softmax combine with the
attn-TP head all-gather into a single 1-shot NVLink push over the
CustomAllReduceV2 push workspace (``comm = get_tp_group().ca_comm``), replacing the
4-rank NCCL RING_LL all-gather (sequential NVLink hops + ring rendezvous, a decode
critical-path bubble). The output is global-head-order ``[num_tokens, num_heads,
head_dim]`` so the caller consumes 64 heads directly, skipping the per-rank slice
and the postmix's own gather.

Returns False to fall back to the stock triton combine + NCCL all-gather (env off,
no custom-AR comm on the attn-TP group, or the shard is not push eligible), so the
caller always has a correct non-push path.

Internal-to-open mapping: SGLANG_INTERNAL_EB5_SINGLE_BATCH_LL ->
envs.SGLANG_ERNIE55_SINGLE_BATCH_LL; the eb5 push kernels come from the ported
ernie55_allgather module; ``wants_nccl`` is the inlined over-threshold check (the
full all-reduce NCCL-fallback hook subsystem is not needed for the decode path,
where the batch is always under the LL token ceiling).
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs


def wants_nccl(
    x: torch.Tensor, num_tokens: Optional[int] = None, op: Optional[str] = None
) -> bool:
    """True when this collective belongs to an over-threshold forward (the push
    kernels decline and the caller takes the stock NCCL all-gather). Mirrors
    internal ll_nccl_fallback.wants_nccl, inlined against the LL token ceiling."""
    if not envs.SGLANG_ERNIE55_SINGLE_BATCH_LL.get() or x.is_cpu or x.dim() == 0:
        return False
    from sglang.srt.layers.quantization.ernie55_halops_ll_gemm import LL_MAX_TOKENS

    tokens = x.shape[0] if num_tokens is None else num_tokens
    return tokens > LL_MAX_TOKENS


def try_fast_combine_push_global(
    part: torch.Tensor,
    maximum: torch.Tensor,
    sums: torch.Tensor,
    sink,
    out_global: torch.Tensor,
    n_local: int,
    head_dim: int,
    num_heads: int,
    nt: int,
    late_sync=None,
) -> bool:
    """Fused swapab combine + attn-TP head gather in one kernel: online-softmax
    merge of ``part/maximum/sums`` into this rank's shard, then push/poll into
    global-head-order ``out_global``. Returns True on success, False to fall back
    to triton combine + separate gather."""
    if not envs.SGLANG_ERNIE55_SINGLE_BATCH_LL.get():
        return False
    if part.dim() != 4 or out_global.dim() != 3:
        return False

    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().attn_tp_size != get_tp_group().world_size:
        return False
    comm = get_tp_group().ca_comm
    if comm is None or getattr(comm, "disabled", True):
        return False
    world_size = getattr(comm, "world_size", 1)
    if world_size <= 1 or n_local * world_size != num_heads:
        return False

    if part.dtype != torch.float32 or not part.is_contiguous():
        return False
    if maximum.dtype != torch.float32 or sums.dtype != torch.float32:
        return False
    if out_global.dtype != torch.bfloat16 or not out_global.is_contiguous():
        return False
    if head_dim % 8 != 0:
        return False
    if part.data_ptr() % 16 != 0 or out_global.data_ptr() % 16 != 0:
        return False
    num_tokens = part.shape[0]
    if out_global.shape[1] != num_heads or out_global.shape[2] != head_dim:
        return False
    nbytes = num_tokens * n_local * head_dim * 2
    if nbytes == 0 or nbytes % 16 != 0:
        return False
    max_push = getattr(comm, "max_push_size", None)
    if max_push is None or nbytes > int(max_push):
        return False

    from sglang.kernels.ops.attention.dsv4.ernie55_allgather import (
        eb5_combine_push_global,
    )

    sink_t = sink if sink is not None else part.new_empty(0, dtype=torch.float32)
    eb5_combine_push_global(
        comm.obj,
        part,
        maximum,
        sums,
        sink_t,
        out_global,
        num_tokens,
        n_local,
        head_dim,
        num_heads,
        nt,
        late_sync=late_sync,
    )
    return True


def try_fast_allgather_push(
    local: torch.Tensor,
    out: torch.Tensor,
    attn_tp: int,
) -> bool:
    """Low-latency attn-TP head all-gather over the CustomAllReduceV2 push
    workspace: a drop-in for ``attn_tp_all_gather_into_tensor(out, local)`` that
    replaces the NCCL RING_LL ring all-gather on the single-token decode / DSPARK
    draft path (where ``try_fast_combine_push_global`` does not apply because the
    swapab combine already ran, or FlashMLA produced the heads directly). ``out``
    is ``[world, num_tokens, n_local, head_dim]`` -- slot ``rank`` holds rank's
    shard, byte-identical to the NCCL all-gather layout the caller then permutes to
    global head order. Returns False to fall back to NCCL (env off, no custom-AR
    comm on the attn-TP group, or the shard is not push eligible), so the caller
    always keeps a correct non-push path."""
    if not envs.SGLANG_ERNIE55_SINGLE_BATCH_LL.get():
        return False
    if not envs.SGLANG_ERNIE55_SWAPAB_PUSH.get():
        return False
    if local.dtype != torch.bfloat16 or not local.is_contiguous():
        return False
    if out.dtype != torch.bfloat16 or not out.is_contiguous():
        return False

    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().attn_tp_size != get_tp_group().world_size:
        return False
    comm = get_tp_group().ca_comm
    if comm is None or getattr(comm, "disabled", True):
        return False
    world_size = getattr(comm, "world_size", 1)
    if world_size <= 1 or world_size != attn_tp:
        return False
    if out.numel() != world_size * local.numel():
        return False
    if local.data_ptr() % 16 != 0 or out.data_ptr() % 16 != 0:
        return False
    nbytes = local.numel() * local.element_size()
    if nbytes == 0 or nbytes % 16 != 0:
        return False
    max_push = getattr(comm, "max_push_size", None)
    if max_push is None or nbytes > int(max_push):
        return False

    from sglang.kernels.ops.attention.dsv4.ernie55_allgather import (
        eb5_allgather_push,
    )

    eb5_allgather_push(comm.obj, local, out)
    return True


def try_fast_allgather_global(
    local: torch.Tensor,
    out_global: torch.Tensor,
    n_local: int,
    head_dim: int,
    num_heads: int,
) -> bool:
    """Fused attn-TP head gather (port of internal ``try_fast_allgather_global``):
    read this rank's (token-strided) shard ``local`` [num_tokens, n_local, head_dim]
    and scatter every rank's heads straight into global-head-order ``out_global``
    [num_tokens, num_heads, head_dim] -- no pre-``contiguous()`` and no
    post-``permute/reshape``. Returns True on success, False to fall back to the
    rank-major gather + permute path."""
    if not envs.SGLANG_ERNIE55_SINGLE_BATCH_LL.get():
        return False
    if local.dim() != 3 or out_global.dim() != 3:
        return False
    if wants_nccl(local, op="fast_allgather_global"):
        return False

    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().attn_tp_size != get_tp_group().world_size:
        return False
    comm = get_tp_group().ca_comm
    if comm is None or getattr(comm, "disabled", True):
        return False
    world_size = getattr(comm, "world_size", 1)
    if world_size <= 1 or n_local * world_size != num_heads:
        return False

    if local.dtype != torch.bfloat16 or out_global.dtype != torch.bfloat16:
        return False
    if not out_global.is_contiguous():
        return False
    if local.shape[1] != n_local or local.shape[2] != head_dim:
        return False
    if out_global.shape[1] != num_heads or out_global.shape[2] != head_dim:
        return False
    if local.shape[0] != out_global.shape[0]:
        return False
    if local.stride(2) != 1 or local.stride(1) != head_dim:
        return False
    src_row_stride = local.stride(0)
    if head_dim % 8 != 0 or src_row_stride % 8 != 0:
        return False
    if local.data_ptr() % 16 != 0 or out_global.data_ptr() % 16 != 0:
        return False
    num_tokens = local.shape[0]
    nbytes = num_tokens * n_local * head_dim * local.element_size()
    if nbytes == 0 or nbytes % 16 != 0:
        return False
    max_push = getattr(comm, "max_push_size", None)
    if max_push is None or nbytes > int(max_push):
        return False

    from sglang.kernels.ops.attention.dsv4.ernie55_allgather import (
        eb5_allgather_push_global,
    )

    eb5_allgather_push_global(
        comm.obj,
        local,
        out_global,
        num_tokens,
        n_local,
        head_dim,
        num_heads,
        src_row_stride,
    )
    return True


def try_fast_allgather_rank_major(local: torch.Tensor, out: torch.Tensor) -> bool:
    """Rank-major push all-gather (port of internal ``try_fast_allgather``): like
    ``try_fast_allgather_push`` but without the SWAPAB_PUSH gate, as the internal
    VHA-postmix fallback gather calls it. ``out`` is ``[world, *local.shape]``."""
    if not envs.SGLANG_ERNIE55_SINGLE_BATCH_LL.get():
        return False
    if wants_nccl(local, op="fast_allgather"):
        return False

    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().attn_tp_size != get_tp_group().world_size:
        return False
    comm = get_tp_group().ca_comm
    if comm is None or getattr(comm, "disabled", True):
        return False
    world_size = getattr(comm, "world_size", 1)
    if world_size <= 1:
        return False
    if local.dtype != torch.bfloat16 or not local.is_contiguous():
        return False
    if out.dtype != torch.bfloat16 or not out.is_contiguous():
        return False
    if out.numel() != local.numel() * world_size:
        return False
    nbytes = local.numel() * local.element_size()
    if nbytes == 0 or nbytes % 16 != 0:
        return False
    if local.data_ptr() % 16 != 0 or out.data_ptr() % 16 != 0:
        return False
    max_push = getattr(comm, "max_push_size", None)
    if max_push is None or nbytes > int(max_push):
        return False

    from sglang.kernels.ops.attention.dsv4.ernie55_allgather import (
        eb5_allgather_push,
    )

    eb5_allgather_push(comm.obj, local, out)
    return True

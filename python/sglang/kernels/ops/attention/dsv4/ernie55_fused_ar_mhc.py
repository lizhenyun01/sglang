"""Open-tree port of the internal EB5 fused all-reduce + mHC post/pre
(sglang/srt/internal/layers/attention/fused_all_reduce_mhc.py), one 16-CTA
thread-block-cluster kernel.

Post residual, one fused ``W r'`` + Gram reduction, split mix, RMSNorm and the
group-32 fp8 quant of ``layer_out`` (``ernie55_fused_all_reduce_mhc.cuh``), on
an already reduced ``x`` (``fused_ar_mhc``, optionally folding a shared-expert
output) or on this rank's attention partial, all-reduced in-kernel over the LL
push plane (``fused_ar_mhc_push``).

The CUDA source is the verbatim internal ``eb5/fused_all_reduce_mhc.cuh``
(copied to ``deepseek_v4/ernie55_fused_all_reduce_mhc.cuh``; only its
``g32_fp8_quant.cuh`` include is renamed to the open copy). It compiles against
open's identical ``sgl_kernel/distributed/communicator.cuh`` and
``csrc/distributed/ll_all_reduce_push.cuh``. ``mhc_post_hc_mean`` is the port of
the internal ``layers/attention/mhc_post_hc_mean.py`` (DSpark capture of
``mhc_post(...).mean(1)`` without the [T, 4, H] intermediate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import msgspec
import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import (
    get_tp_group,
    model_parallel_is_initialized,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CUDA_FILE = "deepseek_v4/ernie55_fused_all_reduce_mhc.cuh"
_HC_MEAN_CUDA_FILE = "deepseek_v4/ernie55_mhc_post_hc_mean.cuh"

HC = 4
HIDDEN = 4096
GROUP = 32
CLUSTER = 16
MAX_TOKENS = 16
# cudaOccupancyMaxActiveClusters for 16-CTA clusters on GB200 (152 SMs); the
# grid must stay co-resident for the in-kernel cross-rank all-reduce.
MAX_CLUSTERS = 7

FusedArMhcOut = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def default_tokens_per_cluster(num_tokens: int) -> int:
    """Fewest tokens per cluster that keep the grid within ``MAX_CLUSTERS``."""
    return -(-num_tokens // MAX_CLUSTERS)


# kProf ring entry of one CTA: kProfWords u64 (fused_all_reduce_mhc.cuh, ProfArgs).
PROF_WORDS = 48
PROF_LAYOUT = "fused_ar_mhc.kprof.v1"


class FusedArMhcProf(msgspec.Struct, frozen=True):
    """Device ring of per-launch fused mHC stamps (profiling only).

    Launch n of any kernel writing it (all sharing one counter) lands in entry
    ``n % num_slots``; ``seq`` holds n in device memory, so a captured graph
    keeps appending on every replay."""

    ring: torch.Tensor  # [num_slots, MAX_CLUSTERS, CLUSTER, PROF_WORDS] int64
    seq: torch.Tensor  # [MAX_CLUSTERS * CLUSTER] int32
    num_slots: int


def make_fused_ar_mhc_prof(device, *, num_slots: int = 2048) -> FusedArMhcProf:
    """A zeroed ring for ``num_slots`` launches (43 KB each). Allocate it outside
    graph capture and pass it as ``prof=`` to the fused mHC entry points."""
    assert num_slots > 0, num_slots
    dev = torch.device(device)
    shape = (num_slots, MAX_CLUSTERS, CLUSTER, PROF_WORDS)
    return FusedArMhcProf(
        ring=torch.zeros(shape, dtype=torch.int64, device=dev),
        seq=torch.zeros(MAX_CLUSTERS * CLUSTER, dtype=torch.int32, device=dev),
        num_slots=num_slots,
    )


def dump_fused_ar_mhc_prof(prof: FusedArMhcProf) -> dict:
    """CPU copy of ``prof``, ordered after the caller's current stream."""
    return dict(
        layout=PROF_LAYOUT,
        num_slots=prof.num_slots,
        ring=prof.ring.cpu(),
        seq=prof.seq.cpu(),
    )


class _Variant(msgspec.Struct, frozen=True):
    """Compile-time switches of one FusedArMhc instance, in template order."""

    tpc: int
    align_fleet: bool
    fuse_ar: bool = False
    has_shared: bool = False
    prof: bool = False
    early_trigger: bool = False
    comb_logits: bool = False


@cache_once
def _jit_module(v: _Variant, use_pdl: bool) -> Module:
    b = lambda f: "true" if f else "false"  # noqa: E731
    flags = (v.align_fleet, use_pdl, v.fuse_ar, v.has_shared, v.prof,
             v.early_trigger, v.comb_logits)  # fmt: skip
    cls = (
        "eb5::fused_ar_mhc::FusedArMhc<"
        + ", ".join([str(v.tpc)] + [b(f) for f in flags])
        + ">"
    )
    run = ("run_push" if v.fuse_ar else "run") + ("_prof" if v.prof else "")
    # No --use_fast_math: the push all-reduce sum must not flush denormals.
    return load_jit(
        "ernie55_fused_ar_mhc",
        str(v.tpc),
        *(str(int(f)) for f in flags),
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            (run, f"{cls}::{run}"),
            ("max_active_clusters", f"{cls}::max_active_clusters"),
        ],
    )


def _module(v: _Variant) -> Module:
    return _jit_module(v, is_arch_support_pdl())


@cache_once
def _max_clusters(v: _Variant) -> int:
    return int(_module(v).max_active_clusters())


@cache_once
def _sinkhorn_module(align_fleet: bool) -> Module:
    cls = "eb5::fused_ar_mhc::FusedArMhcSinkhorn<{}>".format(
        "true" if align_fleet else "false"
    )
    return load_jit(
        "ernie55_fused_ar_mhc_sinkhorn",
        str(int(align_fleet)),
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("run", f"{cls}::run"),
            ("run_prefetch", f"{cls}::run_prefetch"),
        ],
    )


@cache_once
def _prefetch_module() -> Module:
    return load_jit(
        "ernie55_mhc_l2_prefetch",
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[("run", "eb5::fused_ar_mhc::MhcL2Prefetch::run")],
    )


# fused_all_reduce_mhc.cuh kMaxPrefetchRanges.
MAX_PREFETCH_RANGES = 8
# Bytes per cp.async.bulk.prefetch.L2 instruction.
PREFETCH_CHUNK = 64 * 1024


class L2Prefetch(msgspec.Struct, frozen=True):
    """Device tensors a side kernel prefetches into L2 ahead of a later kernel; a
    hint only, no output depends on it."""

    tensors: Tuple[torch.Tensor, ...]
    # Bytes per prefetch instruction, a multiple of 16.
    chunk: int = PREFETCH_CHUNK
    # Minimum one-warp CTA count of the issuing kernel; all of them prefetch.
    ctas: int = 1


def l2_prefetch_plan(tensors) -> torch.Tensor:
    """[S, 2] int64 host (address, bytes) of the 16B granules covering each non-empty
    contiguous CUDA tensor. Allocations are 16B-granular, so the rounding stays
    inside each tensor's allocation."""
    rows = []
    for t in tensors:
        assert t.is_cuda and t.is_contiguous(), (t.device, tuple(t.shape))
        nbytes = t.numel() * t.element_size()
        if nbytes == 0:
            continue
        start = t.data_ptr() & ~15
        rows.append((start, ((t.data_ptr() + nbytes + 15) & ~15) - start))
    assert len(rows) <= MAX_PREFETCH_RANGES, f"{len(rows)} L2 prefetch ranges"
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 2)


@register_custom_op(
    op_name="ernie55_fused_ar_mhc",
    mutates_args=["residual_out", "layer_out", "post_out", "mix_out", "q", "sf"],
)
def _fused_ar_mhc_op(
    x: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    residual_out: torch.Tensor,
    layer_out: torch.Tensor,
    post_out: torch.Tensor,
    mix_out: torch.Tensor,
    q: torch.Tensor,
    sf: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    write_logits: bool,
    align_fleet: bool,
    has_shared: bool,
    comb_logits: bool,
    tpc: int,
) -> None:
    variant = _Variant(
        tpc=tpc,
        align_fleet=align_fleet,
        has_shared=has_shared,
        comb_logits=comb_logits,
    )
    _module(variant).run(
        x, shared, residual, post, comb, fn, hc_scale, hc_base, norm_weight,
        residual_out, layer_out, post_out, mix_out, q, sf,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, norm_eps, write_logits,
    )  # fmt: skip


@register_custom_op(op_name="ernie55_fused_ar_mhc_sinkhorn", mutates_args=["comb_out"])
def _fused_ar_mhc_sinkhorn_op(
    logits: torch.Tensor,
    comb_out: torch.Tensor,
    eps: float,
    repeat: int,
    align_fleet: bool,
) -> None:
    _sinkhorn_module(align_fleet).run(logits, comb_out, eps, repeat)


@register_custom_op(
    op_name="ernie55_fused_ar_mhc_sinkhorn_prefetch", mutates_args=["comb_out"]
)
def _fused_ar_mhc_sinkhorn_prefetch_op(
    logits: torch.Tensor,
    comb_out: torch.Tensor,
    plan: torch.Tensor,
    eps: float,
    repeat: int,
    align_fleet: bool,
    chunk: int,
    ctas: int,
) -> None:
    _sinkhorn_module(align_fleet).run_prefetch(
        logits, comb_out, plan, eps, repeat, chunk, ctas
    )


@register_custom_op(op_name="ernie55_mhc_l2_prefetch", mutates_args=[])
def _mhc_l2_prefetch_op(
    anchor: torch.Tensor, plan: torch.Tensor, ctas: int, chunk: int
) -> None:
    _prefetch_module().run(anchor, plan, ctas, chunk)


def _alloc_layer_out(num_tokens: int, device: torch.device) -> torch.Tensor:
    if model_parallel_is_initialized():
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            return torch.empty(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    return torch.empty(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)


def _prepare(x, residual, post, comb, fn, hc_scale, hc_base, norm_weight, logits_out):
    """Contiguous kernel inputs plus fresh outputs: ``(kernel tensors, result)``."""
    num_tokens, hidden = x.shape
    assert 0 < num_tokens <= MAX_TOKENS, f"num_tokens={num_tokens} not in [1, 16]"
    assert hidden == HIDDEN and residual.shape[-2] == HC
    dev = x.device
    residual_out = torch.empty(num_tokens, HC, HIDDEN, dtype=torch.bfloat16, device=dev)
    layer_out = _alloc_layer_out(num_tokens, dev)
    post_out = torch.empty(num_tokens, HC, dtype=torch.float32, device=dev)
    comb_out = torch.empty(num_tokens, HC * HC, dtype=torch.float32, device=dev)
    q = torch.empty(num_tokens, HIDDEN, dtype=torch.float8_e4m3fn, device=dev)
    sf = torch.empty(num_tokens, HIDDEN // GROUP, dtype=torch.uint8, device=dev)
    if logits_out is not None:
        assert logits_out.shape == (num_tokens, HC * HC) and logits_out.is_contiguous()
    args = (
        x.contiguous(),
        residual.view(num_tokens, HC, HIDDEN).contiguous(),
        post.reshape(num_tokens, HC).contiguous(),
        comb.reshape(num_tokens, HC * HC).contiguous(),
        fn.view(-1, HC, HIDDEN).contiguous(),
        hc_scale,
        hc_base,
        norm_weight.bfloat16().contiguous(),
        residual_out,
        layer_out,
        post_out,
        comb_out if logits_out is None else logits_out,
        q,
        sf,
    )
    return args, (
        residual_out,
        layer_out,
        post_out,
        comb_out.view(num_tokens, HC, HC),
        q,
        sf,
    )


def _scalars(
    rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
    norm_eps, logits_out,
) -> tuple:  # fmt: skip
    return (
        float(rms_eps),
        float(hc_pre_eps),
        float(hc_sinkhorn_eps),
        float(hc_post_mult_value),
        int(sinkhorn_repeat),
        float(norm_eps),
        logits_out is not None,
    )


def _comb_input(comb, comb_logits, num_tokens: int):
    """The kernel's comb operand and whether it holds logits to sinkhorn."""
    assert (comb is None) != (comb_logits is None), (
        "pass exactly one of comb, comb_logits"
    )
    if comb is not None:
        return comb, False
    assert comb_logits.shape == (num_tokens, HC * HC), comb_logits.shape
    assert comb_logits.dtype == torch.float32
    return comb_logits, True


def fused_ar_mhc(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: Optional[torch.Tensor],
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    mhc_align_fleet: bool,
    logits_out: Optional[torch.Tensor] = None,
    shared: Optional[torch.Tensor] = None,
    comb_logits: Optional[torch.Tensor] = None,
    prof: Optional[FusedArMhcProf] = None,
    prof_tag: int = 0,
) -> FusedArMhcOut:
    """Fused mHC post/pre: ``(residual_out [T, 4, H] bf16, layer_out [T, H] bf16,
    post_out [T, 4] f32, comb_out [T, 4, 4] f32, q [T, H] fp8 e4m3,
    sf [T, H / 32] uint8 UE8M0)``.

    With ``logits_out`` ([T, 16] f32) the kernel writes the comb logits there
    and leaves ``comb_out`` unfilled; the caller runs ``fused_ar_mhc_sinkhorn``
    on the stream it chooses. ``shared`` ([T, H] bf16, a folded shared-expert
    output) is added to ``x`` in fp32 first.

    ``comb_logits`` ([T, 16] f32, a producer kernel's ``logits_out``) replaces
    ``comb`` (pass ``comb=None``): the kernel runs that sinkhorn itself, with
    the same result as ``comb = fused_ar_mhc_sinkhorn(comb_logits)``.
    """
    num_tokens = x.shape[0]
    comb_in, from_logits = _comb_input(comb, comb_logits, num_tokens)
    args, out = _prepare(
        x, residual, post, comb_in, fn, hc_scale, hc_base, norm_weight, logits_out
    )
    if shared is not None:
        assert shared.shape == x.shape and shared.dtype == torch.bfloat16
        shared = shared.contiguous()
    shared_in = args[0] if shared is None else shared
    scalars = _scalars(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, norm_eps, logits_out,
    )  # fmt: skip
    tpc = default_tokens_per_cluster(num_tokens)
    if prof is None:
        _fused_ar_mhc_op(
            args[0], shared_in, *args[1:], *scalars, bool(mhc_align_fleet),
            shared is not None, from_logits, tpc,
        )  # fmt: skip
        return out
    variant = _Variant(
        tpc=tpc,
        align_fleet=bool(mhc_align_fleet),
        has_shared=shared is not None,
        prof=True,
        comb_logits=from_logits,
    )
    _module(variant).run_prof(
        prof.ring, prof.seq, int(prof_tag), args[0], shared_in, *args[1:], *scalars
    )
    return out


def fused_ar_mhc_sinkhorn(
    logits_out: torch.Tensor,
    comb_out: torch.Tensor,
    *,
    eps: float,
    repeat: int,
    align_fleet: bool,
    prefetch: Optional[L2Prefetch] = None,
) -> None:
    """``comb_out`` ([T, 4, 4] or [T, 16] f32) = sinkhorn of ``logits_out``,
    bit-identical to the comb the fused mHC kernel computes in-kernel. With
    ``prefetch`` the kernel's lanes first issue that L2 prefetch.
    """
    num_tokens = logits_out.shape[0]
    logits = logits_out.view(num_tokens, HC * HC)
    comb = comb_out.view(num_tokens, HC * HC)
    if prefetch is None:
        _fused_ar_mhc_sinkhorn_op(
            logits, comb, float(eps), int(repeat), bool(align_fleet)
        )
        return
    _fused_ar_mhc_sinkhorn_prefetch_op(
        logits,
        comb,
        l2_prefetch_plan(prefetch.tensors),
        float(eps),
        int(repeat),
        bool(align_fleet),
        int(prefetch.chunk),
        int(prefetch.ctas),
    )


def mhc_l2_prefetch(prefetch: L2Prefetch) -> None:
    """The L2 prefetch of ``prefetch`` alone: ``prefetch.ctas`` one-warp CTAs on
    the current stream, no PDL; a no-op without tensors."""
    if not prefetch.tensors:
        return
    _mhc_l2_prefetch_op(
        prefetch.tensors[0],
        l2_prefetch_plan(prefetch.tensors),
        int(prefetch.ctas),
        int(prefetch.chunk),
    )


def max_active_clusters(num_tokens: int, mhc_align_fleet: bool = True) -> int:
    """Co-resident 16-CTA clusters; the grid needs ``ceil(T / tpc)`` of them."""
    tpc = default_tokens_per_cluster(num_tokens)
    return _max_clusters(_Variant(tpc=tpc, align_fleet=bool(mhc_align_fleet)))


def fused_ar_mhc_push(
    x_partial: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: Optional[torch.Tensor],
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    push_comm,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    mhc_align_fleet: bool,
    logits_out: Optional[torch.Tensor] = None,
    early_trigger: bool = False,
    comb_logits: Optional[torch.Tensor] = None,
    prof: Optional[FusedArMhcProf] = None,
    prof_tag: int = 0,
) -> FusedArMhcOut:
    """``fused_ar_mhc`` of ``x = sum over TP ranks of x_partial``, reduced
    in-kernel over the LL push plane of ``push_comm`` (the TP
    ``CustomAllReduceV2``, world size 4); the sum is bitwise that of the LL push
    all-reduce. A collective: every rank must call it with the same ``T``.

    ``early_trigger`` lets the PDL dependents launch right after this rank's
    push instead of after the cluster reduction; outputs are unchanged.
    """
    num_tokens = x_partial.shape[0]
    tpc = default_tokens_per_cluster(num_tokens)
    clusters = -(-num_tokens // tpc)
    comb_in, from_logits = _comb_input(comb, comb_logits, num_tokens)
    variant = _Variant(
        tpc=tpc,
        align_fleet=bool(mhc_align_fleet),
        fuse_ar=True,
        prof=prof is not None,
        early_trigger=bool(early_trigger),
        comb_logits=from_logits,
    )
    num_blocks = int(push_comm.obj.push.num_blocks)
    assert CLUSTER * clusters <= num_blocks, (
        f"{clusters} clusters > {num_blocks} push blocks"
    )
    max_clusters = _max_clusters(variant)
    assert clusters <= max_clusters, f"{clusters} clusters > {max_clusters} co-resident"
    args, out = _prepare(
        x_partial, residual, post, comb_in, fn, hc_scale, hc_base, norm_weight,
        logits_out,
    )  # fmt: skip
    scalars = _scalars(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, norm_eps, logits_out,
    )  # fmt: skip
    if prof is None:
        _module(variant).run_push(push_comm.obj, *args, *scalars)
    else:
        _module(variant).run_push_prof(
            prof.ring, prof.seq, int(prof_tag), push_comm.obj, *args, *scalars
        )
    return out


def max_active_clusters_push(
    num_tokens: int, mhc_align_fleet: bool = True, early_trigger: bool = False
) -> int:
    """``max_active_clusters`` of the fused push all-reduce kernel."""
    return _max_clusters(
        _Variant(
            tpc=default_tokens_per_cluster(num_tokens),
            align_fleet=bool(mhc_align_fleet),
            fuse_ar=True,
            early_trigger=bool(early_trigger),
        )
    )


def precompile_push(
    *,
    world_size: int,
    mhc_align_fleet: bool = True,
    early_trigger: bool = False,
    comb_logits: bool = False,
    prof: bool = False,
) -> None:
    """Build the push modules ``T = 1..MAX_TOKENS`` uses with these switches,
    plus the unfused and sinkhorn modules of the reference path; launches
    nothing."""
    assert world_size == 4, "the fused push all-reduce is TP4 only"
    fleet = bool(mhc_align_fleet)
    for tpc in sorted(
        {default_tokens_per_cluster(t) for t in range(1, MAX_TOKENS + 1)}
    ):
        _module(
            _Variant(
                tpc=tpc,
                align_fleet=fleet,
                fuse_ar=True,
                prof=bool(prof),
                early_trigger=bool(early_trigger),
                comb_logits=bool(comb_logits),
            )
        )
        _module(_Variant(tpc=tpc, align_fleet=fleet))
    _sinkhorn_module(fleet)


# --- mhc_post(...).mean(1) in one kernel (internal mhc_post_hc_mean.py) -------


@cache_once
def _hc_mean_module(hc_mult: int, use_pdl: bool) -> Module:
    args = make_cpp_args(hc_mult, use_pdl)
    return load_jit(
        f"ernie55_mhc_post_hc_mean_{hc_mult}_{use_pdl}",
        *args,
        cuda_files=[_HC_MEAN_CUDA_FILE],
        cuda_wrappers=[("run", f"mhc_post_hc_mean<{args}>")],
        extra_cuda_cflags=[],
    )


def mhc_post_hc_mean(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """``mhc_post(x, residual, post, comb).mean(dim=1)`` [T, H] bf16, without
    materializing the [T, hc, H] residual."""
    num_tokens, hc, hidden = residual.shape
    x = x.reshape(num_tokens, hidden).contiguous()
    residual = residual.contiguous()
    post = post.reshape(num_tokens, hc).contiguous()
    comb = comb.reshape(num_tokens, hc, hc).contiguous()
    out = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=x.device)
    _hc_mean_module(hc, is_arch_support_pdl()).run(x, residual, post, comb, out)
    return out

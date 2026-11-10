"""Gate and dispatch for the ERNIE 5.5 fused all-reduce + mHC kernels.

Port of internal ``fused_all_reduce_mhc_dispatch.py``. Each decoder layer has
two sides, both on under ``SGLANG_ERNIE55_FUSED_AR_MHC``:

- attention side (attention -> FFN): the attention all-reduce, the FFN
  mhc_post + mhc_pre and the FFN-entry group-32 fp8 quant run as one
  ``fused_ar_mhc_push`` kernel, the all-reduce pushed in-kernel over the
  CustomAllReduceV2 push plane.
- MoE side (FFN -> next attention): the next layer's mhc_post + mhc_pre on the
  already reduced FFN output runs as one ``fused_ar_mhc`` kernel (no
  all-reduce); its group-32 pair goes to the attention's first GEMM when
  ``accepted_input_quant`` holds.

The FFN comb sinkhorn is deferred to a side stream (which also L2-prefetches
the next layer's MoE-side inputs) forked at the MoE launch, since the open mega
MoE has no post-gemm2 hook, and the attention comb is finished inside
the attention-side kernel. ``resolve_gates`` freezes the config-only half at
layer build and ``FusedArMhcGate.runs`` adds the token count; neither reads
rank-local state, so every attention-TP rank makes the same choice. That is
load-bearing: a rank that skipped the all-reduce while a peer ran it would hang
the push plane.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SWITCH = "SGLANG_ERNIE55_FUSED_AR_MHC"
# Kernel coverage: <= 7 co-resident clusters of 16 CTAs, and hc_mult / hidden
# are compile-time constants of the kernel.
MAX_TOKENS = 16
_HC_MULT = 4
_HIDDEN = 4096
# The fused push loop is written for exactly 4 peers.
_PUSH_WORLD = 4

_WARNED: set = set()
# has_shared values already reported by log_moe_variant in this process.
_LOGGED_MOE_VARIANTS: set = set()
# device index -> the deferred-sinkhorn side stream (_sinkhorn_stream).
_SINKHORN_STREAMS: dict = {}


def switch_on() -> bool:
    return bool(envs.SGLANG_ERNIE55_FUSED_AR_MHC.get())


class FusedArMhcGate(msgspec.Struct, frozen=True):
    """Config-only half of one side's gate, frozen per decoder layer."""

    # The fused kernel entry this side runs; None = off.
    kernel: object = None
    # Attention side only: the attention-TP GroupCoordinator whose push plane
    # the kernel rides.
    push_group: object = None
    # Attention side only: fused_ar_mhc_sinkhorn for the deferred FFN comb.
    sinkhorn: object = None

    @property
    def enabled(self) -> bool:
        return self.kernel is not None

    def runs(self, *, num_tokens: int, small_path: bool, prefill_cp: bool) -> bool:
        """Whether this forward runs the kernel, from the token count and frozen
        config only, so every rank gets the same answer at any point of it."""
        return (
            self.enabled
            and not prefill_cp
            and small_path
            and 1 <= num_tokens <= MAX_TOKENS
        )

    def check_ll_push(self, ar_input: torch.Tensor) -> None:
        """Attention side: raise unless the attention all-reduce of ``ar_input``
        would run LL push, the plane the fused kernel rides; never a silent
        fallback."""
        if not _runs_ll_push(self.push_group, ar_input):
            raise RuntimeError(
                f"{_SWITCH}=1: the attention all-reduce of "
                f"{tuple(ar_input.shape)} {ar_input.dtype} would not run LL push"
            )

    def run(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        *,
        comb_is_logits: bool = False,
        emit_logits: bool = False,
        **scalars,
    ) -> Tuple[torch.Tensor, ...]:
        """(residual_out [T,4,H], layer_out [T,H], post_out [T,4], comb_out [T,4,4],
        (q fp8 [T,H], sf uint8 [T,H/32])). On the attention side ``x`` is this
        rank's un-reduced attention output. ``comb_is_logits``: ``comb`` is the
        MoE-side kernel's [T,16] f32 logits, finished in-kernel. ``emit_logits``:
        the kernel skips its sinkhorn and comb_out is its [T,16] f32 comb logits."""
        if self.push_group is not None:
            scalars["push_comm"] = self.push_group.ca_comm
            # PDL dependents launch right after this rank's push.
            scalars["early_trigger"] = True
        if comb_is_logits:
            scalars["comb_logits"] = comb
            comb = None
        logits = None
        if emit_logits:
            logits = torch.empty(
                x.shape[0], _HC_MULT * _HC_MULT, dtype=torch.float32, device=x.device
            )
            scalars["logits_out"] = logits
        residual_out, layer_out, post_out, comb_out, q, sf = self.kernel(
            x, residual, post, comb, fn, hc_scale, hc_base, norm_weight, **scalars
        )
        if logits is not None:
            comb_out = logits
        return residual_out, layer_out, post_out, comb_out, (q, sf)

    def launch_deferred_sinkhorn(
        self,
        logits: torch.Tensor,
        *,
        eps: float,
        repeat: int,
        align_fleet: bool,
        prefetch=None,
    ) -> PendingComb:
        """comb [T,4,4] f32 = sinkhorn(logits) on this device's side stream, forked
        from the current stream here; ``PendingComb.join`` waits for it. The
        sinkhorn kernel first issues ``prefetch`` (an L2Prefetch) when given."""
        main = torch.cuda.current_stream()
        side = _sinkhorn_stream(logits.device)
        comb = torch.empty(
            logits.shape[0],
            _HC_MULT,
            _HC_MULT,
            dtype=torch.float32,
            device=logits.device,
        )
        side.wait_stream(main)
        with torch.cuda.stream(side):
            self.sinkhorn(
                logits,
                comb,
                eps=eps,
                repeat=repeat,
                align_fleet=align_fleet,
                prefetch=prefetch,
            )
            done = torch.cuda.Event()
            done.record(side)
        return PendingComb(comb=comb, logits=logits, done=done)


class PendingComb(msgspec.Struct, frozen=True):
    """A side-stream sinkhorn in flight. Holding ``logits`` (main-stream allocated,
    read by the side stream) until ``join`` keeps its block from being reused."""

    comb: torch.Tensor
    logits: torch.Tensor
    done: torch.cuda.Event

    def join(self) -> torch.Tensor:
        """The finished comb; the current stream waits for the side stream first."""
        torch.cuda.current_stream().wait_event(self.done)
        return self.comb


class DeferredSinkhorn(msgspec.Struct, frozen=True):
    """A side-stream sinkhorn of ``logits`` not launched yet."""

    gate: FusedArMhcGate
    logits: torch.Tensor
    eps: float
    repeat: int
    align_fleet: bool
    # L2Prefetch issued on the side stream with it (add_l2_prefetch); None = none.
    prefetch: object = None

    def launch(self) -> PendingComb:
        """Fork it from the current stream now."""
        return self.gate.launch_deferred_sinkhorn(
            self.logits,
            eps=self.eps,
            repeat=self.repeat,
            align_fleet=self.align_fleet,
            prefetch=self.prefetch,
        )


def add_l2_prefetch(
    work: DeferredSinkhorn,
    *,
    next_inputs: Optional[Tuple[torch.Tensor, ...]],
    residual: torch.Tensor,
    post: torch.Tensor,
) -> DeferredSinkhorn:
    """``work`` that also L2-prefetches what the next layer's MoE-side kernel reads
    first: ``next_inputs`` and this layer's attention-side ``residual`` / ``post``
    outputs (the deferred comb is written by the side sinkhorn, so already in L2).
    ``work`` unchanged when there is no next MoE-side kernel."""
    if next_inputs is None:
        return work
    from sglang.kernels.ops.attention.dsv4.ernie55_fused_ar_mhc import L2Prefetch

    tensors = (*next_inputs, residual, post)
    return msgspec.structs.replace(work, prefetch=L2Prefetch(tensors=tensors))


class SinkhornDeferral(msgspec.Struct, frozen=True):
    """Where one layer forward defers a sinkhorn; decided once at its top."""

    # The MoE-side kernel emits the attention comb as logits, which the
    # attention-side kernel of the same forward finishes in-kernel.
    attn: bool = False
    # The attention-side kernel emits its FFN comb as logits, finished on the
    # side stream and joined right after the MLP.
    ffn: bool = False


def plan_deferral(*, attn_runs: bool, moe_runs: bool) -> SinkhornDeferral:
    """Logits only flow between the fused kernels of one layer forward, so whether
    each side runs in that forward (token count and frozen config only) settles it."""
    return SinkhornDeferral(attn=attn_runs and moe_runs, ffn=attn_runs)


def small_path(num_tokens: int) -> bool:
    """Internal ``_mhc_small_path``: the small-M fused mHC kernels run only off
    deterministic / batch-invariant inference and for 1..LL_MAX_TOKENS rows."""
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.quantization.ernie55_halops_ll_gemm import LL_MAX_TOKENS
    from sglang.srt.runtime_context import get_exec

    if is_batch_invariant_mode_enabled():
        return False
    if get_exec().deterministic.enable_deterministic_inference:
        return False
    return 0 < num_tokens <= LL_MAX_TOKENS


def resolve_gates(
    *, hc_mult: int, hidden_size: int, mhc_fuse_post_pre: bool, fuse_input_norm: bool
) -> Tuple[FusedArMhcGate, FusedArMhcGate]:
    """Layer-build half of the (attention-side, MoE-side) gates. A layer the kernel
    does not cover runs both sides off (logged once); a missing dependency raises."""
    if not switch_on():
        return FusedArMhcGate(), FusedArMhcGate()
    if not _layer_covered(
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        mhc_fuse_post_pre=mhc_fuse_post_pre,
        fuse_input_norm=fuse_input_norm,
    ):
        return FusedArMhcGate(), FusedArMhcGate()
    try:
        from sglang.kernels.ops.attention.dsv4 import ernie55_fused_ar_mhc as kernels
    except ImportError as exc:
        raise RuntimeError(
            f"{_SWITCH}=1: the fused kernels are unavailable ({exc})"
        ) from exc
    attn = FusedArMhcGate(
        kernel=kernels.fused_ar_mhc_push,
        push_group=_push_group(),
        sinkhorn=kernels.fused_ar_mhc_sinkhorn,
    )
    return attn, FusedArMhcGate(kernel=kernels.fused_ar_mhc)


def _sinkhorn_stream(device: torch.device) -> torch.cuda.Stream:
    """This device's deferred-sinkhorn stream, created on the first call; that call
    must not be inside a graph capture (the warmup forwards come first)."""
    stream = _SINKHORN_STREAMS.get(device.index)
    if stream is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{_SWITCH}=1: the first deferred sinkhorn on {device} is inside "
                "a graph capture; run an eager forward first"
            )
        stream = torch.cuda.Stream(device=device)
        _SINKHORN_STREAMS[device.index] = stream
    return stream


def accepted_input_quant(
    quant: Optional[Tuple[torch.Tensor, torch.Tensor]],
    *,
    consumers: Sequence[torch.nn.Module],
    num_tokens: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``quant`` when every linear that reads the attention input takes a group-32
    tuple at ``num_tokens`` rows, else None (the attention quantizes its own input).
    Capture mode only: there the bf16 input would run the group-32 quant + the same
    LL GEMM, while eager forwards keep their DeepGEMM path. Token count, capture
    state and layer config only, so rank-invariant."""
    if quant is None or not consumers:
        return None
    from sglang.srt.layers.quantization.ernie55_halops_ll_gemm import (
        g32_prequant_ready,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    if not get_is_capture_mode() or not all(
        g32_prequant_ready(lin, num_tokens) for lin in consumers
    ):
        return None
    return quant


def log_moe_variant(*, layer_id: int, has_shared: bool, num_tokens: int) -> None:
    """INFO once per process and kernel variant of the MoE-side kernel (sh=1 folds
    a previous shared-expert term; the open tree never passes one)."""
    if has_shared in _LOGGED_MOE_VARIANTS:
        return
    _LOGGED_MOE_VARIANTS.add(has_shared)
    logger.info(
        "%s: MoE-side kernel variant sh=%d (first at layer %d, T=%d)",
        _SWITCH,
        int(has_shared),
        layer_id,
        num_tokens,
    )


def _layer_covered(
    *, hc_mult: int, hidden_size: int, mhc_fuse_post_pre: bool, fuse_input_norm: bool
) -> bool:
    misses = [
        why
        for ok, why in (
            (mhc_fuse_post_pre, "SGLANG_ERNIE55_SINGLE_BATCH_LL is off"),
            # The kernel always folds the input RMSNorm into its mhc_pre, which
            # the open _hc_pre only does under FUSE_INEXACT.
            (fuse_input_norm, "SGLANG_OPT_ERNIE5_5_FUSE_INEXACT is off"),
            (hc_mult == _HC_MULT, f"hc_mult={hc_mult}"),
            (hidden_size == _HIDDEN, f"hidden_size={hidden_size}"),
        )
        if not ok
    ]
    if misses:
        _warn_once(f"{_SWITCH}=1 skips a layer: " + ", ".join(misses))
    return not misses


def _push_group():
    """The attention-TP group whose push plane the attention-side kernel rides,
    or raise."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().attn_tp_group
    misses = []
    if group.world_size != _PUSH_WORLD:
        misses.append(f"attn_tp_size={group.world_size} (need {_PUSH_WORLD})")
    # Only the full tp group owns a custom all-reduce; a narrower attention-TP
    # group (DP attention) is built without one.
    ca_comm = getattr(group, "ca_comm", None)
    if not isinstance(ca_comm, CustomAllReduceV2) or ca_comm.disabled:
        misses.append("attention TP group has no enabled CustomAllReduceV2")
    if misses:
        raise RuntimeError(
            f"{_SWITCH}=1 needs the attention push all-reduce: " + ", ".join(misses)
        )
    return group


def _runs_ll_push(group, x: torch.Tensor) -> bool:
    """Whether ``group.all_reduce(x)`` reaches the CustomAllReduceV2 push kernel:
    the CUDA dispatch of GroupCoordinator.all_reduce (mscclpp and flashinfer
    preempt the custom all-reduce), then CustomAllReduceV2's algorithm pick. Reads
    only tensor metadata, capture state and group config, all rank-invariant."""
    from sglang.srt.layers.quantization.ernie55_halops_ll_gemm import LL_MAX_TOKENS

    comm = group.ca_comm
    mscclpp = group.pymscclpp_comm
    if mscclpp is not None and mscclpp.should_mscclpp_allreduce(x):
        return False
    if group._can_use_flashinfer_allreduce(x):
        return False
    if not comm.should_custom_ar(x) or comm.override_algo is not None:
        return False
    if x.shape[0] > LL_MAX_TOKENS:
        return False
    nbytes = x.numel() * x.element_size()
    if nbytes > comm.max_push_size:
        return False
    config = comm._pick_config(nbytes, comm._can_use_graph())
    return config is not None and config.algo.is_push()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        logger.warning(msg)

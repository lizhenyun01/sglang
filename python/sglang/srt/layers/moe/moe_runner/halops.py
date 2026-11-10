"""HALOps W4A8 MoE runner backend for sgl_open (``moe-runner-backend halops``).

Routes the ERNIE 5.5 routed experts through HALOps' ``w4a8_group_contiguous_gemm``
on the single packed-FP4 (E2M1) expert copy, so the *prefill* FFN reads the same
weights the decode ``low_latency_moe`` mega kernel reads -- eliminating the second
block128-FP8 expert copy (one FP4 copy ~32GB/rank vs fp8+fp4 ~101GB/rank).

Scope vs the internal ``halops_runner``:

* CONTIGUOUS (prefill/extend) ONLY. sgl_open decode (m in [1,16]) is served by the
  fused ``Ernie55MegaMoE`` ``low_latency_moe`` kernel, which runs *before* the
  standard FusedMoE runner and returns, so this runner is only ever reached on the
  prefill/extend path. The masked/decode grouped GEMM and the JIT
  ``situ_group32_quant_sfatom`` kernel (absent from sgl_open) are therefore not needed.
* The gemm1 -> gemm2 intermediate is rebuilt with torch SiTU (``SituAndMul``) +
  HALOps ``group_wise_quant(mode="w8a8-mxfp8", use_swizzle=True)`` -- the validated
  bridge from ``scripts/_test_w4a8_prefill.py`` -- instead of the fused JIT kernel.
* post_permute uses the plain ``post_reorder_deepgemm`` weighted combine (sgl_open's
  FusedMoE does its own all-reduce); the internal ``try_fused_combine_allreduce`` is
  internal-only and not ported.

Self-registering: ``register_moe_runner_core`` adds the "halops" backend name and the
core factory, and the module-level ``register_pre_permute``/``register_post_permute``
wire the standard<->halops permute pair. No edits to ``utils.py`` or ``runner.py``.
The runner reuses ``DeepGemmMoeQuantInfo`` (w13/w2 packed FP4 + group-32 fp32 scales)
and sgl_open's own ``ep_scatter`` / ``post_reorder_deepgemm`` scatter/combine glue,
swapping only the quant layout and the GEMM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.runner import register_moe_runner_core
from sglang.srt.layers.moe.utils import (
    register_moe_runner_backend_name,
    resolve_moe_runner_backend,
)
from sglang.srt.layers.quantization.ernie55_blockwise_fp8 import (
    fp4_weight_scale_to_sfatom,
)

_BACKEND_NAME = "halops"
try:
    _BACKEND = resolve_moe_runner_backend(_BACKEND_NAME)
except ValueError:
    register_moe_runner_backend_name(_BACKEND_NAME)
    _BACKEND = resolve_moe_runner_backend(_BACKEND_NAME)

_CONTIG_BLOCK_E = 128
# Gate/up pairing granularity of the decode ``low_latency_moe`` kernel
# (HALOps ``_PAIR_GRAN``, fixed by its TMEM read pattern). The prefill contiguous
# GEMM is a plain layout-agnostic grouped matmul, so the runner consumes w13 in
# this same gran-8 pair layout -- the single packed-FP4 copy decode already holds
# -- and only has to invert it before SiTU.
_PAIR_GRAN = 8
# SfAtom-repacked expert weight scale, cached by the source tensor's id (the frozen
# weight-scale Parameter). Interleaved-w13 ids tracked so the repack runs once.
_SFB_CACHE: dict = {}
_GATEUP_INTERLEAVED: set = set()


def _interleave_idx(n: int, device) -> torch.Tensor:
    """Gran-8 pair row order for a ``[gate | up]`` block of ``n`` rows.

    Byte-identical to decode's ``_interleave_up_for_pair_layout`` (gate rows
    ``[8p, 8p+8)`` then up rows ``[8p, 8p+8)`` for each pair block ``p``), so the
    packed-FP4 w13 the prefill contiguous GEMM reads is the exact tensor the decode
    ``low_latency_moe`` kernel reads -- one copy, not two packings.
    """
    half = n // 2
    assert half % _PAIR_GRAN == 0, f"N={half} must be a multiple of {_PAIR_GRAN}"
    blocks = half // _PAIR_GRAN
    gate = torch.arange(half, device=device).reshape(blocks, _PAIR_GRAN)
    up = (torch.arange(half, device=device) + half).reshape(blocks, _PAIR_GRAN)
    return torch.stack([gate, up], dim=1).reshape(n)


def _deinterleave_cols(gateup: torch.Tensor) -> torch.Tensor:
    """Gran-8 pair cols ``[g0..g7, u0..u7, g8..g15, ...]`` -> block ``[gate | up]``."""
    lead = gateup.shape[:-1]
    half = gateup.shape[-1] // 2
    x = gateup.reshape(*lead, half // _PAIR_GRAN, 2, _PAIR_GRAN)
    gate = x[..., 0, :].reshape(*lead, half)
    up = x[..., 1, :].reshape(*lead, half)
    return torch.cat([gate, up], dim=-1)


def _sfb(scale_fp32: torch.Tensor) -> torch.Tensor:
    """SfAtom-repacked expert weight scale, cached by the source tensor's id."""
    key = id(scale_fp32)
    cached = _SFB_CACHE.get(key)
    if cached is None:
        cached = fp4_weight_scale_to_sfatom(scale_fp32)
        _SFB_CACHE[key] = cached
    return cached


def precompute_expert_sfb(layer) -> None:
    """Load-time prep of the packed-FP4 expert weights for the contiguous GEMM.

    Gate/up-interleaves the packed w13 weight and its group-32 scale (must precede
    the scale repack, runs once per weight id), then SfAtom-repacks and caches the
    w13/w2 scales. Called from the W4A8 online-quant weight hook, so the allocating
    repack never runs inside a CUDA-graph capture on the first forward.
    """
    w13 = getattr(layer, "w13_weight", None)
    if w13 is not None and id(w13) not in _GATEUP_INTERLEAVED:
        idx = _interleave_idx(w13.shape[1], w13.device)
        w13.data = w13.data.index_select(1, idx).contiguous()
        scale = getattr(layer, "w13_weight_scale_inv", None)
        if scale is not None:
            assert id(scale) not in _SFB_CACHE, "w13 scale repacked before interleave"
            scale.data = scale.data.index_select(1, idx).contiguous()
        _GATEUP_INTERLEAVED.add(id(w13))

    for attr in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        scale = getattr(layer, attr, None)
        if scale is not None:
            _sfb(scale)


@dataclass
class HALOpsRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    sfa: torch.Tensor
    m_indices: torch.Tensor

    @property
    def runner_backend(self):
        return _BACKEND


@dataclass
class HALOpsRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self):
        return _BACKEND


def _situ(gateup: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    """Torch rebuild of the absent ``situ_group32_quant_sfatom`` activation half:
    deinterleave the [gate|up]-interleaved gemm1 output, then SiTU-and-mul. A
    ``gemm1_clamp_limit`` (linear_beta) of 0 means "disabled" -> None (no up clamp,
    and no div-by-zero)."""
    from sglang.srt.layers.activation import SituAndMul

    lb = linear_beta if linear_beta and linear_beta > 0 else None
    return SituAndMul(beta, lb).forward_native(_deinterleave_cols(gateup))


class HALOpsRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        assert config.is_gated, "HALOps MoE runner is gated-only"
        assert config.activation == "situ", (
            f"HALOps MoE runner only wires SiTU activation, got {config.activation}"
        )

    def run(
        self,
        runner_input: HALOpsRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> HALOpsRunnerOutput:
        return HALOpsRunnerOutput(
            hidden_states=self._run_w4a8(runner_input, quant_info, running_state)
        )

    def _run_w4a8(
        self,
        runner_input: HALOpsRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from halops import group_wise_quant, w4a8_group_contiguous_gemm

        beta = float(self.config.gemm1_alpha)
        linear_beta = float(self.config.gemm1_clamp_limit)

        x = runner_input.hidden_states
        w13 = quant_info.w13_weight.view(torch.uint8)
        w2 = quant_info.w2_weight.view(torch.uint8)
        if (
            w13.shape[2] * 2 != x.shape[1]
            or id(quant_info.w13_weight) not in _GATEUP_INTERLEAVED
        ):
            raise RuntimeError(
                "HALOps MoE runner needs gate/up-interleaved packed fp4 expert "
                f"weights [E, N, K // 2] with K={x.shape[1]}, got "
                f"{tuple(w13.shape)}; precompute_expert_sfb did not run at load"
            )
        sfb13 = _sfb(quant_info.w13_scale)
        sfb2 = _sfb(quant_info.w2_scale)

        sfa = runner_input.sfa
        expert_ids = runner_input.m_indices
        n = w13.shape[1]
        k2 = w2.shape[1]
        tokens = x.shape[0]
        device = x.device

        # gemm1: w4a8 contiguous on the gate/up-interleaved packed w13.
        gateup = torch.empty((tokens, n), dtype=torch.bfloat16, device=device)
        w4a8_group_contiguous_gemm(x, w13, sfa, sfb13, expert_ids, gateup)

        # SiTU bridge (torch) + group-32 FP8 + SfAtom quant for gemm2's A operand.
        act = _situ(gateup, beta, linear_beta).bfloat16().contiguous()
        del gateup
        down_in, sfa2 = group_wise_quant(act, mode="w8a8-mxfp8", use_swizzle=True)

        # gemm2: w4a8 contiguous on w2.
        down_out = torch.empty((tokens, k2), dtype=torch.bfloat16, device=device)
        w4a8_group_contiguous_gemm(down_in, w2, sfa2, sfb2, expert_ids, down_out)
        return down_out

    @property
    def runner_backend(self):
        return _BACKEND


def _contiguous_layout(num_assignments: int, num_experts: int) -> tuple[int, int]:
    """(block_e, rows) of the packed prefill input for the HALOps contiguous GEMM.

    HALOps picks the tile from ``rows // num_experts`` and reads one expert id per
    tile, so the per-expert M must only admit tiles that never straddle two expert
    segments (``block_e``-aligned): 64 rows -> 64-aligned; 128*odd -> 128; 128*even
    -> 256-aligned. Every expert costs at least 64 rows whatever the routed count.
    """
    from sglang.srt.layers.moe.moe_runner.deep_gemm import _get_compact_all_tokens

    unit = num_experts * _CONTIG_BLOCK_E
    half = _CONTIG_BLOCK_E // 2
    if _get_compact_all_tokens(num_assignments, num_experts, half) <= unit // 2:
        return half, unit // 2
    rows = _get_compact_all_tokens(num_assignments, num_experts, _CONTIG_BLOCK_E)
    j = -(-rows // unit)
    if j % 2 == 1:
        return _CONTIG_BLOCK_E, unit * j
    block_e = 2 * _CONTIG_BLOCK_E
    rows = _get_compact_all_tokens(num_assignments, num_experts, block_e)
    j = -(-rows // unit)
    return block_e, unit * (j + j % 2)


@register_pre_permute("standard", "halops")
def pre_permute_standard_to_halops(
    dispatch_output,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> HALOpsRunnerInput:
    """Compact contiguous scatter (bf16) + group-32 SfAtom FP8 quant for prefill.

    sgl_open decode runs through the fused mega kernel before this runner, so only
    the prefill/extend contiguous layout is produced here. Reuses sgl_open's compact
    ``ep_scatter`` to bf16, then ``group_wise_quant`` for gemm1's A operand. Trailing
    pad rows carry m_indices -1; clamped to a valid expert they compute harmless
    garbage never gathered back by src2dst, keeping the contiguous GEMM (which does
    not skip negative ids) in bounds.
    """
    from halops import group_wise_quant

    from sglang.kernels.ops.moe.ep_moe_kernels import (
        ep_scatter,
        fused_moe_dispatch_index,
    )

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output

    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["hidden_states_device"] = hidden_states.device
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    device = hidden_states.device
    num_experts = runner_config.num_local_experts
    block_e, all_tokens = _contiguous_layout(topk_ids.numel(), num_experts)

    valid_tokens_per_expert, _ = fused_moe_dispatch_index(topk_ids, num_experts, 1)
    tokens_per_expert = (
        (valid_tokens_per_expert + block_e - 1) // block_e * block_e
    ).to(torch.int32)
    tokens_per_expert[-1] = tokens_per_expert[-1] + (
        all_tokens - tokens_per_expert.sum()
    )

    k = hidden_states.size(1)
    packed_input = torch.empty((all_tokens, k), device=device, dtype=torch.bfloat16)
    packed_input_scale = torch.empty(
        (all_tokens, 1), device=device, dtype=torch.float32
    )
    expert_start_loc = torch.empty(num_experts, device=device, dtype=torch.int32)
    m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
    src2dst = torch.empty_like(topk_ids, dtype=torch.int32)
    block_k = quant_info.block_shape[1] if quant_info.block_shape else 128
    ep_scatter(
        hidden_states,
        None,
        topk_ids,
        tokens_per_expert,
        valid_tokens_per_expert,
        expert_start_loc,
        packed_input,
        packed_input_scale,
        m_indices,
        src2dst,
        scale_ue8m0=True,
        quant_block_size=block_k,
    )
    running_state["src2dst"] = src2dst
    running_state["all_tokens"] = all_tokens

    # "w8a8-mxfp8" names the activation side only: group-32 FP8 + SfAtom UE8M0, the
    # A operand of the w4a8 contiguous GEMM.
    x_fp8, sfa = group_wise_quant(packed_input, mode="w8a8-mxfp8", use_swizzle=True)
    return HALOpsRunnerInput(
        hidden_states=x_fp8,
        sfa=sfa,
        m_indices=m_indices.clamp_min(0),
    )


@register_post_permute("halops", "standard")
def post_permute_halops_to_standard(
    runner_output: HALOpsRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    """Weighted combine back to per-token rows (plain ``post_reorder_deepgemm``).

    The internal fused combine+all-reduce is internal-only; sgl_open's FusedMoE does
    its own all-reduce after this returns, so the plain reorder is sufficient.
    """
    from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils import dispose_tensor

    hidden_states_shape = running_state["hidden_states_shape"]
    hidden_states_dtype = running_state["hidden_states_dtype"]
    hidden_states_device = running_state["hidden_states_device"]
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]
    src2dst = running_state["src2dst"]
    routed_scaling = (
        runner_config.routed_scaling_factor
        if runner_config.routed_scaling_factor is not None
        else 1.0
    )

    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
    post_reorder_deepgemm(
        runner_output.hidden_states,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        runner_config.top_k,
        hidden_states_shape[0],
        hidden_states_shape[1],
        routed_scaling,
    )
    dispose_tensor(runner_output.hidden_states)
    return StandardCombineInput(hidden_states=output)


register_moe_runner_core(_BACKEND_NAME, HALOpsRunnerCore)

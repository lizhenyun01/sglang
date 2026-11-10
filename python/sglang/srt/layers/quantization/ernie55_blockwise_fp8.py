"""ERNIE 5.5 online 128x128 blockwise-FP8 quantisation helpers.

Re-home of the internal ``blockwise_fp8_patch`` quant math into the official
tree, as a standalone module (no plugin / monkeypatch). It turns a BF16
checkpoint's expert / dense weights into the blockwise-FP8 form the fused
``Ernie55MegaMoE`` kernel consumes: an FP8 ``float8_e4m3fn`` weight plus a
``weight_scale_inv`` fp32 128x128 block-scale table.

Pure functions only; callers decide where/when to quantise (ernie55 post-load,
or an offline checkpoint converter). Blackwell/DeepGEMM UE8M0 scales are
supported via ``use_ue8m0_scale`` for the mega ``w8a8-mxfp8`` recipe; the
default ``w8a8-blockwise`` recipe uses plain fp32 scales.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    ceil_to_ue8m0,
    transform_scale_ue8m0,
)
from sglang.srt.utils.common import ceil_align, ceil_div

BLOCKWISE_FP8_WEIGHT_BLOCK_SIZE: List[int] = [128, 128]
_FP8 = torch.float8_e4m3fn


def per_block_no_cast_to_fp8(
    x: torch.Tensor, use_ue8m0_scale: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """128x128 block FP8 quant of a 2D weight. Returns (fp8 weight, fp32 scale).

    ``use_ue8m0_scale`` rounds each block scale up to a power of two (the packed
    UE8M0 format DeepGEMM needs on Blackwell); otherwise precise amax/448 scales.
    """
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_align(m, 128), ceil_align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0) if use_ue8m0_scale else x_amax / 448.0
    x_scaled = (x_view * (1.0 / sf)).to(_FP8)
    out_w = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    out_s = sf.view(x_view.size(0), x_view.size(2))
    return out_w, out_s


def quant_weight_block_fp8(
    weight_bf16: torch.Tensor, use_ue8m0_scale: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blockwise-FP8 quant of a weight with leading batch dims (e.g. an expert
    stack ``[E, n, k]`` or a plain ``[n, k]`` linear)."""
    assert weight_bf16.dtype == torch.bfloat16, f"{weight_bf16.dtype=}"
    *batch_dims, n, k = weight_bf16.shape
    w_flat = weight_bf16.reshape((-1, k))
    out_w_flat, out_s_flat = per_block_no_cast_to_fp8(w_flat, use_ue8m0_scale)
    out_w = out_w_flat.view((*batch_dims, n, k))
    out_s = out_s_flat.view(
        (
            *batch_dims,
            ceil_div(n, BLOCKWISE_FP8_WEIGHT_BLOCK_SIZE[0]),
            ceil_div(k, BLOCKWISE_FP8_WEIGHT_BLOCK_SIZE[1]),
        )
    )
    return out_w, out_s


# ---------------------------------------------------------------------------
# Online-quant integration (mirrors the dsv4 mxfp8-HALOps pattern: a load-time
# quantizer invoked from ``process_weights_after_loading`` on a BF16 checkpoint).
# ---------------------------------------------------------------------------


def online_block_fp8_enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_ERNIE55_ONLINE_BLOCKWISE_FP8.get())


def _is_blackwell_deepgemm() -> bool:
    from sglang.srt.layers.deep_gemm_wrapper import DEEPGEMM_SCALE_UE8M0

    return DEEPGEMM_SCALE_UE8M0


def _make_block_scale_param(scale: torch.Tensor, weight_loader):
    """Wrap a block scale as a loader-aware BlockQuantScaleParameter so a pushed
    (un-sharded) scale narrows to this TP rank on /update_weights_from_tensor;
    a None loader (ReplicatedLinear) degrades to a plain Parameter."""
    from torch.nn import Parameter

    from sglang.srt.layers.parameter import BlockQuantScaleParameter

    if weight_loader is None:
        return Parameter(scale, requires_grad=False)
    return BlockQuantScaleParameter(
        data=scale, input_dim=1, output_dim=0, weight_loader=weight_loader
    )


def _linear_supports_deepgemm_ue8m0(layer) -> bool:
    """Whether this linear can consume the Blackwell packed (UE8M0 int32) scale."""
    if not _is_blackwell_deepgemm():
        return False
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
    )
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0

    method = getattr(layer, "quant_method", None)
    return bool(
        method is not None
        and getattr(method, "w8a8_block_fp8_linear", None)
        is deepgemm_w8a8_block_fp8_linear_with_fallback
        and should_deepgemm_weight_requant_ue8m0(
            weight_block_size=BLOCKWISE_FP8_WEIGHT_BLOCK_SIZE,
            output_dtype=getattr(layer, "orig_dtype", None),
            weight_shape=layer.weight.shape,
        )
    )


def quantize_linear_online_block_fp8(layer) -> None:
    """Online 128x128 blockwise-FP8 quant of a BF16 linear ``layer.weight`` in
    place, registering ``weight_scale_inv``. Blackwell stores UE8M0-packed int32
    scales (what ``deepgemm`` and ``Ernie55MegaMoE._scale_fp32`` both read)."""
    from sglang.srt.layers.quantization.fp8_utils import (
        quant_weight_ue8m0,
        transform_scale_ue8m0,
    )

    weight_data = layer.weight.data
    if weight_data.dtype != torch.bfloat16:
        weight_data = weight_data.to(torch.bfloat16)
    use_ue8m0 = _linear_supports_deepgemm_ue8m0(layer)
    if use_ue8m0:
        qweight, scale = quant_weight_ue8m0(
            weight_data, BLOCKWISE_FP8_WEIGHT_BLOCK_SIZE
        )
        scale = transform_scale_ue8m0(scale, mn=qweight.shape[-2])
    else:
        qweight, scale = quant_weight_block_fp8(weight_data, use_ue8m0_scale=False)

    layer.weight.data = qweight
    layer.weight.requires_grad_(False)
    scale_param = _make_block_scale_param(
        scale, getattr(layer.weight, "weight_loader", None)
    )
    scale_param.format_ue8m0 = use_ue8m0
    if getattr(layer, "weight_scale_inv", None) is not None:
        layer.register_parameter("weight_scale_inv", None)
    layer.register_parameter("weight_scale_inv", scale_param)
    layer.input_scale = None


def _moe_supports_deepgemm_ue8m0(layer) -> bool:
    if not _is_blackwell_deepgemm():
        return False
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0

    return all(
        should_deepgemm_weight_requant_ue8m0(
            weight_block_size=BLOCKWISE_FP8_WEIGHT_BLOCK_SIZE,
            output_dtype=torch.bfloat16,
            weight_shape=w.shape[-2:],
        )
        for w in (layer.w13_weight, layer.w2_weight)
    )


def _attach_moe_scale_loader(layer, prefix: str) -> None:
    """Mirror sibling weight's FusedMoE loader onto the block-scale param."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import (
        FusedMoeWeightScaleSupported,
    )

    scale = getattr(layer, f"{prefix}_weight_scale_inv", None)
    weight = getattr(layer, f"{prefix}_weight", None)
    if scale is None or weight is None:
        return
    weight_loader = getattr(weight, "weight_loader", None)
    if weight_loader is not None:
        scale.weight_loader = weight_loader
    scale.quant_method = FusedMoeWeightScaleSupported.BLOCK.value


# ---------------------------------------------------------------------------
# W4A8 / MXFP4 routed-expert quantiser (the ``w4a8-mxfp4`` mega recipe). Pure
# torch, transcribed bit-identical from DeepGEMM ``deep_gemm/utils/math.py`` (the
# internal ``blockwise_fp8_patch`` FP4 path). Turns a BF16 expert stack into the
# packed E2M1 weight + fp32 power-of-two 1x32 group scale that HALOps'
# ``low_latency_moe`` (quant_mode="w4a8-mxfp4") and ``w4a8_group_contiguous_gemm``
# consume -- ONE FP4 copy shared by decode and (once its runner lands) prefill.
# ---------------------------------------------------------------------------


def _ceil_to_ue8m0_pow2(x: torch.Tensor) -> torch.Tensor:
    """Round each magnitude up to a power of two (UE8M0), bit-identical to DeepGEMM."""
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """e2m1 codebook {0,.5,1,1.5,2,3,4,6}, boundaries (0.25,0.75,1.25,1.75,2.5,3.5,5.0),
    sign bit 3. Returns the 4-bit codes as int8 (one per element, unpacked)."""
    ax = x.abs()
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (ax > boundary).to(torch.uint8)
    sign = (x < 0) & (code != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def per_token_cast_to_fp4(
    x: torch.Tensor, use_ue8m0: bool = True, gran_k: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row FP4 quant with per-``gran_k`` scales. Returns packed e2m1 codes as
    ``int8 (n, k // 2)`` (low nibble = even k, high nibble = odd k) plus the fp32
    scale ``(n, k // gran_k)`` (UE8M0-rounded when ``use_ue8m0``)."""
    m, n = x.shape
    assert n % 2 == 0
    padded_n = ceil_align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = _ceil_to_ue8m0_pow2(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, : n // 2].contiguous(), sf


def quant_experts_fp4(stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 experts ``[E, N, K]`` -> ``(packed FP4 uint8 [E, N, K//2], fp32 power-of-two
    1x32 scale [E, N, K//32])``, one expert at a time on the GPU (bounds the temporaries;
    the source stack may be CPU-staged). The outputs are the kernel's device operands."""
    e, n, k = stack.shape
    gpu = torch.device("cuda", torch.cuda.current_device())
    packed = torch.empty(e, n, k // 2, dtype=torch.uint8, device=gpu)
    scale = torch.empty(e, n, k // 32, dtype=torch.float32, device=gpu)
    for i in range(e):
        src = stack[i]
        if src.dtype != torch.bfloat16:
            src = src.to(torch.bfloat16)
        qi, si = per_token_cast_to_fp4(src.to(gpu), use_ue8m0=True, gran_k=32)
        packed[i].copy_(qi.view(torch.uint8))
        scale[i].copy_(si)
    return packed, scale


# ---------------------------------------------------------------------------
# CUTLASS SfAtom scale repack for the packed-FP4 expert weight scale. Transcribed
# verbatim from the internal ``halops_moe`` (itself a byte-identical torch port of
# ``HALOps/tests/utils/gemm_utils.py::reorder_sf_to_cutlass``): the fp32 power-of-two
# group-32 scale that ``quant_experts_fp4`` emits -> UE8M0 byte -> CUTLASS SfAtom
# uint8, the ``sfb`` operand ``w4a8_group_{contiguous,masked}_gemm`` reads.
# ---------------------------------------------------------------------------

_SF_BLK_MN = 128
_SF_TILE_BYTES = 512


def _ue8m0_byte_from_fp32(scale_fp32: torch.Tensor) -> torch.Tensor:
    """UE8M0 byte (biased exponent) of a power-of-two fp32 scale.

    ``quant_experts_fp4`` stores the scale already rounded to a power of two, so
    the fp32 mantissa is zero and the biased exponent bits ``(bits >> 23) & 0xFF``
    are exactly the UE8M0 byte the kernel dequantizes as ``2^(byte - 127)``.
    """
    bits = scale_fp32.contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def reorder_sf_to_cutlass(byte: torch.Tensor, mn_dim: int, kb_dim: int) -> torch.Tensor:
    """Reorder a UE8M0 byte scale ``(..., mn_dim, kb_dim)`` into CUTLASS SfAtom.

    Vectorised on GPU so it can run once per layer at load. When ``mn_dim`` is not
    a multiple of 128 the result is padded to the next multiple (the kernel's TMA
    descriptor reads the full 128-wide atom)."""
    assert kb_dim % 4 == 0, f"kb_dim must be a multiple of 4, got {kb_dim}"
    lead = byte.reshape(-1, mn_dim, kb_dim)
    b = lead.shape[0]
    mn_padded = ((mn_dim + _SF_BLK_MN - 1) // _SF_BLK_MN) * _SF_BLK_MN
    if mn_padded != mn_dim:
        padded = byte.new_zeros((b, mn_padded, kb_dim))
        padded[:, :mn_dim, :] = lead
    else:
        padded = lead

    device = byte.device
    n_idx = torch.arange(mn_padded, device=device).view(mn_padded, 1)
    kb_idx = torch.arange(kb_dim, device=device).view(1, kb_dim)
    n_local = n_idx % _SF_BLK_MN
    num_k_tiles = kb_dim // 4
    cutlass_byte = (
        (n_idx // _SF_BLK_MN * num_k_tiles + kb_idx // 4) * _SF_TILE_BYTES
        + (n_local % 32) * 16
        + (n_local // 32) * 4
        + kb_idx % 4
    )

    dest = cutlass_byte.reshape(-1).long()
    out = padded.new_empty((b, mn_padded * kb_dim))
    out[:, dest] = padded.reshape(b, -1)
    out = out.reshape(b, mn_padded, kb_dim)
    tail = (mn_padded, kb_dim)
    return out.reshape(*byte.shape[:-2], *tail)


def fp4_weight_scale_to_sfatom(scale_fp32: torch.Tensor) -> torch.Tensor:
    """EB5 online-FP4 expert scale ``(E, n, k//32)`` fp32 -> uint8 SfAtom.

    One-time load-side repack: fp32 UE8M0-rounded -> UE8M0 byte -> SfAtom. ``n`` is
    the SfAtom MN axis, ``k//32`` the K-group axis."""
    assert scale_fp32.dim() == 3, f"expected (E, n, kb), got {tuple(scale_fp32.shape)}"
    _, n, kb = scale_fp32.shape
    return reorder_sf_to_cutlass(_ue8m0_byte_from_fp32(scale_fp32), n, kb)


def quantize_moe_experts_online_block_fp8(layer) -> None:
    """Online 128x128 blockwise-FP8 quant of BF16 MoE experts in place, one
    expert at a time (peak = fp8_all + scale_all + bf16_one_expert), registering
    ``w13_weight_scale_inv``/``w2_weight_scale_inv``."""
    from torch.nn import Parameter

    from sglang.srt.layers.quantization.fp8_utils import (
        quant_weight_ue8m0,
        transform_scale_ue8m0,
    )

    from sglang.srt.environ import envs

    def _transform_ue8m0_inplace(param, mn):
        param.data = transform_scale_ue8m0(param.data, mn=mn)

    use_ue8m0 = _moe_supports_deepgemm_ue8m0(layer)
    # With the fused Ernie55MegaMoE kernel on, the routed experts never reach
    # deepgemm: the m>16 prefill fallback runs the triton ``fused_experts`` (which
    # wants a plain ``[E, N/128, K/128]`` block scale) and decode runs the HALOps
    # mega kernel (which reads that same plain scale via ``_scale_fp32``). So keep
    # the power-of-two block scale in its plain layout and skip the deepgemm
    # TMA-packing swizzle -- that packed int32 layout only fits the deepgemm GEMM
    # and makes the triton kernel's ``cdiv(N,128) == scale.shape[-2]`` assert fail.
    pack_for_deepgemm = use_ue8m0 and not bool(envs.SGLANG_ERNIE55_MEGA_MOE.get())
    w13_src, w2_src = layer.w13_weight.data, layer.w2_weight.data
    on_cpu = w13_src.device.type == "cpu"
    gpu = torch.device("cuda") if on_cpu else None
    E = w13_src.shape[0]
    w13_q_all = w13_s_all = w2_q_all = w2_s_all = None

    def _q(w):
        if w.dtype != torch.bfloat16:
            w = w.to(torch.bfloat16)
        if on_cpu:
            w = w.to(gpu)
        if use_ue8m0:
            return quant_weight_ue8m0(w, BLOCKWISE_FP8_WEIGHT_BLOCK_SIZE)
        return quant_weight_block_fp8(w, use_ue8m0_scale=False)

    for e in range(E):
        w13_q, w13_s = _q(w13_src[e])
        w2_q, w2_s = _q(w2_src[e])
        if w13_q_all is None:
            w13_q_all = torch.empty(w13_src.shape, dtype=w13_q.dtype, device=w13_q.device)
            w2_q_all = torch.empty(w2_src.shape, dtype=w2_q.dtype, device=w2_q.device)
            w13_s_all = torch.empty((E, *w13_s.shape), dtype=w13_s.dtype, device=w13_s.device)
            w2_s_all = torch.empty((E, *w2_s.shape), dtype=w2_s.dtype, device=w2_s.device)
        w13_q_all[e].copy_(w13_q); w2_q_all[e].copy_(w2_q)
        w13_s_all[e].copy_(w13_s); w2_s_all[e].copy_(w2_s)
        del w13_q, w13_s, w2_q, w2_s

    # Assign via ``.data`` so the FusedMoE weight_loader set at create_weights
    # survives (a fresh Parameter drops it and breaks /update_weights_from_tensor).
    layer.w13_weight.data = w13_q_all
    layer.w2_weight.data = w2_q_all
    if getattr(layer, "w13_weight_scale_inv", None) is not None:
        layer.w13_weight_scale_inv.data = w13_s_all
        layer.w2_weight_scale_inv.data = w2_s_all
    else:
        layer.w13_weight_scale_inv = Parameter(w13_s_all, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_s_all, requires_grad=False)
    if pack_for_deepgemm:
        _transform_ue8m0_inplace(layer.w13_weight_scale_inv, mn=layer.w13_weight.shape[-2])
        _transform_ue8m0_inplace(layer.w2_weight_scale_inv, mn=layer.w2_weight.shape[-2])
    layer.w13_weight_scale_inv.format_ue8m0 = pack_for_deepgemm
    layer.w2_weight_scale_inv.format_ue8m0 = pack_for_deepgemm
    _attach_moe_scale_loader(layer, "w13")
    _attach_moe_scale_loader(layer, "w2")
    layer.w13_input_scale = None
    layer.w2_input_scale = None


def quantize_moe_experts_online_w4a8_fp4(layer) -> None:
    """Online W4A8 quant of BF16 MoE experts for the ``w4a8-mxfp4`` *prefill* path
    (``SGLANG_ERNIE55_W4A8_PREFILL``).

    Produces ONE packed-FP4 (E2M1) expert copy with 1x32 fp32 power-of-two scales,
    in place of the block128-FP8 copy, and lays it out exactly as the decode
    ``low_latency_moe`` kernel reads it (gate/up gran-8 pair-interleaved ``w13``), so
    the prefill HALOps ``w4a8_group_contiguous_gemm`` consumes it directly.
    ``precompute_expert_sfb`` does the in-place gran-8 interleave of ``w13`` + its
    scale and caches the SfAtom weight-scale tables the GEMM needs, so the first
    prefill forward (possibly under CUDA-graph capture) allocates nothing."""
    from torch.nn import Parameter

    from sglang.srt.layers.moe.moe_runner.halops import precompute_expert_sfb

    if layer.w13_weight.dtype != torch.bfloat16:
        return  # a pre-serialized FP8/FP4 checkpoint: nothing to quantize online
    w13d, s13 = quant_experts_fp4(layer.w13_weight.data)
    w2d, s2 = quant_experts_fp4(layer.w2_weight.data)
    layer.w13_weight.data = w13d
    layer.w2_weight.data = w2d
    if getattr(layer, "w13_weight_scale_inv", None) is not None:
        layer.w13_weight_scale_inv.data = s13
        layer.w2_weight_scale_inv.data = s2
    else:
        layer.w13_weight_scale_inv = Parameter(s13, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(s2, requires_grad=False)
    layer.w13_weight_scale_inv.format_ue8m0 = False
    layer.w2_weight_scale_inv.format_ue8m0 = False
    layer.w13_input_scale = None
    layer.w2_input_scale = None
    precompute_expert_sfb(layer)

"""InferRock w8a8 MXFP8 dense linear runner (Blackwell SM100)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.layers.quantization.inferrock_mxfp8 import (
    INFERROCK_K_ALIGN,
    INFERROCK_SFATOM_ROW_ALIGN,
    load_inferrock_gemm,
    pad_rows,
    swizzle_scale_to_sfatom,
)
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import ceil_div


def dense_gemm_plan(*, m: int, n: int, has_bias: bool) -> Optional[Tuple[int, bool]]:
    """``(M alignment, synthesize a zero bias)`` for this call, or None if no tile fits.

    Mirrors the w8a8 tile heuristic in InferRock ``csrc/kernel/gemm/dense_gemm.cu``:

    ==================  =============  =======  =======
    call                tile           M align  N align
    ==================  =============  =======  =======
    bias                256x128x128        256      128
    no bias, M >= 4096  256x256x128        256      256
    no bias, M < 4096   128x256x128        128      256
    ==================  =============  =======  =======

    The caller pads M up, so only N decides whether the kernel can run at all.
    When N is 128- but not 256-aligned no no-bias tile accepts it, so a zero bias
    is synthesized to reach the 256x128x128 tile.

    The SwapAB tile (M < 128) is unreachable: the swizzled SfAtom activation
    scale already forces M to a multiple of 128.
    """
    if n % INFERROCK_SFATOM_ROW_ALIGN != 0:
        return None
    if has_bias:
        return 256, False
    if n % 256 != 0:
        return 256, True
    return (256 if m >= 4096 else 128), False


def _flashinfer_fallback(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_fallback: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Serve an MXFP8 linear no InferRock tile can take, via FlashInfer CUTLASS.

    FlashInfer consumes the interleaved scale layout, not the canonical row-major
    one the InferRock load path keeps in ``weight_scale_inv``, so swizzle here:
    this is the cold path by construction and the scale is ``[N, K // 32]`` bytes.

    A prequantized activation is rejected -- its UE8M0 scale is in canonical
    layout and FlashInfer would silently consume it as if it were swizzled.
    """
    if input_scale is not None:
        raise RuntimeError(
            "InferRock MXFP8 has no tile for this GEMM and the FlashInfer fallback "
            "cannot take an already-quantized activation (its UE8M0 scale is in "
            "canonical, not swizzled, layout). Re-run with "
            "--fp8-gemm-backend=flashinfer_cutlass."
        )
    if weight_scale_fallback is None:
        raise RuntimeError(
            "InferRock MXFP8 has no tile for this GEMM and no canonical UE8M0 weight "
            "scale was kept for the FlashInfer fallback. Re-run with "
            "--fp8-gemm-backend=flashinfer_cutlass."
        )

    from flashinfer import block_scale_interleave

    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    # FlashInfer's activation quantizer only takes bf16/fp16; fp32 is one of the
    # reasons we are on this path at all, so downcast and restore the caller's dtype.
    orig_dtype = input.dtype
    if orig_dtype == torch.float32:
        input = input.to(torch.bfloat16)

    # 1D on purpose: flashinfer_mxfp8_blockscaled_linear transposes a 2D scale, and
    # block_scale_interleave already returns the flat interleaved form.
    weight_scale = block_scale_interleave(
        weight_scale_fallback.contiguous()
    ).contiguous()
    output = flashinfer_mxfp8_blockscaled_linear(
        input=input,
        weight=weight,
        weight_scale=weight_scale,
        input_scale=None,
        bias=bias,
    )
    return output if output.dtype == orig_dtype else output.to(orig_dtype)


def inferrock_w8a8_mxfp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    weight_scale_fallback: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MXFP8 dense linear via InferRock ``dense_gemm`` (Blackwell SM100).

    ``weight`` is fp8_e4m3 ``[N, K]`` and ``weight_scale`` is in InferRock's
    SfAtom layout (prepared at load time). Signature mirrors
    ``_deepgemm_w8a8_mxfp8_linear_with_fallback``.
    """
    n, k = weight.shape
    m = input.numel() // input.shape[-1]

    prequantized = input_scale is not None
    if prequantized:
        # Activation arrives as fp8 + canonical UE8M0 scale; such a layer outputs bf16.
        activation_supported = (
            input.dtype == torch.float8_e4m3fn and input_scale.dtype == torch.uint8
        )
        output_dtype = torch.bfloat16
    else:
        activation_supported = input.dtype in (torch.bfloat16, torch.float16)
        output_dtype = input.dtype

    plan = dense_gemm_plan(m=m, n=n, has_bias=bias is not None)
    if (
        plan is None
        or k % INFERROCK_K_ALIGN != 0
        or not activation_supported
        or not get_platform().is_sm100
    ):
        return _flashinfer_fallback(
            input=input,
            weight=weight,
            weight_scale_fallback=weight_scale_fallback,
            input_scale=input_scale,
            bias=bias,
        )

    dense_gemm, group_wise_quant = load_inferrock_gemm()
    m_align, synthesize_zero_bias = plan
    m_padded = ceil_div(m, m_align) * m_align

    # .contiguous() is required, not defensive: dense_gemm and group_wise_quant read
    # raw row-major pointers, so a strided view silently produces wrong numerics.
    input_2d = pad_rows(input.reshape(-1, k).contiguous(), m_padded)

    if prequantized:
        q_input = input_2d
        sfa = swizzle_scale_to_sfatom(
            pad_rows(input_scale.reshape(m, -1).contiguous(), m_padded)
        )
    else:
        q_input, sfa = group_wise_quant(input_2d, is_fp4=False, use_swizzle=True)

    if synthesize_zero_bias:
        # dense_gemm types the bias as the (always bf16) output element type.
        bias = torch.zeros(n, dtype=torch.bfloat16, device=input_2d.device)
    elif bias is not None:
        bias = bias.to(torch.bfloat16)

    # dense_gemm only writes bf16; cast back below for an fp16 layer.
    output = torch.empty((m_padded, n), dtype=torch.bfloat16, device=input_2d.device)
    dense_gemm(
        input=q_input,
        weight=weight,
        sfa=sfa,
        sfb=weight_scale,
        output=output,
        bias=bias,
        alpha=None,
        sm_count=-1,
        mode="w8a8",
    )
    return output[:m].to(output_dtype).reshape(*input.shape[:-1], n)

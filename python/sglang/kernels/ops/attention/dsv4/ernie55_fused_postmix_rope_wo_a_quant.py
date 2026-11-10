"""ERNIE5.5 HCA output-side fused kernel (open port of internal
``eb5_fused_postmix_rope_wo_a_quant``): inverse-RoPE + VHA cross-head postmix +
FP8 wo_a group-major quant in one launch, emitting int32-packed UE8M0 scales that
``deep_gemm.fp8_einsum`` consumes with no repack. Mirrors the internal kernel
bitwise; only the JIT source path and module name are open-tree local."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_GROUP_SIZE = 128


@cache_once
def _jit_module_fused_postmix_rope(in_dtype: torch.dtype, use_pdl: bool) -> "Module":
    args = make_cpp_args(in_dtype, use_pdl)
    return load_jit(
        "ernie55_fused_postmix_rope_wo_a_quant",
        *args,
        cuda_files=["deepseek_v4/ernie55_fused_postmix_rope_wo_a_quant.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"FP8WoAFusedPostmixRopeGroupMajorQuantUE8M0Kernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["--use_fast_math"],
    )


def ernie55_fused_postmix_rope_wo_a_quant(
    x: torch.Tensor,
    postmix_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    num_groups: int,
    out_heads: "int | None" = None,
    out_head_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse-RoPE -> cross-head postmix -> group-major FP8 wo_a quant.

    ``x``: [T, nh, head_dim] bf16/fp16 raw attention output (nh=64, head_dim=512;
    under attention TP the ALL-heads tensor gathered across the attn-tp group).
    ``postmix_weight``: [nh, nh] bf16 folded ``I + V @ U.T`` (vha_postmix_m).
    Returns ``(x_q [T,G,D] fp8_e4m3fn, x_s [T,G,P] int32 UE8M0)`` ready for
    ``fp8_einsum(recipe=(1,128,128))``. ``out_heads``/``out_head_offset`` are the
    write-mask hybrid used under attn TP (every rank runs the full [nh,nh] mix,
    writes only its out-head slice)."""
    num_tokens, nh, head_dim = x.shape
    if out_heads is None:
        out_heads = nh
    assert 0 < out_heads <= nh and 0 <= out_head_offset <= nh - out_heads, (
        f"out_heads/out_head_offset must select a sub-range of nh={nh}, "
        f"got out_heads={out_heads}, out_head_offset={out_head_offset}"
    )
    out_hidden = out_heads * head_dim
    assert out_hidden % num_groups == 0, "num_groups must divide out_heads*head_dim"
    hidden_per_group = out_hidden // num_groups
    scale_groups = hidden_per_group // _GROUP_SIZE
    packed = (scale_groups + 3) // 4
    aligned_tokens = (num_tokens + 3) // 4 * 4

    x_q = torch.empty(
        (num_tokens, out_hidden), device=x.device, dtype=torch.float8_e4m3fn
    )
    x_s_storage = torch.empty(
        (num_groups, packed, aligned_tokens), device=x.device, dtype=torch.int32
    )

    if x.numel() > 0:
        if freqs_cis.is_complex():
            freqs_cis = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)
        _jit_module_fused_postmix_rope(x.dtype, is_arch_support_pdl()).run(
            x,
            postmix_weight,
            x_q,
            x_s_storage,
            freqs_cis,
            positions,
            num_groups,
            out_head_offset,
            out_heads,
        )

    x_s = x_s_storage.transpose(1, 2)[:, :num_tokens, :].transpose(0, 1)
    return x_q.view(num_tokens, num_groups, hidden_per_group), x_s

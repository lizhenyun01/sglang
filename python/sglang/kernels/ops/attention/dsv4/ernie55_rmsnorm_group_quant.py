"""Open-tree port of the internal no-residual RMSNorm + group FP8 (UE8M0) quant
entries (internal ``srt/internal/layers/attention/add_rmsnorm_quant.py``).

The CUDA source is the verbatim internal ``add_rmsnorm_group_quant_ue8m0.cuh``,
already copied to ``deepseek_v4/ernie55_add_rmsnorm_group_quant_ue8m0.cuh``.

``group_size=32`` emits the HALOps ``group_wise_quant(round_bf16(n),
mode="w8a8-mxfp8", use_swizzle=False)`` layout: fp8 e4m3 ``[M, H]`` plus a
uint8 row-major UE8M0 ``[M, H/32]`` scale, i.e. the group-32 tuple the LL GEMM
reads. Other group sizes emit the column-major TMA-aligned int32-packed scale.
"""

from __future__ import annotations

from typing import Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

_CUDA_FILE = "deepseek_v4/ernie55_add_rmsnorm_group_quant_ue8m0.cuh"


@cache_once
def _jit_module(group_size: int, use_pdl: bool):
    args = make_cpp_args(group_size, use_pdl)
    return load_jit(
        "ernie55_rmsnorm_group_quant_ue8m0",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("run_norm", f"rmsnorm_group_quant_ue8m0<{args}>"),
            ("run_norm_keep", f"rmsnorm_group_quant_ue8m0_with_norm<{args}>"),
        ],
        extra_cuda_cflags=[],
    )


def _alloc_q_sf(m: int, hidden: int, group_size: int, device):
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=device)
    if group_size == 32:
        sf = torch.empty((m, hidden // 32), dtype=torch.uint8, device=device)
        return q, sf
    from sglang.kernels.ops.quantization.fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
    )

    sf = create_per_token_group_quant_fp8_output_scale(
        x_shape=(m, hidden),
        device=device,
        group_size=group_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return q, sf


def rmsnorm_group_quant_ue8m0(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``rms_norm(x) * weight`` -> ``(q, sf)``, one fused kernel."""
    m, hidden = x.shape
    q, sf = _alloc_q_sf(m, hidden, group_size, x.device)
    _jit_module(group_size, is_arch_support_pdl()).run_norm(
        x, weight, q, sf, float(eps)
    )
    return q, sf


def rmsnorm_group_quant_ue8m0_with_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``rmsnorm_group_quant_ue8m0`` that also returns the bf16 normed ``[M, H]``
    (``(normed, q, sf)``), for the DSA q-path where the indexer reads bf16 while
    q_b_proj eats ``(q, sf)``. ``x`` may be a strided row slice."""
    m, hidden = x.shape
    normed = torch.empty((m, hidden), dtype=torch.bfloat16, device=x.device)
    q, sf = _alloc_q_sf(m, hidden, group_size, x.device)
    _jit_module(group_size, is_arch_support_pdl()).run_norm_keep(
        x, weight, normed, q, sf, float(eps)
    )
    return normed, q, sf

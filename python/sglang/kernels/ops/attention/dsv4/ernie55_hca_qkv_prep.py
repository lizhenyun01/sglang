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


@cache_once
def _jit_module_hca_qkv_prep(page_size: int, use_pdl: bool) -> "Module":
    args = make_cpp_args(page_size, use_pdl)
    return load_jit(
        "ernie55_hca_qkv_prep",
        *args,
        cuda_files=["deepseek_v4/ernie55_hca_qkv_prep.cuh"],
        cuda_wrappers=[("run", f"Eb5HcaQkvPrepKernel<{args}>::run")],
    )


def ernie55_hca_qkv_prep(
    qkv_a: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    kv_weight: torch.Tensor,
    kv_eps: float,
    freqs_real: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
    q_lora_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ERNIE5.5 HCA decode prepare in one launch: the q-role is the same output as
    ``rmsnorm_group_quant_ue8m0(qkv_a[:, :q_lora_rank], ..., 32)`` (returned as
    ``q`` fp8 + ``sf`` uint8 UE8M0 group-32 scale), and the k-role is the
    ``fused_k_norm_rope_flashmla`` store of ``qkv_a[:, q_lora_rank:]`` into
    ``kvcache`` (in place). ``freqs_real`` is the fp32 ``[max_pos, 64]`` view of
    ``freqs_cis``. This mirrors the internal ``eb5_hca_qkv_prep`` kernel bitwise;
    only the JIT source path and module name are open-tree local."""
    m = qkv_a.shape[0]
    q = torch.empty((m, q_lora_rank), dtype=torch.float8_e4m3fn, device=qkv_a.device)
    sf = torch.empty((m, q_lora_rank // 32), dtype=torch.uint8, device=qkv_a.device)
    _jit_module_hca_qkv_prep(page_size, is_arch_support_pdl()).run(
        qkv_a,
        q_weight,
        q,
        sf,
        float(q_eps),
        kv_weight,
        freqs_real,
        positions,
        out_loc,
        kvcache,
        float(kv_eps),
        q_lora_rank,
    )
    return q, sf

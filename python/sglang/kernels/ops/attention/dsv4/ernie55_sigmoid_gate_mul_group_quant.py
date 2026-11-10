"""ERNIE5.5 gated-attention output + per-token group FP8 (UE8M0) quant (open port
of internal ``sigmoid_gate_mul_group_quant``): one kernel for the gated attention
-> wo_b path. ``sig = round_bf16(sigmoid(gate)); gated = round_bf16(o * sig)`` then
per-group FP8-e4m3 + UE8M0 scale, dropping the separate bf16 gated_o round trip.
With ``group_size=32`` it emits the HALOps ``group_wise_quant`` uint8 ``[M,
hidden/32]`` scale the group-32 LL kernel reads at decode. Mirrors the internal
kernel bitwise; only the JIT source path and module name are open-tree local."""

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
def _jit_module(group_size: int) -> "Module":
    args = make_cpp_args(group_size, is_arch_support_pdl())
    return load_jit(
        f"ernie55_sigmoid_gate_mul_group_quant_{group_size}_{is_arch_support_pdl()}",
        *args,
        cuda_files=["deepseek_v4/ernie55_sigmoid_gate_mul_group_quant.cuh"],
        cuda_wrappers=[("run", f"sigmoid_gate_mul_group_quant<{args}>")],
        extra_cuda_cflags=[],
    )


def ernie55_sigmoid_gate_mul_group_quant_ue8m0(
    o: torch.Tensor,
    gate: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``round_bf16(o * round_bf16(sigmoid(gate)))`` -> group fp8 + UE8M0 scale in
    one fused kernel. ``group_size=32`` returns the ``group_wise_quant`` uint8
    ``[M, hidden/32]`` scale wo_b's LL fp8 GEMM consumes; otherwise col-major
    TMA-aligned int32-packed UE8M0. Returns ``(q, sf)``."""
    o = o.contiguous()
    # The kernel reads the gate through its row stride, so the merged [q_up | gate]
    # q_b output's strided gate slice goes in without a copy.
    if not (
        gate.dim() == 2
        and gate.shape == o.shape
        and gate.stride(1) == 1
        and gate.stride(0) % 8 == 0
        and gate.data_ptr() % 16 == 0
    ):
        gate = gate.reshape(o.shape).contiguous()
    m, hidden = o.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=o.device)
    if group_size == 32:
        sf = torch.empty((m, hidden // 32), dtype=torch.uint8, device=o.device)
    else:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale,
        )

        sf = create_per_token_group_quant_fp8_output_scale(
            x_shape=(m, hidden),
            device=o.device,
            group_size=group_size,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
    _jit_module(group_size).run(o, gate, q, sf)
    return q, sf

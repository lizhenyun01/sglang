"""ERNIE 5.5 gated-attention output: ``x * sigmoid(gate)``, bit-exact with the
trainer (port of internal ``srt/internal/layers/attention/eb5_sigmoid_gate_mul.py``).

Double rounding (bf16 after the sigmoid, bf16 after the multiply) is the
contract: it matches ``x * torch.sigmoid(gate)`` on every bf16 pair, which the
single-rounding shared kernel does not. PDL, an ``EVEN_N`` no-mask fast path and
a two-tier block/warp choice on top. bf16 only: see ``covered``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

# (BLOCK_SIZE, num_warps) per tier.
_SMALL_CONFIG = (1024, 4)
_LARGE_CONFIG = (2048, 8)
_SMALL_MAX_ELEMENTS = 1 << 17

_PDL_OK: Optional[bool] = None


def _pdl_available() -> bool:
    """PDL (Programmatic Dependent Launch) requires SM90+ and non-interpret mode."""
    global _PDL_OK
    if _PDL_OK is None:
        if os.environ.get("TRITON_INTERPRET") == "1" or not torch.cuda.is_available():
            _PDL_OK = False
        else:
            major, _ = torch.cuda.get_device_capability()
            _PDL_OK = major >= 9 and hasattr(tl.extra.cuda, "gdc_wait")
    return _PDL_OK


def _launch_config(n_elements: int) -> tuple[int, int]:
    if n_elements <= _SMALL_MAX_ELEMENTS:
        return _SMALL_CONFIG
    return _LARGE_CONFIG


@triton.jit
def _ernie55_sigmoid_gate_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EVEN_N: tl.constexpr,
    USE_GDC: tl.constexpr,
    N_COLS: tl.constexpr,
    GATE_ROW_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Must precede the loads: gdc_wait makes the predecessor kernel's stores visible.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    # A non-zero GATE_ROW_STRIDE is the merged q-up GEMM case: ``gate`` is the
    # tail columns of a wider [T, q_out + gate_out] result.
    if GATE_ROW_STRIDE == 0:
        gate_offsets = offsets
    else:
        rows = offsets // N_COLS
        gate_offsets = rows * GATE_ROW_STRIDE + (offsets - rows * N_COLS)

    if EVEN_N:
        x = tl.load(x_ptr + offsets)
        g = tl.load(gate_ptr + gate_offsets)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        g = tl.load(gate_ptr + gate_offsets, mask=mask)

    dtype = x_ptr.dtype.element_ty
    # The double rounding is the contract: bf16 after sigmoid, bf16 after the multiply.
    s = tl.sigmoid(g.to(tl.float32)).to(dtype)
    out = (x.to(tl.float32) * s.to(tl.float32)).to(dtype)

    if EVEN_N:
        tl.store(out_ptr + offsets, out)
    else:
        tl.store(out_ptr + offsets, out, mask=mask)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def _gate_row_stride(x: torch.Tensor, gate: torch.Tensor) -> Optional[int]:
    """0 for a flat contiguous gate, its row stride for a row-strided 2-D one, else None."""
    if gate.is_contiguous():
        return 0
    if (
        x.dim() == 2
        and gate.dim() == 2
        and x.shape == gate.shape
        and x.is_contiguous()
        and gate.stride(1) == 1
        and gate.stride(0) >= gate.shape[1]
    ):
        return gate.stride(0)
    return None


def covered(x: torch.Tensor, gate: torch.Tensor) -> bool:
    """Whether this kernel can serve the pair, bit-exactness included (bf16 only;
    equal element count, a shorter gate would be broadcast by torch)."""
    return (
        x.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and x.is_cuda
        and gate.is_cuda
        and x.numel() == gate.numel()
        and x.numel() > 0
        and x.is_contiguous()
        and _gate_row_stride(x, gate) is not None
    )


def ernie55_sigmoid_gate_mul(
    x: torch.Tensor,
    gate: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    use_pdl: bool = True,
) -> torch.Tensor:
    """``x * sigmoid(gate)``, bit-identical to the unfused torch pair.

    Caller checks ``covered`` first. ``out`` may alias ``x``.
    """
    assert covered(x, gate), (x.dtype, x.shape, gate.dtype, gate.shape)
    if out is None:
        out = torch.empty_like(x)
    else:
        assert (
            out.dtype == x.dtype
            and out.numel() == x.numel()
            and out.is_contiguous()
            and out.is_cuda
        ), (out.dtype, out.shape)
    n_elements = x.numel()
    block_size, num_warps = _launch_config(n_elements)
    gdc = use_pdl and _pdl_available()
    gate_row_stride = _gate_row_stride(x, gate)
    _ernie55_sigmoid_gate_mul_kernel[(triton.cdiv(n_elements, block_size),)](
        x,
        gate,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        EVEN_N=n_elements % block_size == 0,
        USE_GDC=gdc,
        N_COLS=x.shape[-1],
        GATE_ROW_STRIDE=gate_row_stride,
        num_warps=num_warps,
        **({"launch_pdl": True} if gdc else {}),
    )
    return out


def ernie55_sigmoid_gate_mul_torch(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """The spelling the kernel reproduces, rounding at both steps."""
    return x * torch.sigmoid(gate)

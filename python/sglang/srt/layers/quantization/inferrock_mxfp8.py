"""InferRock MXFP8 (group-32) dense-GEMM weight helpers.

The load-time half of the InferRock dense GEMM: the optional-package loader and
the CUTLASS SfAtom 128x4 scale layout.
"""

from functools import lru_cache

import torch

# Every InferRock w8a8 tile needs K % 128 == 0, and the SfAtom scale layout
# needs its K/32 group count divisible by 4 -- the same 128.
INFERROCK_K_ALIGN = 128

# Row alignment of the CUTLASS SfAtom 128x4 scale layout. Also the loosest N
# any InferRock w8a8 tile accepts.
INFERROCK_SFATOM_ROW_ALIGN = 128


@lru_cache(maxsize=1)
def load_inferrock_gemm():
    """Return InferRock's ``(dense_gemm, group_wise_quant)``.

    Imported lazily so the wheel is only required once a layer actually needs
    it, and raises with an install hint rather than a bare ImportError so a
    ``--fp8-gemm-backend=inferrock`` run fails at load, not at first token.
    """
    try:
        from infer_rock import dense_gemm, group_wise_quant
    except ImportError as exc:
        raise RuntimeError(
            "InferRock GEMM backend requested, but the `infer_rock` package is not "
            "installed."
        ) from exc
    return dense_gemm, group_wise_quant


def weight_supported(*, n: int, k: int) -> bool:
    return k % INFERROCK_K_ALIGN == 0 and n % INFERROCK_SFATOM_ROW_ALIGN == 0


def pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """Zero-extend ``x`` to ``rows`` rows; returns ``x`` itself when already that tall."""
    if x.shape[0] == rows:
        return x
    padded = x.new_zeros((rows, *x.shape[1:]))
    padded[: x.shape[0]] = x
    return padded


def swizzle_scale_to_sfatom(scale_u8: torch.Tensor) -> torch.Tensor:
    """Row-major UE8M0 ``[M, K // 32]`` -> the CUTLASS SfAtom 128x4 layout.

    Byte-identical to ``group_wise_quant(use_swizzle=True)``, so a scale that is
    already quantized still reaches ``dense_gemm``. Requires ``M % 128 == 0``
    and ``K % 128 == 0``.
    """
    m, k_groups = scale_u8.shape
    return (
        scale_u8.contiguous()
        .view(m // 128, 128, k_groups // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(m // 128, k_groups // 4, 4, 32, 4)
        .permute(0, 1, 3, 2, 4)
        .reshape(m, k_groups)
        .contiguous()
    )

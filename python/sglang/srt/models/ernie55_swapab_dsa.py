"""ERNIE 5.5 native 16-head swapAB bf16 decode for the full-MLA (DSA) layers.

Port of internal ``srt/internal/layers/attention/dsv4/swapab_decode.py``
``maybe_swapab_dsa`` / ``maybe_swapab_dsa_early``. Drop-in for the
``flash_mla_sparse_fwd`` call in ``Ernie5_5MlaAttnBackend._forward_flashmla_sparse``:
returns a local ``[T, 16, 512]`` combine, or ``None`` to keep the stock path.
Needs the bf16 MLA latent sub-pool (``SGLANG_ERNIE55_MLA_SWAPAB``); the VHA postmix
does its own attn-TP head gather.

Internal-to-open mapping: ``SGLANG_INTERNAL_EB5_SINGLE_BATCH_LL`` ->
``envs.SGLANG_ERNIE55_SINGLE_BATCH_LL`` and ``envs.SGLANG_ERNIE55_MLA_SWAPAB``;
``is_sm100_supported`` -> ``current_arch().is_sm(10)``; the kernels come from the
ported ``ernie55_swapab_attention`` module.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs

_DSA_DK = 576
_DSA_DV = 512


def mla_swapab_enabled() -> bool:
    return bool(
        envs.SGLANG_ERNIE55_MLA_SWAPAB.get() and envs.SGLANG_ERNIE55_SINGLE_BATCH_LL.get()
    )


def _is_sm100() -> bool:
    from sglang.kernels.arch import current_arch

    return bool(current_arch().is_sm(10))


def _c(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


def maybe_swapab_dsa(
    *,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    sm_scale: float,
    v_head_dim: int,
    q_all: Optional[torch.Tensor] = None,
    q_nope: Optional[torch.Tensor] = None,
    q_rope: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Native 16-head swapAB DSA sparse decode, or ``None`` to fall back.

    ``kv_cache`` is the bf16 [num_slots(,1),576] latent cache, ``page_table_1``
    [T, nk] the selected slot ids (top-k + window rows) and ``topk_length`` [T]
    the per-row valid counts. q comes split (``q_nope`` [T,16,512] + ``q_rope``
    [T,16,64]) or concatenated (``q_all`` [T,16,576]). ``attn_sink`` is this
    rank's per-head sink (>= 16 entries, scaled-logit domain like FlashMLA's).
    """
    if not mla_swapab_enabled():
        return None
    split = q_nope is not None and q_rope is not None
    q_probe = q_nope if split else q_all
    if q_probe is None or q_probe.dim() != 3:
        return None
    num_tokens, num_heads = q_probe.shape[0], q_probe.shape[1]
    if num_heads != 16 or v_head_dim != _DSA_DV:
        return None
    if split:
        if q_nope.shape[-1] != _DSA_DV or q_rope.shape[-1] != _DSA_DK - _DSA_DV:
            return None
    elif q_all.shape[-1] != _DSA_DK:
        return None
    if not (0 < num_tokens <= 8):
        return None
    kv = kv_cache.reshape(-1, _DSA_DK)
    if kv.dtype != torch.bfloat16 or kv.stride(-1) != 1:
        return None
    if not _is_sm100():
        return None

    from sglang.kernels.ops.attention.dsv4.ernie55_swapab_attention import (
        eb5_swapab_combine_local,
        eb5_swapab_partial_bf16,
    )

    idx = page_table_1
    if idx.dtype != torch.int32:
        idx = idx.to(torch.int32)
    if idx.stride(-1) != 1:
        idx = idx.contiguous()
    nk = idx.shape[-1]
    if topk_length is not None and topk_length.shape[0] == num_tokens:
        lengths = topk_length.to(torch.int32)
    else:
        lengths = torch.full((num_tokens,), nk, dtype=torch.int32, device=q_probe.device)

    if split:
        part, mx, sm, nt = eb5_swapab_partial_bf16(
            _c(q_nope), kv, idx, lengths, sm_scale, q_rope=_c(q_rope)
        )
    else:
        part, mx, sm, nt = eb5_swapab_partial_bf16(_c(q_all), kv, idx, lengths, sm_scale)

    sink16 = None
    if attn_sink is not None:
        sink16 = attn_sink.reshape(-1)[:num_heads].to(torch.float32).contiguous()
    return eb5_swapab_combine_local(part, mx, sm, sink16, nt)


def maybe_swapab_dsa_early(
    *,
    early,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    sm_scale: float,
    v_head_dim: int,
    q_nope: Optional[torch.Tensor],
    q_rope: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """``maybe_swapab_dsa`` for a layer whose window rows, row lengths, top-k
    column counts and fp32 sink in ``early`` were built on a side lane by
    ``Ernie55LLDsaPrepare``: skips the append kernel and the sink cast, and reads
    the top-k columns straight from ``page_table_1``. Returns None, for the stock
    path, whenever ``maybe_swapab_dsa`` itself would differ."""
    if not mla_swapab_enabled() or q_nope is None or q_rope is None:
        return None
    if q_nope.dim() != 3 or q_rope.dim() != 3:
        return None
    num_tokens, num_heads = q_nope.shape[0], q_nope.shape[1]
    if num_heads != 16 or v_head_dim != _DSA_DV:
        return None
    if q_nope.shape[-1] != _DSA_DV or q_rope.shape[-1] != _DSA_DK - _DSA_DV:
        return None
    if num_tokens != early.rows or sm_scale != early.scale:
        return None
    kv = kv_cache.reshape(-1, _DSA_DK)
    if kv.dtype != torch.bfloat16 or kv.stride(-1) != 1:
        return None
    pt = page_table_1
    if not (
        pt.dtype == torch.int32
        and pt.dim() == 2
        and pt.stride(-1) == 1
        and pt.shape[0] >= num_tokens
        and pt.shape[1] == early.topk
    ):
        return None

    from sglang.kernels.ops.attention.dsv4.ernie55_swapab_attention import (
        eb5_swapab_combine_local,
        eb5_swapab_partial_bf16,
    )

    part, mx, sm, nt = eb5_swapab_partial_bf16(
        _c(q_nope),
        kv,
        early.rows_idx,
        early.lengths,
        sm_scale,
        q_rope=_c(q_rope),
        topk=pt,
        indexer_lengths=early.ilen,
    )
    return eb5_swapab_combine_local(part, mx, sm, early.sink16, nt)

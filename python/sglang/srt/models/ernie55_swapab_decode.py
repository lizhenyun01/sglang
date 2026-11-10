"""Open-tree port of the internal EB5 HCA swapAB decode dispatcher
(sglang/srt/internal/layers/attention/dsv4/swapab_decode.py).

``maybe_swapab_decode`` is a drop-in for the FlashMLA SWA decode call in
DeepseekV4AttnBackend: it returns a native 16-head swapAB attention output when
eligible, else ``None`` so the caller keeps the stock FlashMLA / open
swapab_attention path unchanged.

The 16-head kernel is native (no 16->64 pad). q arrives padded to 64 heads
(FlashMLA specializes h_q in {64,128}); we slice the first ``tp_q_head_num`` real
heads and the matching sink, squeeze the FlashMLA 4D cache / 3D index layout to
what ``eb5_swapab_partial`` expects, then partial + combine.

At attn_tp>1 the combine is fused with the attn-TP head all-gather
(``try_fast_combine_push_global``, one NVLink push) and returns the FULL
``[bs, num_heads, head_dim_v]`` in global head order, signalled via
``forward_batch._eb5_swapab_gathered = True`` so the caller skips the per-rank
head slice and the VHA postmix's own gather. When the push declines it falls back
to a local ``[bs, 16, head_dim_v]`` combine (flag False), identical to the stock
per-rank path.

Open-tree arming: the q-norm-rope fold (``_eb5_swapab_qnr`` / ``raw_q16``) is
armed by the HCA layer's Ernie55LLAttnBackend.arm_swapab under
SGLANG_ERNIE55_SWAPAB_QNR_FOLD, and the compressor-split side stream
(``_eb5_compressor_stream`` / ``late_sync``) by its prepare_hca_qkv under
SGLANG_ERNIE55_SWAPAB_LATE_C128. With both off this runs the plain
partial+combine (``tile_mode=0``, ``late_sync=None``). ``_eb5_swapab_qnr`` is
consumed (cleared) here so the layer's end_forward can tell an armed fold that
never reached this decode.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

_TOKEN_BYTES = 584
_LATE_SYNC = {}
_LATE_SYNC_TOKENS = 8  # _eligible caps the decode batch at 8 tokens


def _late_sync_buffer(device) -> torch.Tensor:
    """[2:4] u64 combine done count (seq = count >> 20), [8 + b] token b's late-tile
    flag. A flag never exceeds seq + 1, so zeroing (first use, possibly inside a
    capture) keeps every later wait consistent."""
    buf = _LATE_SYNC.get(device)
    if buf is None:
        buf = _LATE_SYNC[device] = torch.zeros(
            8 + _LATE_SYNC_TOKENS, dtype=torch.int32, device=device
        )
    return buf


def swapab_decode_eligible(
    forward_mode, tp_q_head_num: int, num_tokens: int, kv_itemsize: int, kv_token_bytes: int
) -> bool:
    """The one swapab-decode predicate, shared by maybe_swapab_decode and the HCA
    layer's arm_swapab (q-norm-rope fold), so the two can never disagree."""
    if not envs.SGLANG_ERNIE55_SWAPAB_DECODE.get():
        return False
    if not (forward_mode.is_decode() or forward_mode.is_target_verify()):
        return False
    if tp_q_head_num != 16:
        return False
    # The 584B DSA token is reinterpreted as uint8 inside eb5_swapab_partial
    # (``kv.view(torch.uint8)``), so any 1-byte element type is eligible: open's
    # HCA SWA pool is typed float8_e4m3fn, the internal one uint8 -- byte-identical
    # (448 fp8 nope + 64 bf16 rope + 8 UE8M0 scale/token).
    if kv_itemsize != 1 or kv_token_bytes != _TOKEN_BYTES:
        return False
    if num_tokens == 0 or num_tokens > 8:
        return False
    from sglang.kernels.arch import current_arch

    return bool(current_arch().is_sm(10))


def swa_kv_format(pool) -> Tuple[int, int]:
    """(element size, last dim) of the SWA cache view DeepseekV4AttnBackend.
    _forward_attention hands maybe_swapab_decode, read off the pool without
    get_swa_key_buffer_radix's layer-transfer wait."""
    window = getattr(pool, "request_window", None)
    dtype = window.state.dtype if window is not None else pool.swa_kv_pool.dtype
    itemsize = torch.empty((), dtype=dtype).element_size()
    return itemsize, int(pool.get_swa_key_bytes_per_token())


def _eligible(forward_mode, tp_q_head_num, swa_k_cache, q) -> bool:
    return swapab_decode_eligible(
        forward_mode,
        tp_q_head_num,
        int(q.shape[0]),
        swa_k_cache.element_size(),
        int(swa_k_cache.shape[-1]),
    )


def maybe_swapab_decode(
    *,
    q: torch.Tensor,
    forward_batch,
    tp_q_head_num: int,
    head_dim_v: int,
    softmax_scale: float,
    swa_k_cache: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    attn_sink: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_lengths: Optional[torch.Tensor] = None,
    raw_q16: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Native 16-head swapAB HCA decode, or ``None`` to fall back to FlashMLA."""
    forward_batch._eb5_swapab_gathered = False

    qnr = getattr(forward_batch, "_eb5_swapab_qnr", None)
    # Consumed here; the HCA layer's end_forward raises if it is still armed.
    forward_batch._eb5_swapab_qnr = None
    elig = _eligible(forward_batch.forward_mode, tp_q_head_num, swa_k_cache, q)
    assert qnr is None or elig, (
        "fused_q_norm_rope was skipped for swapab but maybe_swapab_decode declined; "
        "eligibility must match _eligible()"
    )
    if not elig:
        return None

    from sglang.kernels.ops.attention.dsv4.ernie55_swapab_attention import (
        eb5_swapab_combine_local,
        eb5_swapab_partial,
    )

    if qnr is not None:
        assert raw_q16 is not None, "q-norm-rope fold armed but raw 16-head q missing"
        q16 = raw_q16
    else:
        q16 = q[:, 0, :tp_q_head_num, :]
    sink16 = attn_sink[:tp_q_head_num]

    swa_kv = swa_k_cache.squeeze(2)
    swa_idx = swa_page_indices.squeeze(1)

    extra_kv = None if extra_k_cache is None else extra_k_cache.squeeze(2)
    extra_idx = (
        None
        if extra_indices_in_kvcache is None
        else extra_indices_in_kvcache.squeeze(1)
    )

    fold_freqs = qnr[0] if qnr is not None else None
    fold_pos = qnr[1] if qnr is not None else None
    fold_eps = qnr[2] if qnr is not None else 1e-6

    stream_compressor = getattr(forward_batch, "_eb5_compressor_stream", None)
    forward_batch._eb5_compressor_stream = None
    partial_args = dict(
        block=64,
        extra_kv=extra_kv,
        extra_indices=extra_idx,
        extra_lengths=extra_topk_lengths,
        freqs=fold_freqs,
        positions=fold_pos,
        q_eps=fold_eps,
    )
    current_stream = torch.cuda.current_stream()
    if stream_compressor is not None:
        assert extra_idx is not None and fold_pos is not None
        assert q.shape[0] % forward_batch.batch_size == 0
        partial_args["tokens_per_req"] = q.shape[0] // forward_batch.batch_size
        q_ready = current_stream.record_event()
    partial, maximum, sums, nt = eb5_swapab_partial(
        q16,
        swa_kv,
        swa_idx,
        swa_topk_lengths,
        softmax_scale,
        tile_mode=1 if stream_compressor is not None else 0,
        **partial_args,
    )
    forward_batch._eb5_partial_done_event = current_stream.record_event()
    gate_issuer = getattr(forward_batch, "_eb5_gate_issuer", None)
    if gate_issuer is not None:
        gate_issuer()

    bs = q.shape[0]
    from sglang.srt.distributed.parallel_state import get_tp_group

    world = get_tp_group().world_size
    late_sync = None
    if stream_compressor is not None:
        decouple = world > 1
        sync = _late_sync_buffer(q.device) if decouple else None
        stream_compressor.wait_event(q_ready)
        with torch.cuda.stream(stream_compressor):
            eb5_swapab_partial(
                q16,
                swa_kv,
                swa_idx,
                swa_topk_lengths,
                softmax_scale,
                tile_mode=2,
                out=(partial, maximum, sums),
                late_sync=sync,
                **partial_args,
            )
        if decouple:
            for t in (q16, partial, maximum, sums):
                t.record_stream(stream_compressor)
            assert fold_pos.dtype == torch.int64
            assert extra_topk_lengths.dtype == torch.int32
            late_sync = (
                sync,
                extra_topk_lengths,
                fold_pos,
                partial_args["tokens_per_req"],
                128,
            )
            forward_batch._eb5_compressor_join_late = stream_compressor
        else:
            current_stream.wait_stream(stream_compressor)
    if world > 1 and envs.SGLANG_ERNIE55_SWAPAB_PUSH.get():
        num_heads = tp_q_head_num * world
        out_global = torch.empty(
            (bs, num_heads, head_dim_v), dtype=torch.bfloat16, device=q.device
        )
        from sglang.srt.models.ernie55_fast_allgather import (
            try_fast_combine_push_global,
        )

        if try_fast_combine_push_global(
            partial,
            maximum,
            sums,
            sink16,
            out_global,
            tp_q_head_num,
            head_dim_v,
            num_heads,
            nt,
            late_sync=late_sync,
        ):
            forward_batch._eb5_swapab_gathered = True
            return out_global
    if late_sync is not None:
        raise RuntimeError(
            "HCA combine decoupling armed the late-tile flag but the fused combine "
            "push declined; the flag sequence would desync"
        )

    out = eb5_swapab_combine_local(partial, maximum, sums, sink16, nt)
    assert out.shape[-1] == head_dim_v
    return out

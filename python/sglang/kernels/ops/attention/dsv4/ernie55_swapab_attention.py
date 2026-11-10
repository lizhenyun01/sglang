"""EB5 SWA-decode dispatcher over the swapAB partial + combine kernels.

Drop-in for the ``flash_mla_with_kvcache`` call in ``EB5AttnBackend._swa_decode_flashmla``:
same q / cache / indices / sink / scale contract, same [bs, 16, 512] output. The
partial kernel (gluon, SM100) writes FP32 per-tile partials; ``_combine`` merges them
and folds the per-head attention sink exactly once.

PDL (both boundaries, non-conservative):
  * partial_gluon: griddepcontrol.wait after the c128 KV load and the q-independent
    setup (RoPE tables, smem/tmem, mbarrier), so they overlap the q-prep producer, but
    before the q and SWA-window loads (tile_mode 2 loads all its KV after the wait);
    launch_dependents at entry (tile_mode 0/1) or right after the PART/MAX/SUM stores.
    gluon has no gdc_* wrapper, so both are emitted as inline PTX
    (griddepcontrol.{wait,launch_dependents}).
  * _combine: gdc_wait after the producer-independent sink load (overlaps partial's
    epilogue) but before the PART/MAX/SUM reads; gdc_launch_dependents at exit for its
    own successor. Requires launch_pdl on both launches (set when _pdl_supported()).
"""

import torch
import triton
import triton.language as tl

_D = 512
_H = 16


def _pdl_supported() -> bool:
    try:
        from sglang.kernels.jit.utils import is_arch_support_pdl

        return bool(is_arch_support_pdl())
    except Exception:
        return False


@triton.jit
def _combine(
    PART,
    MAX,
    SUM,
    SINK,
    OUT,
    NT: tl.constexpr,
    ST: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
    HAS_SINK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t, d = tl.arange(0, ST), tile * BD + tl.arange(0, BD)
    if HAS_SINK:
        sink = tl.load(SINK + h)
    else:
        sink = -float("inf")
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    den = tl.load(SUM + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.load(MAX + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.where(den > 0, mx, -float("inf"))
    m = tl.maximum(tl.max(mx, 0), sink)
    m = tl.where(tl.abs(m) == float("inf"), 0.0, m)
    factor = tl.exp(mx - m)
    denominator = tl.sum(den * factor, 0) + tl.exp(sink - m)
    vals = tl.load(
        PART + ((b * NT + t[:, None]) * H + h) * 512 + d[None, :],
        t[:, None] < NT,
        0,
    )
    out = tl.sum(vals * factor[:, None], 0) / denominator
    out = tl.where((denominator > 0) & (sink != float("inf")), out, 0.0)
    tl.store(OUT + (b * H + h) * 512 + d, out)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def eb5_swapab_partial(
    q,
    kv,
    indices,
    lengths,
    scale,
    block: int = 64,
    extra_kv=None,
    extra_indices=None,
    extra_lengths=None,
    freqs=None,
    positions=None,
    q_eps: float = 1e-6,
    tile_mode: int = 0,
    tokens_per_req: int = 1,
    out=None,
    late_sync=None,
):
    """Run the swapAB partial (attention MMAs, split-KV) -> per-tile partials.

    Mirrors the reference ``swapab_attention`` input handling: tiles [0, KT) cover
    the SWA window (``kv``/``indices``/``lengths``), tiles [KT, NT) cover the c128
    compressed history (``extra_*``). ``kv``/``extra_kv`` are the 584B DSA caches
    ``[num_pages, page_size, 584]`` (uint8-viewable; 448 fp8 nope + 64 bf16 rope +
    8 UE8M0 scale/token). ``q`` is the 512-wide [q_nope(448); q_rope(64)] per head.
    When ``extra_kv`` is None the reference aliases it to the main pool with NE=0.

    ``tile_mode`` splits one partial into two launches over the same outputs: 1 skips
    each token's last c128 tile when this forward's compressor may have written it,
    2 (grid ``(bs, 1)``, into ``out`` from the mode-1 call) computes exactly those
    tiles, so mode 1 can start before the compressor finishes. Needs ``positions``
    and ``tokens_per_req`` (tokens are request-major, that many per request).
    ``late_sync`` (mode 2) is the combine's int32 sync buffer: each late token
    releases ``late_sync[8 + b]`` after its stores (see ``_late_sync_buffer``).

    Returns (partial [b,nt,16,512] f32, maximum [b,nt,16] f32, sums [b,nt,16] f32, nt).
    """
    from sglang.kernels.ops.attention.dsv4.ernie55_swapab_gluon_partial import (
        partial_gluon,
    )

    q = q.reshape(q.shape[0], _H, q.shape[-1])
    bs, _, qk = q.shape
    assert qk == 512 and q.dtype == torch.bfloat16 and q.stride(-1) == 1
    assert 0 < bs <= 8
    nk = indices.shape[-1]
    ne = 0 if extra_indices is None else extra_indices.shape[-1]
    assert nk > 0 and indices.stride(-1) == 1 and lengths.is_contiguous()
    if extra_kv is None:
        extra_kv, extra_indices, extra_lengths = kv, indices, lengths
    kv = kv.view(torch.uint8)
    extra_kv = extra_kv.view(torch.uint8)
    assert kv.shape[-1] == 584 and extra_kv.shape[-1] == 584
    kt = triton.cdiv(nk, block)
    nt = kt + triton.cdiv(ne, block)

    assert out is not None or tile_mode != 2
    assert late_sync is None or tile_mode == 2
    if out is not None:
        partial, maximum, sums = out
    else:
        partial = torch.empty((bs, nt, _H, _D), dtype=torch.float32, device=q.device)
        maximum = torch.empty((bs, nt, _H), dtype=torch.float32, device=q.device)
        sums = torch.empty_like(maximum)

    fuse_qnr = freqs is not None and positions is not None
    if fuse_qnr:
        assert freqs.dtype == torch.float32 and freqs.is_contiguous()
        assert freqs.shape[-1] == 64 and positions.numel() >= bs
    if tile_mode:
        assert ne > 0 and positions is not None and positions.numel() >= bs
    freqs_arg = freqs if fuse_qnr else q
    pos_arg = positions if fuse_qnr or tile_mode else lengths
    use_pdl = _pdl_supported()

    partial_gluon[(bs, 1) if tile_mode == 2 else (bs, nt)](
        q,
        kv,
        extra_kv,
        indices,
        extra_indices,
        lengths,
        extra_lengths,
        partial,
        maximum,
        sums,
        freqs_arg,
        pos_arg,
        late_sync if late_sync is not None else lengths,
        QS=q.stride(0),
        QH=q.stride(-2),
        IS=indices.stride(0),
        EIS=extra_indices.stride(0),
        KP=kv.shape[1],
        KS=kv.stride(0),
        EP=extra_kv.shape[1],
        ES=extra_kv.stride(0),
        NK=nk,
        NE=ne,
        NT=nt,
        KT=kt,
        BT=block,
        H=_H,
        SCALE=scale,
        KTOKENS=kv.shape[0] * kv.shape[1],
        ETOKENS=extra_kv.shape[0] * extra_kv.shape[1],
        COMPENSATE=True,
        SWAP_AB=True,
        FUSE_QNR=fuse_qnr,
        QEPS=q_eps,
        ROPE=64,
        USE_PDL=use_pdl,
        TILE_MODE=tile_mode,
        TOKENS_PER_REQ=tokens_per_req,
        DECOUPLE=late_sync is not None,
        num_warps=4,
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return partial, maximum, sums, nt


def eb5_swapab_combine_local(partial, maximum, sums, sink, nt):
    """Triton online-softmax merge of pre-computed partials -> local [bs,16,512].

    The non-fused path (attn_tp==1, or the fused combine+push not eligible); the
    caller's postmix then does its own head gather if needed."""
    bs = partial.shape[0]
    out = torch.empty((bs, _H, _D), dtype=torch.bfloat16, device=partial.device)
    use_pdl = _pdl_supported()
    extra = {"launch_pdl": True} if use_pdl else {}
    _combine[(bs, _H, 1)](
        partial,
        maximum,
        sums,
        sink,
        out,
        NT=nt,
        ST=triton.next_power_of_2(nt),
        H=_H,
        BD=_D,
        HAS_SINK=sink is not None,
        USE_PDL=use_pdl,
        **extra,
    )
    return out


_DK_BF16 = 576


def eb5_swapab_partial_bf16(
    q,
    kv,
    indices,
    lengths,
    scale,
    block: int = 64,
    q_rope=None,
    topk=None,
    indexer_lengths=None,
):
    """swapAB partial for the DSA full-MLA sparse path (bf16 KV, 576-wide QK).

    Unlike ``eb5_swapab_partial`` (HCA fp8-584B, 512-wide), this consumes the
    gathered **bf16** KV ``dsa_backend._forward_flashmla_sparse`` builds:
    ``kv`` [num_tokens,576] bf16 (576 = latent 512 + rope 64), ``indices``
    [bs,topk] int32 slot ids, ``lengths`` [bs] valid counts. Score runs over the
    full 576; the value/output is the 512 latent only, matching
    ``flash_mla_sparse_fwd(d_v=512)``.

    ``q``/``q_rope`` two forms (both avoid a caller-side concat when split):
      * q_rope is None: ``q`` is the concatenated [bs(,1),16,576].
      * q_rope given: ``q`` is q_nope [bs(,1),16,512], ``q_rope`` [bs(,1),16,64]
        -- loaded from separate tensors (no concat_mla_absorb_q).

    ``topk`` given (DSA early window rows): ``indices`` is the
    ``swa_window_rows_triton`` buffer (window columns only), ``indexer_lengths``
    its per-row top-k column count, and the top-k columns are read straight from
    ``topk`` (the raw top-k output the append kernel would copy).

    Returns (partial [bs,nt,16,512] f32, maximum [bs,nt,16] f32, sums [..], nt).
    """
    from sglang.kernels.ops.attention.dsv4.ernie55_swapab_gluon_partial_bf16 import (
        partial_gluon_bf16,
    )

    if q_rope is None:
        q = q.reshape(q.shape[0], _H, q.shape[-1])
        assert q.shape[-1] == _DK_BF16
        qr, qroff = q, _D
    else:
        q = q.reshape(q.shape[0], _H, q.shape[-1])
        qr = q_rope.reshape(q_rope.shape[0], _H, q_rope.shape[-1])
        assert q.shape[-1] == _D and qr.shape[-1] == _DK_BF16 - _D
        assert qr.dtype == torch.bfloat16 and qr.stride(-1) == 1
        qroff = 0
    bs = q.shape[0]
    assert q.dtype == torch.bfloat16 and q.stride(-1) == 1
    assert 0 < bs <= 8
    kv = kv.reshape(-1, _DK_BF16)
    assert kv.dtype == torch.bfloat16 and kv.stride(-1) == 1
    nk = indices.shape[-1]
    assert nk > 0 and indices.stride(-1) == 1 and lengths.is_contiguous()
    nt = triton.cdiv(nk, block)
    direct = topk is not None
    if direct:
        assert indexer_lengths is not None and indexer_lengths.dtype == torch.int32
        assert indexer_lengths.is_contiguous() and indexer_lengths.numel() >= bs
        assert topk.dtype == torch.int32 and topk.dim() == 2
        assert topk.stride(-1) == 1 and topk.shape[0] >= bs

    partial = torch.empty((bs, nt, _H, _D), dtype=torch.float32, device=q.device)
    maximum = torch.empty((bs, nt, _H), dtype=torch.float32, device=q.device)
    sums = torch.empty_like(maximum)
    use_pdl = _pdl_supported()

    partial_gluon_bf16[(bs, nt)](
        q,
        qr,
        kv,
        indices,
        lengths,
        partial,
        maximum,
        sums,
        topk if direct else indices,
        indexer_lengths if direct else lengths,
        QS=q.stride(0),
        QH=q.stride(-2),
        QRS=qr.stride(0),
        QRH=qr.stride(-2),
        QROFF=qroff,
        IS=indices.stride(0),
        KS=kv.stride(0),
        NK=nk,
        NT=nt,
        BT=block,
        H=_H,
        DL=_D,
        DR=_DK_BF16 - _D,
        SCALE=scale,
        KTOKENS=kv.shape[0],
        COMPENSATE=True,
        USE_PDL=use_pdl,
        DIRECT_TOPK=direct,
        TS=topk.stride(0) if direct else 0,
        TOPKW=topk.shape[-1] if direct else 1,
        num_warps=4,
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return partial, maximum, sums, nt

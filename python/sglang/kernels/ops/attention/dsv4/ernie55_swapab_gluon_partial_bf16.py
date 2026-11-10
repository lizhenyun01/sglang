"""Native 16-head Blackwell (SM100) MMA for EB5 DSA full-MLA sparse decode.

Sibling of ``gluon_partial.partial_gluon`` (the HCA c128 fp8-584B kernel). The
DSA full-MLA layers use a DIFFERENT layout, verified from the runtime:

  * KV is **bf16**, gathered ``[num_tokens, 576]`` (``dsa_backend`` builds it via
    ``_cat([k, k_rope])`` / ``dequantize_k_cache_paged``), NOT the paged fp8 584B
    cache. 576 = kv_lora_rank(512) latent + qk_rope_head_dim(64).
  * score contraction D_QK = 512 + 64 = **576** (q also 576-wide).
  * value / output width d_v = **512**: absorbed MLA attends the score over the
    full 576 but the value is the 512 latent only (``k[:, :512]``); rope does NOT
    enter the output. This matches ``flash_mla_sparse_fwd(d_v=512)``.

So vs the HCA kernel: no fp8 dequant / UE8M0 scales, no page split, single KV
region (no c128 extra), score = latent-MMA + rope-MMA, PV over latent only.
q is already norm+rope'd by the model, so no fused_q_norm_rope here.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
)

try:
    # Triton 3.7.1 API
    from triton.experimental.gluon.language.nvidia.blackwell import get_tmem_reg_layout
except ImportError:
    # Triton 3.8 moved this onto the descriptor type (triton-lang/triton#9594).
    from triton.experimental.gluon.language.nvidia.blackwell import (
        tensor_memory_descriptor_type,
    )
    from triton.runtime.jit import constexpr_function

    @constexpr_function
    def get_tmem_reg_layout(element_ty, shape, layout, num_warps):
        ty = tensor_memory_descriptor_type(element_ty, shape, layout, shape)
        return ty.get_reg_layout(num_warps=num_warps)


@gluon.jit
def _load_bf16(
    CACHE,
    ids,
    valid,
    STRIDE: gl.constexpr,
    COFF: gl.constexpr,
    DCOL: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    """Gather ``[BT, DCOL]`` bf16 rows (columns ``[COFF, COFF+DCOL)``) by ``ids``."""
    d = gl.arange(0, DCOL, gl.SliceLayout(0, LAYOUT))
    ptr = CACHE + ids.to(gl.int64)[:, None] * STRIDE + (COFF + d[None, :])
    return gl.load(ptr, valid[:, None], 0)


@gluon.jit
def _tile_ids(
    IDX,
    L,
    TOPK,
    ILEN,
    b,
    t,
    IS: gl.constexpr,
    NK: gl.constexpr,
    BT: gl.constexpr,
    TS: gl.constexpr,
    TOPKW: gl.constexpr,
    DIRECT_TOPK: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    """Slot ids of tile ``t`` of row ``b`` and the row length.

    Default: ``IDX`` is the packed ``top-k | last-window`` row the append kernel
    built. DIRECT_TOPK: ``IDX`` holds only the window columns (from ``ILEN[b]``)
    and the ``c < ILEN[b]`` columns are read from the raw top-k output ``TOPK``,
    the values the append kernel would copy there; the unwritten head of the
    window-only row is never read."""
    n = gl.arange(0, BT, gl.SliceLayout(1, LAYOUT))
    at = t * BT + n
    length = gl.load(L + b)
    if DIRECT_TOPK:
        live = (at < NK) & (at < length)
        ilen = gl.load(ILEN + b)
        from_topk = at < ilen
        tk = gl.load(TOPK + b * TS + at, live & from_topk & (at < TOPKW), -1)
        wn = gl.load(IDX + b * IS + at, live & (at >= ilen), -1)
        ids = gl.where(from_topk, tk, wn)
    else:
        ids = gl.load(IDX + b * IS + at, at < NK, -1)
    return at, ids, length


@gluon.jit
def partial_gluon_bf16(
    Q,
    QR,
    K,
    IDX,
    L,
    PART,
    MAX,
    SUM,
    TOPK,
    ILEN,
    QS: gl.constexpr,
    QH: gl.constexpr,
    QRS: gl.constexpr,
    QRH: gl.constexpr,
    QROFF: gl.constexpr,
    IS: gl.constexpr,
    KS: gl.constexpr,
    NK: gl.constexpr,
    NT: gl.constexpr,
    BT: gl.constexpr,
    H: gl.constexpr,
    DL: gl.constexpr,
    DR: gl.constexpr,
    SCALE: gl.constexpr,
    KTOKENS: gl.constexpr,
    COMPENSATE: gl.constexpr,
    USE_PDL: gl.constexpr,
    DIRECT_TOPK: gl.constexpr,
    TS: gl.constexpr,
    TOPKW: gl.constexpr,
):
    gl.static_assert(H == 16 and (BT == 64 or BT == 128))
    b, t = gl.program_id(0), gl.program_id(1)
    # DIRECT_TOPK reads nothing before the PDL wait: its first reads (top-k,
    # window rows) are what its producers write.
    if USE_PDL and DIRECT_TOPK:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    kv_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    at, ids, length = _tile_ids(
        IDX, L, TOPK, ILEN, b, t, IS, NK, BT, TS, TOPKW, DIRECT_TOPK, kv_layout
    )
    valid = (at < NK) & (at < length) & (ids >= 0) & (ids < KTOKENS)
    ids0 = gl.maximum(ids, 0)
    kv_lat = _load_bf16(K, ids0, valid, KS, 0, DL, kv_layout)
    kv_rope = _load_bf16(K, ids0, valid, KS, DL, DR, kv_layout)

    if USE_PDL and not DIRECT_TOPK:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )

    qh = gl.arange(0, H, gl.SliceLayout(1, kv_layout))
    ql = gl.arange(0, DL, gl.SliceLayout(0, kv_layout))
    qr = gl.arange(0, DR, gl.SliceLayout(0, kv_layout))
    q_lat = gl.load(Q + b * QS + qh[:, None] * QH + ql[None, :])
    q_rope = gl.load(QR + b * QRS + qh[:, None] * QRH + (QROFF + qr[None, :]))

    nvmma: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=16
    )
    ql_smem = gl.allocate_shared_memory(gl.bfloat16, [H, DL], nvmma, value=q_lat)
    qr_smem = gl.allocate_shared_memory(gl.bfloat16, [H, DR], nvmma, value=q_rope)
    kvl_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, DL], nvmma, value=kv_lat)
    kvr_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, DR], nvmma, value=kv_rope)
    score_tmem = allocate_tensor_memory(
        gl.float32, [BT, H], TensorMemoryLayout(block=(BT, H), col_stride=1)
    )
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    fence_async_shared()
    tcgen05_mma(kvl_smem, ql_smem.permute((1, 0)), score_tmem, use_acc=False)
    tcgen05_mma(kvr_smem, qr_smem.permute((1, 0)), score_tmem, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    score_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BT, H), TensorMemoryLayout(block=(BT, H), col_stride=1), 4
    )
    scores = score_tmem.load(score_layout) * SCALE
    valid = gl.convert_layout(valid, gl.SliceLayout(1, score_layout))
    scores = gl.where(valid[:, None], scores, -float("inf"))
    mx = gl.max(scores, 0)
    mx = gl.where(mx == -float("inf"), 0.0, mx)
    prob = gl.exp(scores - mx[None, :])
    denom = gl.sum(prob, 0)
    p_hi = prob.to(gl.bfloat16)
    p_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [BT, H],
        gl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16),
        value=p_hi,
    )
    out_tmem = allocate_tensor_memory(
        gl.float32, [DL, H], TensorMemoryLayout(block=(128, H), col_stride=1)
    )
    fence_async_shared()
    tcgen05_mma(kvl_smem.permute((1, 0)), p_smem, out_tmem, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=1)
    if COMPENSATE:
        p_lo = (prob - p_hi.to(gl.float32)).to(gl.bfloat16)
        p_smem.store(p_lo)
        fence_async_shared()
        tcgen05_mma(kvl_smem.permute((1, 0)), p_smem, out_tmem, use_acc=True)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    out_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (DL, H), TensorMemoryLayout(block=(128, H), col_stride=1), 4
    )
    value = out_tmem.load(out_layout)
    hd_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    value_hd = gl.convert_layout(value.permute((1, 0)), hd_layout)
    h = gl.arange(0, H, gl.SliceLayout(1, hd_layout))
    d = gl.arange(0, DL, gl.SliceLayout(0, hd_layout))
    gl.store(PART + ((b * NT + t) * H + h[:, None]) * DL + d[None, :], value_hd)
    stat_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    hs = gl.arange(0, H, stat_layout)
    gl.store(MAX + (b * NT + t) * H + hs, gl.convert_layout(mx, stat_layout))
    gl.store(SUM + (b * NT + t) * H + hs, gl.convert_layout(denom, stat_layout))
    if USE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )

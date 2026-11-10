"""Native 16-head Blackwell (SM100) MMA for EB5's DSA paged FP8 KV cache.

Adapted from the swapAB reference (sgl-project/sglang#39674). The EB5 FP8 KV
layout is produced by ``kernels/ops/attention/dsv4/quant_k_cache.py`` +
``index_buf_accessor.SetKAndS`` and is **584 bytes/token**, laid out per page as:

  * data region, ``page_size`` tokens x 576 bytes each:
      - bytes [0, 448)   : 448-ch fp8 nope latent (qk_nope_head_dim)
      - bytes [448, 576) : 64-ch bf16 rope (qk_rope_head_dim, 128 bytes)
  * scale region, after all ``page_size`` data blocks (offset ``page_size*576``):
      - ``page_size`` x 8 bytes: 7 UE8M0 uint8 exponents (one per 64-ch tile) + 1 pad

So the score contraction D_QK = 448 + 64 = **512** (a power of 2 -> one MMA), and
value/output = the same 512-wide [nope; rope] vector (v_head_dim = 512). This is
NOT 512-nope+64-rope=576: the fp8 nope is 448, and rope shares the 512 vector.
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
def _load_eb5(
    CACHE, ids, valid, PAGE: gl.constexpr, STRIDE: gl.constexpr, KV_LAYOUT: gl.constexpr
):
    """One BT-token tile of EB5's 584B DSA cache -> combined [BT, 512] bf16.

    [0,448) = fp8 nope * UE8M0 tile scale; [448,512) = bf16 rope. Byte addressing
    matches SetKAndS: nope/rope at ``slot*576``, scale at ``PAGE*576 + slot*8``.
    """
    d = gl.arange(0, 512, gl.SliceLayout(0, KV_LAYOUT))
    base = (ids // PAGE).to(gl.int64)[:, None] * STRIDE
    slot = (ids % PAGE)[:, None]
    mask = valid[:, None] & (d[None, :] < 448)
    bits = gl.load(CACHE + base + slot * 576 + d[None, :], mask, 0)
    fp8 = bits.to(gl.float8e4nv, bitcast=True).to(gl.float32)
    exponent = gl.load(
        CACHE + base + PAGE * 576 + slot * 8 + d[None, :] // 64, mask, 0
    ).to(gl.int32)
    scale = gl.where(exponent == 0, 0x00400000, exponent << 23).to(
        gl.float32, bitcast=True
    )
    rope_ptr = (CACHE + base + slot * 576 + 448 + (d[None, :] - 448) * 2).to(
        gl.pointer_type(gl.bfloat16)
    )
    rope = gl.load(rope_ptr, valid[:, None] & (d[None, :] >= 448), 0)
    return gl.where(d[None, :] < 448, fp8 * scale, rope.to(gl.float32)).to(gl.bfloat16)


@gluon.jit
def _load_tile(
    K,
    E,
    IDX,
    EI,
    L,
    EL,
    b,
    t,
    IS: gl.constexpr,
    EIS: gl.constexpr,
    KP: gl.constexpr,
    KS: gl.constexpr,
    EP: gl.constexpr,
    ES: gl.constexpr,
    NK: gl.constexpr,
    NE: gl.constexpr,
    KT: gl.constexpr,
    BT: gl.constexpr,
    KTOKENS: gl.constexpr,
    ETOKENS: gl.constexpr,
    KV_LAYOUT: gl.constexpr,
    SKIP_SWA: gl.constexpr = False,
):
    """Tile ``t`` of token ``b``: SWA window for t < KT, else c128 history.

    SKIP_SWA leaves an SWA tile unread (zeros, all invalid); the caller loads it later.
    """
    n = gl.arange(0, BT, gl.SliceLayout(1, KV_LAYOUT))
    if t < KT:
        if SKIP_SWA:
            valid = n < 0
            kv = gl.zeros([BT, 512], gl.bfloat16, KV_LAYOUT)
        else:
            at = t * BT + n
            length = gl.load(L + b)
            ids = gl.load(IDX + b * IS + at, at < NK, -1)
            valid = (at < NK) & (at < length) & (ids >= 0) & (ids < KTOKENS)
            kv = _load_eb5(K, gl.maximum(ids, 0), valid, KP, KS, KV_LAYOUT)
    else:
        at = (t - KT) * BT + n
        length = gl.load(EL + b)
        ids = gl.load(EI + b * EIS + at, at < NE, -1)
        valid = (at < NE) & (at < length) & (ids >= 0) & (ids < ETOKENS)
        kv = _load_eb5(E, gl.maximum(ids, 0), valid, EP, ES, KV_LAYOUT)
    return kv, valid


@gluon.jit
def _is_late(EL, POS, b, C_RATIO: gl.constexpr, TOKENS_PER_REQ: gl.constexpr):
    """Token b's last c128 entry is written by this forward's compressor iff its
    block closed at or after the request's first position in this forward
    (tokens are request-major, TOKENS_PER_REQ each)."""
    el = gl.load(EL + b)
    return gl.load(POS + b).to(gl.int32) + 1 - el * C_RATIO <= b % TOKENS_PER_REQ


@gluon.jit
def _partial_tile(
    Q,
    K,
    E,
    IDX,
    EI,
    L,
    EL,
    PART,
    MAX,
    SUM,
    FREQS,
    POS,
    b,
    t,
    QS: gl.constexpr,
    QH: gl.constexpr,
    IS: gl.constexpr,
    EIS: gl.constexpr,
    KP: gl.constexpr,
    KS: gl.constexpr,
    EP: gl.constexpr,
    ES: gl.constexpr,
    NK: gl.constexpr,
    NE: gl.constexpr,
    NT: gl.constexpr,
    KT: gl.constexpr,
    BT: gl.constexpr,
    H: gl.constexpr,
    SCALE: gl.constexpr,
    KTOKENS: gl.constexpr,
    ETOKENS: gl.constexpr,
    COMPENSATE: gl.constexpr,
    FUSE_QNR: gl.constexpr,
    QEPS: gl.constexpr,
    ROPE: gl.constexpr,
    USE_PDL: gl.constexpr,
    TILE_MODE: gl.constexpr,
    C_RATIO: gl.constexpr,
    TOKENS_PER_REQ: gl.constexpr,
):
    """One (token b, tile t) work item; mode 1 skips, mode 2 remaps t to the late tile."""
    if TILE_MODE != 0:
        late = _is_late(EL, POS, b, C_RATIO, TOKENS_PER_REQ)
        last = KT + (gl.maximum(gl.load(EL + b), 1) - 1) // BT
        if TILE_MODE == 1:
            if late and t == last:
                return
        else:
            if not late:
                return
            t = last
    kv_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    # Mode 2 reads the c128 entry the producer (cmp_norm_rope) just wrote, so only
    # the other modes load KV ahead of the PDL wait. Under PDL the SWA window waits
    # too: it holds this forward's entry from hca_qkv_prep, which triggers its
    # dependents at entry and may still be running.
    swa_after_wait: gl.constexpr = USE_PDL and TILE_MODE != 2
    if TILE_MODE != 2:
        kv, valid = _load_tile(
            K, E, IDX, EI, L, EL, b, t, IS, EIS, KP, KS, EP, ES, NK, NE, KT, BT,
            KTOKENS, ETOKENS, kv_layout, swa_after_wait,
        )  # fmt: skip

    # Everything that does not read the producer's q goes ahead of the PDL wait.
    qh = gl.arange(0, H, gl.SliceLayout(1, kv_layout))
    qd = gl.arange(0, 512, gl.SliceLayout(0, kv_layout))
    nvmma: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=16
    )
    p_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=32, element_bitwidth=16
    )
    if FUSE_QNR:
        pos = gl.load(POS + b).to(gl.int64)
        j = qd - (512 - ROPE)
        is_rope = j >= 0
        pair = j // 2
        cos_off = pos * ROPE + (2 * pair).to(gl.int64)
        cosf = gl.load(FREQS + cos_off, is_rope, 1.0)
        sinf = gl.load(FREQS + cos_off + 1, is_rope, 0.0)
        sign = gl.where(is_rope & ((j & 1) == 0), -1.0, 1.0)
    if TILE_MODE == 2:
        kv_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, 512], nvmma)
    else:
        kv_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, 512], nvmma, value=kv)
    score_tmem = allocate_tensor_memory(
        gl.float32, [BT, H], TensorMemoryLayout(block=(BT, H), col_stride=1)
    )
    out_tmem = allocate_tensor_memory(
        gl.float32, [512, H], TensorMemoryLayout(block=(128, H), col_stride=1)
    )
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    if USE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    if TILE_MODE == 2:
        kv, valid = _load_tile(
            K, E, IDX, EI, L, EL, b, t, IS, EIS, KP, KS, EP, ES, NK, NE, KT, BT,
            KTOKENS, ETOKENS, kv_layout,
        )  # fmt: skip
        kv_smem.store(kv)
    elif swa_after_wait:
        if t < KT:
            kv, valid = _load_tile(
                K, E, IDX, EI, L, EL, b, t, IS, EIS, KP, KS, EP, ES, NK, NE, KT, BT,
                KTOKENS, ETOKENS, kv_layout,
            )  # fmt: skip
            kv_smem.store(kv)

    q = gl.load(Q + b * QS + qh[:, None] * QH + qd[None, :])
    if FUSE_QNR:
        qf = q.to(gl.float32)
        ss = gl.sum(qf * qf, axis=1)
        rinv = (1.0 / gl.sqrt(ss * (1.0 / 512.0) + QEPS))[:, None]
        # Each thread holds 8 contiguous columns, so the RoPE partner (column
        # j ^ 1) is in its own registers: swap the even/odd halves in place.
        even, odd = gl.split(gl.reshape(qf, [H, 256, 2]))
        swapped = gl.convert_layout(
            gl.reshape(gl.join(odd, even), [H, 512]), qf.type.layout
        )
        qp = gl.where(is_rope[None, :], swapped, qf)
        q = (
            (qf * rinv) * cosf[None, :] + sign[None, :] * (qp * rinv) * sinf[None, :]
        ).to(gl.bfloat16)
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [H, 512], nvmma, value=q)
    fence_async_shared()
    tcgen05_mma(kv_smem, q_smem.permute((1, 0)), score_tmem, use_acc=False)
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
    p_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, H], p_layout, value=p_hi)
    if COMPENSATE:
        # Both PV MMAs go out back to back (in-order on one accumulator), one wait.
        p_lo = (prob - p_hi.to(gl.float32)).to(gl.bfloat16)
        p_lo_smem = gl.allocate_shared_memory(
            gl.bfloat16, [BT, H], p_layout, value=p_lo
        )
        fence_async_shared()
        tcgen05_mma(kv_smem.permute((1, 0)), p_smem, out_tmem, use_acc=False)
        tcgen05_mma(kv_smem.permute((1, 0)), p_lo_smem, out_tmem, use_acc=True)
    else:
        fence_async_shared()
        tcgen05_mma(kv_smem.permute((1, 0)), p_smem, out_tmem, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=1)
    mbarrier.invalidate(bar)
    out_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (512, H), TensorMemoryLayout(block=(128, H), col_stride=1), 4
    )
    value = out_tmem.load(out_layout)
    hd_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    value_hd = gl.convert_layout(value.permute((1, 0)), hd_layout)
    h = gl.arange(0, H, gl.SliceLayout(1, hd_layout))
    d = gl.arange(0, 512, gl.SliceLayout(0, hd_layout))
    gl.store(PART + ((b * NT + t) * H + h[:, None]) * 512 + d[None, :], value_hd)
    stat_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    hs = gl.arange(0, H, stat_layout)
    gl.store(MAX + (b * NT + t) * H + hs, gl.convert_layout(mx, stat_layout))
    gl.store(SUM + (b * NT + t) * H + hs, gl.convert_layout(denom, stat_layout))


@gluon.jit
def partial_gluon(
    Q,
    K,
    E,
    IDX,
    EI,
    L,
    EL,
    PART,
    MAX,
    SUM,
    FREQS,
    POS,
    SYNC,
    QS: gl.constexpr,
    QH: gl.constexpr,
    IS: gl.constexpr,
    EIS: gl.constexpr,
    KP: gl.constexpr,
    KS: gl.constexpr,
    EP: gl.constexpr,
    ES: gl.constexpr,
    NK: gl.constexpr,
    NE: gl.constexpr,
    NT: gl.constexpr,
    KT: gl.constexpr,
    BT: gl.constexpr,
    H: gl.constexpr,
    SCALE: gl.constexpr,
    KTOKENS: gl.constexpr,
    ETOKENS: gl.constexpr,
    COMPENSATE: gl.constexpr,
    SWAP_AB: gl.constexpr,
    FUSE_QNR: gl.constexpr,
    QEPS: gl.constexpr,
    ROPE: gl.constexpr,
    USE_PDL: gl.constexpr,
    TILE_MODE: gl.constexpr = 0,
    C_RATIO: gl.constexpr = 128,
    TOKENS_PER_REQ: gl.constexpr = 1,
    DECOUPLE: gl.constexpr = False,
):
    gl.static_assert(SWAP_AB and H == 16 and (BT == 64 or BT == 128))
    if USE_PDL and TILE_MODE != 2:
        # Scheduling only: the dependent's griddepcontrol.wait still waits for
        # this grid to finish, so the combine may become resident early.
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    b = gl.program_id(0)
    _partial_tile(
        Q, K, E, IDX, EI, L, EL, PART, MAX, SUM, FREQS, POS, b, gl.program_id(1),
        QS, QH, IS, EIS, KP, KS, EP, ES, NK, NE, NT, KT, BT, H, SCALE, KTOKENS,
        ETOKENS, COMPENSATE, FUSE_QNR, QEPS, ROPE, USE_PDL, TILE_MODE, C_RATIO,
        TOKENS_PER_REQ,
    )  # fmt: skip
    if USE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    if TILE_MODE == 2 and DECOUPLE:
        # Release this token's late tile to the combine, which waits on the flag
        # instead of a graph edge (eb5_combine_push_global_kernel, params.sync).
        if _is_late(EL, POS, b, C_RATIO, TOKENS_PER_REQ):
            # seq = done count >> 20; it cannot advance before this flag is seen,
            # since the combine waits on it.
            gl.barrier()
            done = gl.load((SYNC + 2).to(gl.pointer_type(gl.int64)), volatile=True)
            seq = (done >> 20).to(gl.int32)
            gl.atomic_xchg(SYNC + 8 + b, seq + 1, sem="release", scope="gpu")

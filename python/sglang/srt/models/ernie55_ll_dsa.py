"""ERNIE 5.5 LowLatency prepare of the full-MLA (DSA) layers.

Port of the internal ``EB5LLDsaPrepare`` (``srt/internal/models/eb5/ll_attn_backend.py``),
the fused q_a norm + group-32 quant (``EB5AttentionMLA._fused_q_a_norm``), the
indexer k-side / q-side hooks (``EB5FullMLAIndexer.eb5_issue_k_side`` and its
``_fused_q_prepare_and_store``) and the attn-TP VHA postmix
(``vha_postmix_attn_tp.apply``). Everything here is reached only with
``SGLANG_ERNIE55_MLA_SWAPAB`` and ``SGLANG_ERNIE55_SINGLE_BATCH_LL`` both on
(``mla_swapab_enabled``); the indexer hooks are per instance, so
``dsa_indexer.py`` is unchanged.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from types import SimpleNamespace
from typing import Optional

import torch

from sglang.srt.layers.layernorm import RMSNorm

logger = logging.getLogger(__name__)

_DSA_LOGGED: set = set()


class _FusedQALayerNorm(RMSNorm):
    """q_a_layernorm that lets its owning MLA attention take the fused norm +
    group-32 quant path; the bf16 normed output is still returned."""

    def forward(self, x, *args, **kwargs):
        owner = self.__dict__.get("_eb5_owner")
        if owner is not None and not args and not kwargs:
            normed = owner._fused_q_a_norm(x)
            if normed is not None:
                return normed
        return super().forward(x, *args, **kwargs)


def _capture_mode() -> bool:
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return bool(get_is_capture_mode())


def fused_q_a_norm(host, q: torch.Tensor) -> Optional[torch.Tensor]:
    """q_a RMSNorm + g32 quant in one kernel: the bf16 normed q, with the
    (fp8, uint8 sf) tuple stashed for q_b_proj / gate_proj / indexer wq_b.
    None -> stock RMSNorm."""
    from sglang.srt.layers.quantization.ernie55_halops_ll_gemm import (
        g32_prequant_ready,
    )

    host._q_lora_prequant = None
    indexer = getattr(host, "indexer", None)
    wq_b = indexer.wq_b if indexer is not None else None
    if wq_b is not None:
        wq_b._shared_q_prequant = None
    if not (
        q.dim() == 2
        and _capture_mode()
        and q.shape[-1] % 32 == 0
        and q.stride(-1) == 1
        and g32_prequant_ready(host.q_b_proj, q.shape[0])
    ):
        return None
    from sglang.kernels.ops.attention.dsv4.ernie55_rmsnorm_group_quant import (
        rmsnorm_group_quant_ue8m0_with_norm,
    )

    norm = host.q_a_layernorm
    normed, q_fp8, sf = rmsnorm_group_quant_ue8m0_with_norm(
        q, norm.weight, norm.variance_epsilon, group_size=32
    )
    host._q_lora_prequant = (q_fp8, sf)
    if wq_b is not None and g32_prequant_ready(wq_b, q.shape[0]):
        wq_b._shared_q_prequant = host._q_lora_prequant
    return normed


def _prequant_aware_wq_b_forward(self, x):
    # Consume the q_lora (fp8, sf) the parent MLA stashed from its fused q_a
    # norm-quant, so wq_b does not quantize the same q_lora again.
    pre = self._shared_q_prequant
    if pre is not None and not isinstance(x, tuple):
        return type(self).forward(self, pre)
    return type(self).forward(self, x)


def install_indexer_hooks(indexer) -> None:
    """Per-instance hooks on a stock ``Indexer``: wq_b takes the shared q
    prequant, and ``_fused_q_prepare_and_store`` runs only its q half when
    :func:`issue_k_side` already issued the k half on a side lane."""
    from sglang.kernels.ops.attention.dsa.indexer_q import (
        fused_q_indexer_rope_first_quant,
    )

    indexer._eb5_k_side = None
    indexer._eb5_q_side_fork = None
    wq_b = indexer.wq_b
    wq_b._shared_q_prequant = None
    wq_b.forward = functools.partial(_prequant_aware_wq_b_forward, wq_b)
    original = indexer._fused_q_prepare_and_store

    def _fused_q_prepare_and_store(
        x, q_lora, positions, forward_batch, layer_id, act_quant, **kwargs
    ):
        pre = indexer._eb5_k_side
        if pre is None:
            return original(
                x, q_lora, positions, forward_batch, layer_id, act_quant, **kwargs
            )
        # The q half of the single-stream path; the k half ran at layer entry.
        indexer._eb5_k_side = None
        weights_raw, stream = pre
        assert kwargs.get("num_tokens") is None
        q_scale_gate = indexer.softmax_scale * indexer.n_heads**-0.5
        q = indexer.wq_b(q_lora)[0].view(-1, indexer.n_heads, indexer.head_dim)
        fork, indexer._eb5_q_side_fork = indexer._eb5_q_side_fork, None
        if fork is not None:
            fork()
        current = torch.cuda.current_stream()
        current.wait_stream(stream)
        weights_raw.record_stream(current)
        return fused_q_indexer_rope_first_quant(
            q.contiguous(),
            weights_raw,
            q_scale_gate,
            indexer._indexer_cos_sin_cache,
            positions,
            hadamard=indexer.fused_indexer_hadamard,
            neox_style=indexer.is_neox_style,
        )

    indexer._fused_q_prepare_and_store = _fused_q_prepare_and_store
    indexer._eb5_hooked = True


def issue_k_side(indexer, x, positions, forward_batch, layer_id, stream) -> None:
    """wk_weights_proj + index-K store on ``stream`` (the k half of the
    single-stream ``_fused_q_prepare_and_store``); the next indexer call picks
    the head-gate half up instead of recomputing it."""
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant

    assert indexer._eb5_k_side is None
    with torch.cuda.stream(stream):
        kw, _ = indexer.wk_weights_proj(x)
        key, weights_raw = kw.split([indexer.head_dim, indexer.n_heads], dim=-1)
        indexer._fused_k_prepare_and_store(
            key,
            positions,
            forward_batch,
            layer_id,
            act_quant,
            out_cache_loc=forward_batch.out_cache_loc,
        )
    indexer._eb5_k_side = (weights_raw, stream)


class Ernie55LLDsaPrepare:
    """LowLatency target-verify prepare of the full-MLA (DSA) layers: the base
    absorb prepare's kernels on three side lanes, so only the indexer chain stays
    on the main stream.

    * lane_k (forked at layer entry): wk_weights_proj + index-K store, which read
      only the layer input; main joins it before the indexer q rope/quant.
    * main: wqkv_a -> q norm -> indexer wq_b -> q rope/quant -> logits -> top-k.
    * lane_q (forked right after the indexer wq_b): kv norm -> q_b -> w_kc ->
      rope -> KV store, none of which the indexer reads.
    * lane_w (forked at layer entry, swapAB layers only): the partial's window
      rows and fp32 sink, so the partial reads the top-k output directly.

    Main joins every lane before returning, so the attention core sees the same
    tensors as after the base prepare; it only skips the KV store (``core``).
    """

    def __init__(self, host) -> None:
        self._h = host
        self._kv_presaved = False

    @contextlib.contextmanager
    def core(self):
        """forward_absorb_core: skip the KV store the prepare already did."""
        if not self._kv_presaved:
            yield
            return
        self._kv_presaved = False
        mqa = self._h.attn_mqa
        own_forward = mqa.__dict__.get("forward")
        mqa.forward = functools.partial(mqa.forward, save_kv_cache=False)
        try:
            yield
        finally:
            if own_forward is None:
                del mqa.forward
            else:
                mqa.forward = own_forward

    def prepare(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        llama_4_scaling,
        prev_topk_indices,
    ):
        """forward_absorb_prepare's first ten slots, or None for the stock prepare."""
        self._kv_presaved = False
        forward_batch._eb5_dsa_early = None
        if not self._ready(
            hidden_states, forward_batch, llama_4_scaling, prev_topk_indices
        ):
            return None
        return self._prepare(positions, hidden_states, forward_batch, zero_allocator)

    def _ready(
        self, hidden_states, forward_batch, llama_4_scaling, prev_topk_indices
    ) -> bool:
        """Whether this prepare reproduces the base absorb prepare's kernels for
        this call (target-verify capture, fused indexer, bf16 w_kc)."""
        from sglang.srt.layers.attention.dsa.dsa_prefill_cuda_graph import (
            _is_in_piecewise_or_breakable_cuda_graph,
        )
        from sglang.srt.layers.attention.ernie55_backend import (
            Ernie5_5MlaAttnBackend,
        )
        from sglang.srt.layers.cp.utils import is_cp_active
        from sglang.srt.layers.layer_boundary import get_attn_tp_context
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.models.deepseek_common.attention_forward_methods import (
            forward_mla as base_mla,
        )
        from sglang.srt.models.ernie55_swapab_dsa import mla_swapab_enabled

        h = self._h
        indexer = getattr(h, "indexer", None)
        lanes = getattr(h, "attn_alt_streams", None)
        if not (
            mla_swapab_enabled()
            and getattr(indexer, "_eb5_hooked", False)
            and lanes is not None
            and len(lanes) >= 4
            and _capture_mode()
            and forward_batch.forward_mode.is_target_verify()
            and isinstance(hidden_states, torch.Tensor)
            and isinstance(get_attn_backend(), Ernie5_5MlaAttnBackend)
        ):
            return False
        mixin = base_mla.DeepseekMLAForwardMixin
        return (
            h.q_lora_rank is not None
            and h.use_dsa
            and h.should_run_indexer(prev_topk_indices)
            and llama_4_scaling is None
            and h.current_attention_backend
            in base_mla.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
            and indexer.use_dsa_indexer_fusion
            and not getattr(indexer, "use_gfx950_fused_indexer", False)
            and not indexer._aiter_fused_fp8_active(forward_batch)
            and not indexer.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is None
            and not is_cp_active(forward_batch)
            and not get_attn_tp_context().input_scattered
            and not _is_in_piecewise_or_breakable_cuda_graph()
            and not indexer._should_skip_logits_computation(forward_batch)
            and not h._can_fuse_bmm_into_attention(forward_batch)
            and getattr(h, "dcp_size", 1) == 1
            and not h.is_dcp_q_replicate_phase(forward_batch)
            and not h.needs_unfused_mla_qkv
            and not base_mla._ENABLE_DSA_Q8KV8_BORN_FP8_Q
            and not base_mla._ENABLE_DSA_Q8KV8_QPREP_OVERLAP
            and not base_mla._SGLANG_EXPERIMENTAL_LORA_OPTI
            and not base_mla.is_kv_b_lora_active(h)
            and base_mla.kv_b_lora_correction(h) is None
            and not getattr(h, "_kimi_split_gguf_kv_b", False)
            and not h.use_deep_gemm_bmm
            and h.w_kc.dtype == torch.bfloat16
            and h.rotary_emb is not None
            and not h._fuse_rope_for_trtllm_mla(forward_batch)
            and getattr(h.prepare_mla_qkv, "__func__", None) is mixin.prepare_mla_qkv
            and getattr(h.run_mla_indexer, "__func__", None) is mixin.run_mla_indexer
        )

    def _early_ready(self, backend, num_tokens) -> bool:
        """Whether maybe_swapab_dsa_early would take the swapAB partial for this
        layer, i.e. the window rows can be built up front."""
        from sglang.kernels.arch import current_arch

        h = self._h
        return (
            getattr(h, "learnable_sink_param", None) is not None
            and h.num_local_heads == 16
            and h.kv_lora_rank == 512
            and h.qk_rope_head_dim == 64
            and 0 < num_tokens <= 8
            and backend.window > 0
            and not backend.dsa_kv_cache_store_fp8
            and backend._read_rows is None
            and bool(current_arch().is_sm(10))
        )

    def _prepare(self, positions, hidden_states, forward_batch, zero_allocator):
        from sglang.srt.layers.layer_boundary import get_attn_tp_context
        from sglang.srt.model_executor.forward_context import get_attn_backend

        h = self._h
        h._q8kv8_qprep_overlap_pending = False
        current = torch.cuda.current_stream()
        lanes = h.attn_alt_streams
        lane_q, lane_k, lane_w = lanes[1], lanes[2], lanes[3]
        indexer = h.indexer
        num_tokens = hidden_states.shape[0]
        lane_k.wait_stream(current)
        issue_k_side(
            indexer, hidden_states, positions, forward_batch, h.layer_id, lane_k
        )
        backend = get_attn_backend()
        early = None
        early_ok = self._early_ready(backend, num_tokens)
        if (h.layer_id, num_tokens) not in _DSA_LOGGED:
            _DSA_LOGGED.add((h.layer_id, num_tokens))
            logger.info(
                "LL DSA prepare: layer %d tokens %d early %s",
                h.layer_id,
                num_tokens,
                early_ok,
            )
        if early_ok:
            lane_w.wait_stream(current)
            with torch.cuda.stream(lane_w):
                rows_idx, lengths, ilen = backend.eb5_window_rows(num_tokens)
                sink16 = (
                    h.learnable_sink_param.reshape(-1)[: h.num_local_heads]
                    .to(torch.float32)
                    .contiguous()
                )
            early = SimpleNamespace(
                rows=num_tokens,
                topk=int(backend.dsa_index_topk),
                scale=h.attn_mqa.scaling,
                rows_idx=rows_idx,
                lengths=lengths,
                ilen=ilen,
                sink16=sink16,
            )

        q, latent_cache = (
            get_attn_tp_context()
            .fetch_qkv_latent()
            .split([h.q_lora_rank, h.kv_lora_rank + h.qk_rope_head_dim], dim=-1)
        )
        k_nope = latent_cache[..., : h.kv_lora_rank]
        q_lora = h.q_a_layernorm(q)
        side = {}

        def q_side():
            lane_q.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(lane_q):
                kn = h.kv_a_layernorm(k_nope).unsqueeze(1)
                qb = h.q_b_proj_forward(q_lora)
                q_nope, q_pe, k_pe = h._split_q_nope_pe(qb, latent_cache)
                q_nope_out = torch.bmm(q_nope.transpose(0, 1), h.w_kc).transpose(0, 1)
                q_pe, k_pe = h.rotary_emb(positions, q_pe, k_pe)
                attn = h.attn_mqa
                backend.kv_layers.layer(attn.layer_id).store_mla(
                    attn,
                    forward_batch.out_cache_loc,
                    kn.view(-1, attn.tp_k_head_num, attn.v_head_dim),
                    k_pe,
                )
            side.update(q_pe=q_pe, k_pe=k_pe, q_nope_out=q_nope_out, k_nope=kn)

        # The indexer fires q_side right after its wq_b.
        indexer._eb5_q_side_fork = q_side
        try:
            topk_indices = h.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=h.layer_id,
            )
        finally:
            indexer._eb5_q_side_fork = None
            k_unjoined = indexer._eb5_k_side is not None
            indexer._eb5_k_side = None
        if not side:
            q_side()
        if k_unjoined:
            current.wait_stream(lane_k)
        current.wait_stream(lane_q)
        # Lane-allocated tensors read on main stay reserved until those reads are done.
        for t in (side["q_nope_out"], side["q_pe"], side["k_nope"], side["k_pe"]):
            t.record_stream(current)
        if early is not None:
            current.wait_stream(lane_w)
            for t in (early.rows_idx, early.lengths, early.ilen, early.sink16):
                t.record_stream(current)
            forward_batch._eb5_dsa_early = early
        self._kv_presaved = True
        return (
            side["q_pe"],
            side["k_pe"],
            side["q_nope_out"],
            side["k_nope"],
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            None,
            None,
        )


class _DeferredGate:
    """The output-gate GEMM issued on a side stream; ``join`` orders the main
    stream after it."""

    __slots__ = ("gate", "stream")

    def __init__(self, gate: torch.Tensor, stream) -> None:
        self.gate = gate
        self.stream = stream

    def join(self) -> torch.Tensor:
        current = torch.cuda.current_stream()
        current.wait_stream(self.stream)
        self.gate.record_stream(current)
        return self.gate


def build_full_mix(postmix, out: torch.Tensor) -> None:
    """Fill ``out`` ([H, H]) in place with the global ``I + V @ U.T`` from this
    rank's U/V shards ([H / tp, rank] each), gathered over attention TP."""
    from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
    from sglang.srt.runtime_context import get_parallel

    tp = get_parallel().attn_tp_size
    u_local = postmix.vha_postmix_U.data.contiguous()
    v_local = postmix.vha_postmix_V.data.contiguous()
    n_local, rank = u_local.shape
    full = []
    for local in (u_local, v_local):
        gathered = torch.empty(
            (tp, n_local, rank), dtype=local.dtype, device=local.device
        )
        attn_tp_all_gather_into_tensor(gathered, local)
        full.append(gathered.reshape(tp * n_local, rank).float())
    u_full, v_full = full
    num_heads = u_full.shape[0]
    eye = torch.eye(num_heads, dtype=torch.float32, device=u_full.device)
    # In place: a captured CUDA graph reads this buffer's address.
    out.copy_(torch.addmm(eye, v_full, u_full.t()))


def gathered_postmix(
    full_mix: torch.Tensor, attn_output: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Attn-TP VHA postmix (internal ``vha_postmix_attn_tp.apply``): gather this
    rank's heads to the full head set, apply the full ``[H, H]`` mix and keep
    this rank's out-head slice. Returns ``attn_output``'s shape."""
    from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
    from sglang.srt.models.ernie55_fast_allgather import (
        try_fast_allgather_global,
        try_fast_allgather_rank_major,
    )
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tp_size, tp_rank = parallel.attn_tp_size, parallel.attn_tp_rank
    n_local = num_heads // tp_size
    head_dim = attn_output.shape[-1]
    if attn_output.dim() < 3 or attn_output.shape[-2] != n_local:
        head_dim //= n_local
    o_local = attn_output.reshape(-1, n_local, head_dim)
    num_tokens = o_local.shape[0]
    o_full = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=o_local.dtype, device=o_local.device
    )
    if not try_fast_allgather_global(o_local, o_full, n_local, head_dim, num_heads):
        gathered = torch.empty(
            (tp_size, num_tokens, n_local, head_dim),
            dtype=o_local.dtype,
            device=o_local.device,
        )
        o_c = o_local.contiguous()
        if not try_fast_allgather_rank_major(o_c, gathered):
            attn_tp_all_gather_into_tensor(gathered, o_c)
        o_full = gathered.permute(1, 0, 2, 3).reshape(num_tokens, num_heads, head_dim)
    mix_local = full_mix[tp_rank * n_local : (tp_rank + 1) * n_local]
    return torch.matmul(mix_local, o_full).reshape(attn_output.shape)

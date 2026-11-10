"""Open-tree port of the internal EB5LLAttnBackend (sglang/srt/internal/models/
eb5/ll_attn_backend.py): the LowLatency (single-batch decode) execution side of
the ERNIE5.5 HCA gated attention.

The ERNIE5.5 HCA layer owns one of these as ``self._ll`` when the LowLatency
switch (``SGLANG_ERNIE55_HCA_QKV_PREP``) is on, and routes its attention forward
through this object's seams -- prepare_hca_qkv / capture_gate_input / take_gate /
gate_mul_g32 / fused_postmix_rope_wo_a_quant / arm_swapab / end_forward -- each
of which returns None (or no-ops) to hand control back to the stock path, the
same discipline internal uses so the layer with ``_ll = None`` behaves exactly
like mainline. This is a dedicated backend, NOT a hook grafted onto open's
generic MQALayer multi-stream prepare.

``host`` is the layer that owns the modules (wq_b / q_norm / kv_norm / wo_a /
wo_b / compressor / wqkv_a); the backend reads live attributes off it and owns
all LowLatency state.

Gate layout (``host.gate_merged_into_wq_b``):
  * Merged (default): the gate is the second half of wq_b
    (Ernie5_5GatedQMixin, MergedColumnParallelLinear [q_up, gate]) and falls out
    of _gated_q_norm_rope; the deferred-gate seams below stay inert.
  * Split (SGLANG_ERNIE55_HCA_SPLIT_GATE, internal's LowLatency layout): a
    separate gate_proj reads the q_norm latent. capture_gate_input (q_norm hook,
    or replayed by prepare_hca_qkv with the g32 prequant tuple) stashes the
    input; issue_gate_deferred launches gate_proj on alt_streams[-1] (LL GEMM
    capped at 64 SMs) right after the swapab partial, so it overlaps the combine
    push + K3 + wo_a; take_gate joins it before wo_b.

HCA decode prepare (prepare_hca_qkv): q_norm + g32 quant and the kv store run as
one main-stream launch behind wqkv_a, only the compressor forks.
  * SGLANG_ERNIE55_SWAPAB_QNR_FOLD: arm_swapab arms skip_qnr when the decode
    will take maybe_swapab_decode; wq_b's raw 16-head q is stashed on
    attn_mqa._eb5_raw_q16 and the gluon partial folds q-norm + RoPE.
  * SGLANG_ERNIE55_SWAPAB_LATE_C128 (needs the fold): the compressor join is
    handed to maybe_swapab_decode, which waits on it only for the split-off
    partial tile reading this forward's new c128 entry; under attn-TP that wait
    moves into the combine push (per-token late_sync flag) and the stream join
    to take_gate / end_forward.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# SM cap for the side-stream gate_proj LL GEMM, so it overlaps the main stream.
_GATE_SM_COUNT = 64
# SM cap for wq_b's LL GEMM so the compressor kernel stays resident next to it.
_WQ_B_SM_COUNT = 120


class Ernie55LLAttnBackend:
    def __init__(
        self, host, alt_streams: Optional[List[torch.cuda.Stream]] = None
    ) -> None:
        self._h = host
        self.skip_qnr = False
        self._freqs_real: Optional[torch.Tensor] = None
        self.gate_pending: Optional[torch.Tensor] = None
        self.gate_pending_on_alt = False
        # Set at q_norm, consumed by issue_gate_deferred at the K3 entry.
        self.gate_deferred = False
        # In-flight forward_batch, carries the swapab partial-done event.
        self._fb = None
        self.alt_stream: Optional[torch.cuda.Stream] = None
        # Full [64,64] folded postmix mix, built lazily for the K3 fused kernel.
        self._full_mix: Optional[torch.Tensor] = None
        # Only the split layout has a separate gate_proj to defer.
        if self._split_gate:
            if (
                torch.cuda.is_available()
                and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
                and alt_streams is not None
                and len(alt_streams) >= 2
            ):
                self.alt_stream = alt_streams[-1]
            from sglang.srt.layers.quantization.ernie55_halops_ll_gemm import (
                LL_GEMM_SM_COUNT_ATTR,
            )

            setattr(host.gate_proj, LL_GEMM_SM_COUNT_ATTR, _GATE_SM_COUNT)

    @property
    def _split_gate(self) -> bool:
        return not getattr(self._h, "gate_merged_into_wq_b", True)

    def cap_wq_b_sms(self) -> None:
        """Cap wq_b's LL GEMM SMs so the side-stream gate_proj and compressor
        kernels stay resident next to it (internal cap_wq_b_sms)."""
        from sglang.srt.layers.quantization.ernie55_halops_ll_gemm import (
            LL_GEMM_SM_COUNT_ATTR,
        )

        setattr(self._h.wq_b, LL_GEMM_SM_COUNT_ATTR, _WQ_B_SM_COUNT)

    def q_b(self, q, positions, q_out=None):
        """swapAB q-norm-rope fold on the split layout: run wq_b and stash the
        RAW 16-head q for the gluon partial, skipping the standalone norm+rope.
        Returns None when the fold is not armed, so the caller takes the stock
        path (internal q_b)."""
        h = self._h
        if not self.skip_qnr:
            self._clear_raw_q16()
            return None
        q_heads, _ = h.wq_b(q)
        q_heads = q_heads.view(-1, h.n_local_heads, h.head_dim)
        h.attn_mqa._eb5_raw_q16 = q_heads
        return q_out if q_out is not None else torch.empty_like(q_heads)

    def q_b_merged(self, q, q_out=None):
        """Merged-layout hook: only the swapAB fold's raw-q stash lives here. `q`
        is the already-split q-up view. Returns None when the fold is not armed
        (internal q_b_merged)."""
        if self.skip_qnr:
            self._h.attn_mqa._eb5_raw_q16 = q.contiguous()
            return q_out if q_out is not None else torch.empty_like(q)
        self._clear_raw_q16()
        return None

    def _clear_raw_q16(self) -> None:
        mqa = self._h.attn_mqa
        if getattr(mqa, "_eb5_raw_q16", None) is not None:
            mqa._eb5_raw_q16 = None

    def _hca_qkv_prep_ready(self, x: torch.Tensor) -> bool:
        """Mirror of internal _hca_qkv_prep_ready for open's merged-gate HCA:
        capture mode, fused wqkv_a, indexer-less compressor layer, group-32 /
        head_dim 512 shapes, the compressed SWA pool, and wq_b has a group-32 LL
        plan at this M (so it can consume the kernel's (fp8, uint8) q-output)."""
        h = self._h
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        if not (
            get_is_capture_mode()
            and getattr(h, "fuse_wqa_wkv", False)
            and getattr(h, "compressor", None) is not None
            and getattr(h, "indexer", None) is None
        ):
            return False
        if h.q_lora_rank % 32 or h.head_dim != 512:
            return False
        if envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get():
            return False
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        pool = get_token_to_kv_pool()
        if not isinstance(pool, DeepSeekV4TokenToKVPool):
            return False
        swa = pool.swa_kv_pool
        if (
            swa is None
            or getattr(pool, "uniform_fp8", False)
            or getattr(swa, "is_bf16_kv_cache", False)
        ):
            return False
        from sglang.srt.layers.quantization.ernie55_halops_ll_gemm import (
            g32_prequant_ready,
        )

        m = int(x.shape[0])
        if not g32_prequant_ready(h.wq_b, m):
            return False
        # Split layout: gate_proj consumes the same prequant tuple.
        return not self._split_gate or g32_prequant_ready(h.gate_proj, m)

    def prepare_hca_qkv(
        self, x, positions, forward_batch, attn_backend, q_out=None, x_quant=None
    ):
        """Internal prepare_hca_qkv for an indexer-less HCA layer: q_norm + g32
        quant and the kv store fused into one main-stream launch behind wqkv_a,
        only the compressor forks. Returns (q, None), or None for the stock
        MQALayer prepare."""
        if not self._hca_qkv_prep_ready(x):
            return None
        from sglang.kernels.ops.attention.dsv4.ernie55_hca_qkv_prep import (
            ernie55_hca_qkv_prep,
        )
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        h = self._h
        assert h.alt_streams is not None and len(h.alt_streams) >= 2
        current_stream = torch.cuda.current_stream()
        stream_compressor = h.alt_streams[1]
        stream_compressor.wait_stream(current_stream)
        qkv_a, _ = h.wqkv_a(x_quant if x_quant is not None else x)
        if self._freqs_real is None:
            self._freqs_real = (
                torch.view_as_real(h.freqs_cis).flatten(-2).contiguous().float()
            )
        pool = get_token_to_kv_pool()
        prequant = ernie55_hca_qkv_prep(
            qkv_a,
            h.q_norm.weight,
            h.q_norm.variance_epsilon,
            h.kv_norm.weight.data,
            h.eps,
            self._freqs_real,
            positions,
            attn_backend.get_swa_out_cache_loc(forward_batch),
            pool.get_swa_raw_buffer(h.layer_id),
            pool.swa_kv_pool.page_size,
            h.q_lora_rank,
        )
        # q_norm did not run on this path, so its gate-capture hook is replayed
        # here (split layout: stashes the tuple for the deferred gate_proj).
        self.capture_gate_input(None, None, prequant)
        with torch.cuda.stream(stream_compressor):
            attn_backend.forward_core_compressor(
                x, forward_batch, h.layer_id, h.compressor
            )
        q = h._compute_q_b(prequant, positions, q_out)
        if self._defer_compressor_join():
            # maybe_swapab_decode takes the join (see the module docstring).
            forward_batch._eb5_compressor_stream = stream_compressor
        else:
            current_stream.wait_stream(stream_compressor)
        return q, None

    def _defer_compressor_join(self) -> bool:
        """Internal hands the compressor join to the swapab decode whenever the
        fold is armed. Open gates it behind SGLANG_ERNIE55_SWAPAB_LATE_C128, and
        at attn-TP>1 also on SGLANG_ERNIE55_SWAPAB_PUSH: the late_sync flag is
        consumed only by the fused combine push (internal pushes whenever
        SINGLE_BATCH_LL is on), so without it the decode would have to raise."""
        if not (self.skip_qnr and envs.SGLANG_ERNIE55_SWAPAB_LATE_C128.get()):
            return False
        from sglang.srt.distributed.parallel_state import get_tp_group

        return get_tp_group().world_size == 1 or envs.SGLANG_ERNIE55_SWAPAB_PUSH.get()

    # --- deferred gate (split layout; inert when merged) --------------------
    def capture_gate_input(self, _module, _inp, out) -> None:
        """q_norm forward hook: stash the gate input and DEFER the side-stream
        gate_proj to `issue_gate_deferred` (right after the swapab partial, or the
        K3 entry as fallback), so the gate waits only on the partial and overlaps
        combine_push + K3 + wo_a. `take_gate` joins it before wo_b. No-op in the
        merged layout, where the gate is sliced in _gated_q_norm_rope."""
        if not self._split_gate:
            return None
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        gate_input = out
        if isinstance(out, tuple) and not self._is_prequant(out):
            gate_input = out[0]
        self._h._gate_input = gate_input
        self.gate_pending = None
        self.gate_pending_on_alt = False
        self.gate_deferred = self.alt_stream is not None and get_is_capture_mode()
        return None

    @staticmethod
    def _is_prequant(x) -> bool:
        return isinstance(x, tuple) and x[0].dtype == torch.float8_e4m3fn

    def issue_gate_deferred(self) -> None:
        """Launch the deferred gate_proj on the side stream, called right after
        the swapab partial (before the combine push). Waits on the swapab
        partial-done event when present, else on the whole main stream."""
        if not self.gate_deferred:
            return
        self.gate_deferred = False
        gate_input = self._h._gate_input
        partial_done = (
            None
            if self._fb is None
            else getattr(self._fb, "_eb5_partial_done_event", None)
        )
        if partial_done is not None:
            self.alt_stream.wait_event(partial_done)
        else:
            self.alt_stream.wait_stream(torch.cuda.current_stream())
        for t in gate_input if self._is_prequant(gate_input) else (gate_input,):
            t.record_stream(self.alt_stream)
        with torch.cuda.stream(self.alt_stream):
            gate, _ = self._h.gate_proj(gate_input)
        self.gate_pending = gate
        self.gate_pending_on_alt = True

    def join_compressor_late(self) -> None:
        """Join the compressor stream maybe_swapab_decode left open for the
        decoupled combine."""
        fb = self._fb
        stream = None if fb is None else getattr(fb, "_eb5_compressor_join_late", None)
        if stream is not None:
            fb._eb5_compressor_join_late = None
            torch.cuda.current_stream().wait_stream(stream)

    def take_gate(self) -> Optional[torch.Tensor]:
        """The side-stream gate joined onto the main stream, or None when the
        caller must run gate_proj itself (deferred issue never fired). Also joins
        the late compressor stream, since wo_b is the first consumer after it."""
        self.join_compressor_late()
        self.gate_deferred = False
        pending = self.gate_pending
        if pending is None:
            return None
        self.gate_pending = None
        if self.gate_pending_on_alt:
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(self.alt_stream)
            pending.record_stream(current_stream)
            self.gate_pending_on_alt = False
        return pending

    @staticmethod
    def engaged() -> bool:
        """The layer arms / ends a forward through this backend only when one of
        the HCA alignment switches is on, so with all three off the forward is
        byte-for-byte the previous one."""
        return (
            envs.SGLANG_ERNIE55_HCA_SPLIT_GATE.get()
            or envs.SGLANG_ERNIE55_SWAPAB_QNR_FOLD.get()
            or envs.SGLANG_ERNIE55_SWAPAB_LATE_C128.get()
        )

    def end_forward(self) -> None:
        """Fallback join when wo_b's take_gate did not run, plus the checks that
        the swapab decode took the deferred compressor join and the q-norm-rope
        fold. Open addition: the per-layer forward_batch seams are cleared so a
        later MLA / draft layer never sees this layer's issuer or event."""
        fb = self._fb
        if fb is None:
            return
        self.join_compressor_late()
        self._fb = None
        if getattr(fb, "_eb5_compressor_stream", None) is not None:
            raise RuntimeError(
                "HCA prepare deferred the compressor join but the swapab decode never "
                "took it; the layer would read an unfinished c128 entry"
            )
        if getattr(fb, "_eb5_swapab_qnr", None) is not None:
            raise RuntimeError(
                "fused_q_norm_rope was skipped for swapab but maybe_swapab_decode "
                "never ran; attention would have read the raw q"
            )
        fb._eb5_gate_issuer = None
        fb._eb5_partial_done_event = None
        self.skip_qnr = False
        self._clear_raw_q16()

    # --- swapab q-norm-rope fold --------------------------------------------
    def arm_swapab(self, forward_batch, positions, hidden_states) -> None:
        """swapab q-norm-rope fold: if this decode will run the swapab kernel,
        skip the standalone fused_q_norm_rope and let the gluon partial fold it
        in. Eligibility is the same swapab_decode_eligible() maybe_swapab_decode
        evaluates (a fail-safe assert there catches any divergence before raw q
        could reach FlashMLA)."""
        h = self._h
        self._fb = forward_batch
        # Cleared per layer so a layer that skips swapab never reads a stale event.
        forward_batch._eb5_partial_done_event = None
        # maybe_swapab_decode calls this right after the partial; the K3-entry
        # call stays as the fallback and is a no-op once the gate is issued.
        forward_batch._eb5_gate_issuer = self.issue_gate_deferred
        forward_batch._eb5_compressor_stream = None
        forward_batch._eb5_compressor_join_late = None
        # Open addition: no earlier forward's raw q may stand in for this one's.
        self._clear_raw_q16()
        skip_qnr = envs.SGLANG_ERNIE55_SWAPAB_QNR_FOLD.get() and self._swapab_will_run(
            forward_batch, hidden_states
        )
        self.skip_qnr = skip_qnr
        if skip_qnr:
            if self._freqs_real is None:
                self._freqs_real = (
                    torch.view_as_real(h.freqs_cis).flatten(-2).contiguous().float()
                )
            forward_batch._eb5_swapab_qnr = (self._freqs_real, positions, h.eps)
        else:
            forward_batch._eb5_swapab_qnr = None

    def _swapab_will_run(self, forward_batch, hidden_states) -> bool:
        """True iff DeepseekV4AttnBackend._forward_attention will reach
        maybe_swapab_decode for this layer and it will accept. Internal reads
        `backend.is_bf16_kv_cache` (open's backend has no such attribute); open
        evaluates the decode's own predicate on the same SWA cache format, plus
        the backend's earlier exits (trtllm, mega decode)."""
        h = self._h
        if getattr(h, "use_mega_attention", False):
            return False
        # Open addition: only MQALayer's multi-stream prepare (capture mode, same
        # predicate as MQALayer.forward's enable_multi_stream for a ratio-128 layer
        # at T <= 8) routes q through _compute_q_b, which stashes the raw q. The
        # eager _forward_prepare runs wq_b + norm-rope itself and would leave the
        # fold armed with no raw q.
        from sglang.srt.layers.cp.utils import is_cp_active
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        if not (
            get_is_capture_mode()
            and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and h.alt_streams is not None
            and not is_cp_active(forward_batch)
        ):
            return False
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )
        from sglang.srt.models.ernie55_swapab_decode import (
            swa_kv_format,
            swapab_decode_eligible,
        )

        if getattr(get_attn_backend(), "trtllm_attn", False):
            return False
        pool = get_token_to_kv_pool()
        if not isinstance(pool, DeepSeekV4TokenToKVPool) or pool.swa_kv_pool is None:
            return False
        kv_itemsize, kv_token_bytes = swa_kv_format(pool)
        return swapab_decode_eligible(
            forward_batch.forward_mode,
            h.attn_mqa.tp_q_head_num,
            int(hidden_states.shape[0]),
            kv_itemsize,
            kv_token_bytes,
        )

    # --- output-side folds (ported, live behind g32_prequant_ready) ---------
    def gate_mul_g32(self, o: torch.Tensor, gate: torch.Tensor):
        """Fold sigmoid-gate-mul into wo_b's fp8 input quant when wo_b has a
        group-32 LL plan: wo_b is fed a pre-quantized (q, sf) uint8 tuple
        (HALOps group_wise_quant layout), dropping the bf16 gated_o round trip
        and the separate quant. g32_prequant_ready is the "tuple will be
        accepted" signal. Returns (tuple,) or None (caller keeps
        fused_sigmoid_mul). Mirrors internal gate_mul_g32."""
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        m, hidden = o.shape[0], o.shape[-1]
        if not (get_is_capture_mode() and hidden % 32 == 0):
            return None
        from sglang.srt.layers.quantization.ernie55_halops_ll_gemm import (
            g32_prequant_ready,
        )

        if not g32_prequant_ready(self._h.wo_b, m):
            return None
        from sglang.kernels.ops.attention.dsv4.ernie55_sigmoid_gate_mul_group_quant import (
            ernie55_sigmoid_gate_mul_group_quant_ue8m0,
        )

        return (ernie55_sigmoid_gate_mul_group_quant_ue8m0(o, gate, group_size=32),)

    def fused_postmix_rope_wo_a_quant(self, o, positions, gathered: bool = False):
        """K3: fold inverse-RoPE + VHA cross-head postmix + wo_a fp8 quant into one
        kernel emitting int32-packed UE8M0 scales (same layout open's
        postmix_fp8_wo_a_quant emits, so _fp8_wo_a_einsum consumes it unchanged).

        `o` arrives as this rank's LOCAL heads [T, n_local_heads, 512]. The postmix
        sums over all 64 in-heads; under attn TP we all-gather o up to the full 64
        heads and feed the layer's full [64,64] mix, the kernel writes only this
        rank's out-head slice (write-mask hybrid). At attn_tp==1 the gather is
        skipped. `gathered=True` means the swapab decode already fused that gather
        into its combine+push, so `o` is the full 64 heads in global order and we
        skip the gather here. Returns (o_fp8 [T,G,D], o_s int32) or None to fall
        back to the layer's existing _project_wo_a path. Mirrors internal
        fused_postmix_rope_wo_a_quant, adapted to open's attribute names."""
        # Fallback issue point of the deferred gate (no-op once issued).
        self.issue_gate_deferred()
        h = self._h
        mix = self._full_postmix_mix()
        if mix is None or not getattr(h, "wo_a_fp8", False):
            return None
        from sglang.srt.layers.deep_gemm_wrapper.configurer import (
            DEEPGEMM_SCALE_UE8M0,
        )

        if not DEEPGEMM_SCALE_UE8M0:
            return None
        n_local_heads = h.n_local_heads
        expect_heads = 64 if gathered else n_local_heads
        if not (
            o.dim() == 3
            and o.shape[2] == 512
            and o.shape[1] == expect_heads
            and n_local_heads * self._attn_tp_size() == 64
            and mix.shape == (64, 64)
            and mix.dtype == torch.bfloat16
        ):
            return None

        attn_tp = self._attn_tp_size()
        if gathered:
            # swapab already pushed the full 64 heads into global order.
            o = o.contiguous()
        elif attn_tp > 1:
            num_tokens, head_dim = o.shape[0], o.shape[2]
            gathered = torch.empty(
                (attn_tp, num_tokens, n_local_heads, head_dim),
                dtype=o.dtype,
                device=o.device,
            )
            o_c = o.contiguous()
            from sglang.srt.models.ernie55_fast_allgather import (
                try_fast_allgather_push,
            )

            # Single-token decode / DSPARK draft gather: the fused swapab-combine
            # push does not fire here, so without this the gather fell back to the
            # NCCL RING_LL ring all-gather (~402us ring rendezvous, the top decode
            # kernel). Route it through the same 1-shot NVLink push workspace; keep
            # NCCL as the correctness fallback when the shard is not push eligible.
            if not try_fast_allgather_push(o_c, gathered, attn_tp):
                from sglang.srt.layers.dp_attention import (
                    attn_tp_all_gather_into_tensor,
                )

                attn_tp_all_gather_into_tensor(gathered, o_c)
                import os as _os

                if _os.environ.get("SGLANG_DEBUG_AG_COUNT"):
                    _g = globals()
                    _c = _g.setdefault("_AG_NCOUNT", {})
                    _c[num_tokens] = _c.get(num_tokens, 0) + 1
                    try:
                        with open(_os.environ["SGLANG_DEBUG_AG_COUNT"], "w") as _fh:
                            _fh.write(repr(dict(sorted(_c.items()))))
                    except Exception:
                        pass
            o = gathered.permute(1, 0, 2, 3).reshape(num_tokens, 64, head_dim)
        else:
            o = o.contiguous()

        from sglang.kernels.ops.attention.dsv4.ernie55_fused_postmix_rope_wo_a_quant import (
            ernie55_fused_postmix_rope_wo_a_quant,
        )

        return ernie55_fused_postmix_rope_wo_a_quant(
            o,
            mix,
            h.freqs_cis,
            positions,
            h.n_local_groups,
            out_heads=n_local_heads,
            out_head_offset=self._attn_tp_rank() * n_local_heads,
        )

    # --- helpers for the output postmix fold --------------------------------
    def _attn_tp_size(self) -> int:
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().attn_tp_size

    def _attn_tp_rank(self) -> int:
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().attn_tp_rank

    def _full_postmix_mix(self) -> Optional[torch.Tensor]:
        """Build (once) the full [64,64] folded mix I + V @ U.T the fused kernel
        needs. Open's sharded VhaPostMix keeps only this rank's U/V [16, rank]
        and reduces a low-rank latent at runtime; the kernel instead all-gathers
        heads and runs the full [64,64] mix, so every rank needs the full
        matrix. All-gather U/V across the attn-tp group and fold here. For an
        unsharded postmix the layer already holds the full mix buffer."""
        if self._full_mix is not None:
            return self._full_mix
        h = self._h
        postmix = getattr(h, "postmix", None)
        if postmix is None:
            return None
        if not postmix.sharded:
            self._full_mix = postmix.mix.to(torch.bfloat16).contiguous()
            return self._full_mix
        attn_tp = self._attn_tp_size()
        u_local = postmix.vha_postmix_U.data  # [16, rank]
        v_local = postmix.vha_postmix_V.data  # [16, rank]
        if u_local.shape[0] * attn_tp != 64:
            return None
        from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor

        rank_dim = u_local.shape[1]
        u_g = torch.empty(
            (attn_tp, u_local.shape[0], rank_dim),
            dtype=u_local.dtype,
            device=u_local.device,
        )
        v_g = torch.empty_like(u_g)
        attn_tp_all_gather_into_tensor(u_g, u_local.contiguous())
        attn_tp_all_gather_into_tensor(v_g, v_local.contiguous())
        u_full = u_g.reshape(64, rank_dim).float()
        v_full = v_g.reshape(64, rank_dim).float()
        eye = torch.eye(64, dtype=torch.float32, device=u_full.device)
        self._full_mix = torch.addmm(eye, v_full, u_full.t()).to(torch.bfloat16).contiguous()
        return self._full_mix

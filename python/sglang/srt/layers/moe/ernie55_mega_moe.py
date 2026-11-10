# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ERNIE 5.5 latent MoE as one HALOps ``low_latency_moe`` launch (the TP MoE backend).

The whole small-batch decode MoE of a layer -- router, top-k, latent fc1, routed
experts, the shared expert, the combine and the AllReduce + RMSNorm + fc2 tail --
runs in a single kernel launch for ``1 <= m <= 16`` on a 4-GPU TP group (SM100,
NVLink multicast)::

    out[m, H] = fc2( RMSNorm( AllReduce_tp( sum_k w_k * expert_k( fc1_latent(x) ) ) ) )
                + AllReduce_tp( shared_expert(x) )

Off by default; turn it on with ``SGLANG_ERNIE55_MEGA_MOE=1``. The backend is a
plain helper owned by :class:`~sglang.srt.models.ernie55.Ernie5_5Moe`: it reads
that module's weights, builds the kernel operands once after the loader has
quantised them, and issues the fused launch on the decode path. The model's own
MoE path stays the fallback for prefill and any ineligible shape.

Recipes (``SGLANG_ERNIE55_MEGA_MOE_QUANT``): ``w8a8-blockwise`` (the model's FP8
weights with 128x128 block scales) and ``w8a8-mxfp8`` (1x32 UE8M0 SfAtom scales). The
mxfp8 recipe quantizes the latent fc1 and shared up/down natively per 1x32 group from
BF16 (``SGLANG_ERNIE55_MEGA_MOE_TRUE_GROUP32``, default on; off derives the SfAtom
scale from the block128 weight). The routed experts stay on the derived scale unless
``..._TRUE_GROUP32_EXPERTS`` is set, since a second expert copy is ~69 GB/rank and the
block128 prefill fallback is still needed. fc2 stays block128 while its kernel is not
on group32 yet. The routed ``w4a8-mxfp4`` recipe quantizes the routed experts to a
packed FP4 (E2M1) copy with 1x32 UE8M0 SfAtom scales (the ``per_token_cast_to_fp4``
quantiser) and runs ``low_latency_moe`` with ``quant_mode="w4a8-mxfp4"``; the latent
fc1, the shared expert and the fc2 tail stay MXFP8, exactly as the mxfp8 recipe. The
FP4 experts are a SECOND copy alongside the loader's block128 FP8 that the triton
prefill fallback still reads (a single FP4 copy awaits a HALOps W4A8 prefill runner).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MAX_M = 16
_NARROW_M = 8  # m <= 8 runs the narrow kernel variant, larger m the wide one
_WORLD = 4
_NUM_EXPERTS = 512
_FP8 = torch.float8_e4m3fn
_QUANT_MODES = ("w8a8-blockwise", "w8a8-mxfp8", "w4a8-mxfp4")
_SF_ATOM_ROWS, _SF_ATOM_COLS, _SF_ATOM_BYTES = 128, 4, 512  # one CUTLASS SfAtom

# One workspace (scratchpad + symmetric-memory tail mailbox) per (device, kernel
# variant). Shared by every layer of the process and created at model build, never
# inside a CUDA graph capture (a captured allocation is re-zeroed on every replay).
_WORKSPACES: dict = {}
_logged: set = set()

def mega_moe_enabled() -> bool:
    """Master switch for the fused TP MoE backend."""
    return bool(envs.SGLANG_ERNIE55_MEGA_MOE.get())


def _tp_size() -> int:
    from sglang.srt.layers.dp_attention import get_parallel

    return get_parallel().tp_size


def _a2a_none() -> bool:
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    return get_moe_a2a_backend().is_none()


def _static_ok(moe: torch.nn.Module) -> bool:
    """The build-time conditions the fused kernel requires, independent of any
    per-forward state. Both the shared-expert sharding decision and the backend's
    own readiness gate on exactly this, so they never disagree."""
    cfg = moe.config
    return bool(
        mega_moe_enabled()
        and envs.SGLANG_ERNIE55_MEGA_MOE_QUANT.get() in _QUANT_MODES
        and _tp_size() == _WORLD
        and _a2a_none()
        and (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.n_group, cfg.topk_group)
        == (_NUM_EXPERTS, 10, 1, 1)
        and getattr(cfg, "hidden_act", None) == "situ"
    )


def shared_expert_sharded(moe: torch.nn.Module) -> bool:
    """Should the shared expert be built TP-sharded (each rank a PARTIAL sum)?

    Sharded only when the fused kernel is on and eligible, so its combining
    AllReduce completes the shared sum for free. Every other reader of the shared
    output goes through :func:`complete_shared_output`, which AllReduces the partial
    back to a full tensor on the paths the kernel does not run (prefill, ineligible
    shapes). Call it in ``Ernie5_5Moe.__init__`` before building ``shared_experts``.
    """
    return _static_ok(moe)


def complete_shared_output(
    moe: torch.nn.Module, shared_output: torch.Tensor
) -> torch.Tensor:
    """The shared-expert output as a COMPLETE tensor: AllReduced when it is TP-sharded."""
    if getattr(moe, "_shared_sharded", False):
        return tensor_model_parallel_all_reduce(shared_output)
    return shared_output


def _scale_fp32(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 128x128 block scales ``[..., N/128, K/128]`` of a blockwise-FP8 weight.

    On Blackwell the loader stores them UE8M0-packed (4 exponents per int32, each
    128-row block repeated per row); unpack and drop the K padding. The values are
    powers of two, so applying them after the unscaled MMA is exact.
    """
    if scale.dtype == torch.int32:
        from sglang.srt.layers.quantization.fp8_utils import (
            _inverse_transform_scale_ue8m0_impl,
        )

        scale = _inverse_transform_scale_ue8m0_impl(scale)
        scale = scale[..., : -(-weight.shape[-1] // 128)]
    return scale.to(torch.float32).contiguous()


def _blockwise_scale_fp32(layer: torch.nn.Module) -> torch.Tensor:
    """The fp32 128x128 block scales of a blockwise-FP8 linear ``[N/128, K/128]``."""
    scale = getattr(layer, "weight_scale_inv", None)
    if scale is None:
        raise RuntimeError(
            "the fused MoE kernel needs a blockwise-FP8 layer (weight_scale_inv); "
            f"got {layer.weight.dtype} with no block scales"
        )
    return _scale_fp32(layer.weight, scale)


def _ue8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    """UE8M0 bytes (the biased exponent) of power-of-two fp32 scales."""
    bits = scale.to(torch.float32).contiguous().view(torch.int32)
    if (bits & 0x7FFFFF).any():
        raise ValueError("an MX scale table can only be built from power-of-two scales")
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def _to_sf_atom(plain: torch.Tensor) -> torch.Tensor:
    """Plain ``[..., rows, cols]`` UE8M0 bytes -> the CUTLASS SfAtom layout the kernel
    reads (same shape): one 512-B atom per 128 rows x 4 K/32 groups, atoms row-major."""
    rows, cols = plain.shape[-2], plain.shape[-1]
    assert rows % _SF_ATOM_ROWS == 0 and cols % _SF_ATOM_COLS == 0, plain.shape
    m = torch.arange(rows, device=plain.device)[:, None]
    g = torch.arange(cols, device=plain.device)[None, :]
    offsets = (
        (m // _SF_ATOM_ROWS) * (cols // _SF_ATOM_COLS) * _SF_ATOM_BYTES
        + (g // _SF_ATOM_COLS) * _SF_ATOM_BYTES
        + (m % 32) * 16
        + ((m % _SF_ATOM_ROWS) // 32) * _SF_ATOM_COLS
        + (g % _SF_ATOM_COLS)
    ).reshape(-1)
    out = torch.empty_like(plain.contiguous())
    out.reshape(-1, rows * cols)[:, offsets] = plain.reshape(-1, rows * cols)
    return out


def _mx_scale_from_block(block_scale: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-two 128x128 block scales ``[..., N/128, K/128]`` -> the MXFP8
    SfAtom table ``[..., N, K/32]`` (every 1x32 group takes its block's scale)."""
    return _to_sf_atom(
        _ue8m0_bytes(block_scale.repeat_interleave(128, -2).repeat_interleave(4, -1))
    )


def _mx_scale_from_group32(scale: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-two 1x32 scales ``[..., N, K/32]`` (the FP4 quantiser's) -> the
    SfAtom table. Unlike the block scale, every group already has its own exponent."""
    return _to_sf_atom(_ue8m0_bytes(scale))


def _group_wise_quant(weight: torch.Tensor):
    """BF16 ``[M, K]`` -> ``(FP8 weight, row-major UE8M0 scale [M, K/32])`` with HALOps'
    native 1x32-group quantiser (one amax per group, not derived from a block)."""
    from sglang.srt.layers.quantization.halops_mxfp8 import (
        HALOPS_MODE,
        load_halops_gemm,
    )

    _, group_wise_quant = load_halops_gemm()
    weight = weight.to(torch.bfloat16).cuda().contiguous()
    fp8_weight, sf_rowmajor = group_wise_quant(
        weight, mode=HALOPS_MODE, use_swizzle=False
    )
    return fp8_weight, sf_rowmajor.view(torch.uint8)


def _quant_group32(weight: torch.Tensor):
    """BF16 ``[N, K]`` linear weight -> ``(FP8 weight, SfAtom group32 scale [N, K/32])``."""
    fp8_weight, sf_rowmajor = _group_wise_quant(weight)
    return fp8_weight, _to_sf_atom(sf_rowmajor)


def _quant_group32_experts(stack: torch.Tensor):
    """BF16 experts ``[E, N, K]`` -> ``(FP8 [E, N, K], SfAtom scale [E, N, K/32])`` in a
    single quant pass: each 1x32 group's amax is row-independent, so flattening the
    experts into rows is identical to quantising them one by one."""
    e, n, k = stack.shape
    fp8_weight, sf_rowmajor = _group_wise_quant(stack.reshape(e * n, k))
    return fp8_weight.view(e, n, k), _to_sf_atom(sf_rowmajor.view(e, n, k // 32))


def _get_workspace(wide: bool, group, device: torch.device, hidden_size: int):
    if (device, wide) not in _WORKSPACES:
        from halops import get_workspace_for_moe

        probe = torch.empty(
            _MAX_M if wide else 1, hidden_size, dtype=torch.bfloat16, device=device
        )
        _WORKSPACES[(device, wide)] = get_workspace_for_moe(
            probe, group.cpu_group.group_name, group.rank_in_group, group.world_size
        )
    return _WORKSPACES[(device, wide)]


class Ernie55MegaMoE:
    """Per-``Ernie5_5Moe`` helper: eligibility, weight preparation, and the launch.

    Build it at the END of ``Ernie5_5Moe.__init__``: it reads the layer's modules
    and rendezvouses the kernel's symmetric workspace (collective, once per process)
    at model build. Any failure leaves the layer on its normal MoE path.
    """

    def __init__(self, moe: torch.nn.Module):
        self.moe = moe
        self.only = False  # MEGA_MOE_ONLY took effect: no other expert weights kept
        self.ready = False  # static eligibility + kernel/comm available
        self.prepared = False  # weights converted
        self.quant_mode = envs.SGLANG_ERNIE55_MEGA_MOE_QUANT.get()
        self._true_group32 = bool(envs.SGLANG_ERNIE55_MEGA_MOE_TRUE_GROUP32.get())
        # The routed experts' second copy is ~69 GB/rank, so it is a separate opt-in;
        # the tiny latent/shared operands ride on _true_group32 alone.
        self._true_group32_experts = self._true_group32 and bool(
            envs.SGLANG_ERNIE55_MEGA_MOE_TRUE_GROUP32_EXPERTS.get()
        )
        self._g32: dict = {}  # native group32 (FP8 weight, SfAtom scale) per operand
        self._experts_ready = False
        if not mega_moe_enabled():
            return
        reason = self._static_reason()
        if reason is not None:
            logger.info("ERNIE5.5 mega MoE: layer %s disabled (%s)", moe.layer_id, reason)
            return
        try:
            import halops  # noqa: F401
            from sglang.srt.distributed.parallel_state import get_tp_group

            device = torch.device("cuda", torch.cuda.current_device())
            group = get_tp_group()
            if group.world_size != _WORLD:
                raise RuntimeError(f"needs tp_size == {_WORLD}, got {group.world_size}")
            self._group = group
            for w in (False, True):
                _get_workspace(w, group, device, moe.config.hidden_size)
        except Exception as e:  # noqa: BLE001 - any failure keeps the normal path
            logger.warning(
                "ERNIE5.5 mega MoE: layer %s disabled (%s: %s)",
                moe.layer_id,
                type(e).__name__,
                e,
            )
            return
        if "ignored-params" not in _logged:
            _logged.add("ignored-params")  # once per process, not per layer
            logger.warning(
                "ERNIE5.5 mega MoE ignores e_score_correction_bias and per_expert_scale"
            )
        self.ready = True
        self.only = bool(envs.SGLANG_ERNIE55_MEGA_MOE_ONLY.get())
        self._install_hooks()

    def _static_reason(self) -> Optional[str]:
        moe, cfg = self.moe, self.moe.config
        if self.quant_mode not in _QUANT_MODES:
            return (
                f"SGLANG_ERNIE55_MEGA_MOE_QUANT must be one of {_QUANT_MODES}, "
                f"got {self.quant_mode!r}"
            )
        if _tp_size() != _WORLD or not _a2a_none():
            return "needs pure TP with tp_size == 4 (no all-to-all MoE)"
        if not getattr(moe, "_shared_sharded", False):
            return "needs the TP-sharded shared expert"
        if (
            cfg.n_routed_experts,
            cfg.num_experts_per_tok,
            cfg.n_group,
            cfg.topk_group,
        ) != (_NUM_EXPERTS, 10, 1, 1):
            return "needs 512 experts, top-10, one expert group"
        if getattr(cfg, "hidden_act", None) != "situ":
            return "needs the SiTU activation"
        return None

    # ---- weight preparation --------------------------------------------------

    def _install_hooks(self) -> None:
        """Convert the layer's weights for the kernel once the loader has processed
        them (and again on every reload). The routed experts and the dense modules
        (fc1/fc2/shared) are read as their FP8 primary weight; the only new tensors
        are the gate/up pair-interleaved ``w13`` copy and the packed router/shared-up
        operands, all built on device in :meth:`try_prepare`."""
        moe = self.moe
        shared = moe.shared_experts
        # The true-group32 quantiser captures each operand (fc2 excepted) from BF16,
        # before the loader's own hook overwrites it with block128 FP8.
        owners = [
            (moe.experts, "experts"),
            (moe.fc1_latent_proj, "fc1"),
            (moe.fc2_latent_proj, "fc2"),
            (shared.gate_up_proj, "shared_up"),
            (shared.down_proj, "shared_dn"),
        ]
        mx = self.quant_mode != "w8a8-blockwise"
        for owner, key in owners:
            qm = getattr(owner, "quant_method", None)
            if qm is None or getattr(qm, "_ernie55_mega_wrapped", False):
                continue
            orig = qm.process_weights_after_loading
            is_experts = owner is moe.experts
            # fc2 stays block128. The routed experts are captured when: (a) the recipe
            # is w4a8-mxfp4 (the FP4 copy IS the recipe, mandatory), or (b) the memory-
            # heavy native-group32 opt-in is set. The tiny dense modules ride on
            # _true_group32 alone.
            fp4_experts = is_experts and self.quant_mode == "w4a8-mxfp4"
            capture = mx and key != "fc2" and (
                fp4_experts
                or (self._true_group32_experts if is_experts else self._true_group32)
            )

            def wrapped(
                layer,
                *a,
                _orig=orig,
                _is_experts=is_experts,
                _key=key,
                _capture=capture,
                **k,
            ):
                if _capture:
                    self._capture_group32(_key, layer)
                out = _orig(layer, *a, **k)
                if _is_experts:
                    self._reuse_experts()
                self.try_prepare()
                return out

            qm.process_weights_after_loading = wrapped
            qm._ernie55_mega_wrapped = True

    def _capture_group32(self, key: str, layer: torch.nn.Module) -> None:
        """Quantize this operand's BF16 weight to native 1x32-group FP8 and stash it for
        the kernel; the loader then still writes its block128 FP8 for the prefill
        fallback. No-op when the weight is not BF16 (a pre-serialized FP8 checkpoint),
        which leaves the SfAtom-from-block128 scale in place."""
        if key == "experts":
            w13 = getattr(layer, "w13_weight", None)
            w2 = getattr(layer, "w2_weight", None)
            if w13 is None or w2 is None or w13.dtype != torch.bfloat16:
                return
            if tuple(w13.shape) != (_NUM_EXPERTS, 1024, 2048) or tuple(w2.shape) != (
                _NUM_EXPERTS,
                2048,
                512,
            ):
                return  # the block128 path logs the shape mismatch and disables
            if self.quant_mode == "w4a8-mxfp4":
                # ONE packed FP4 (E2M1) copy + 1x32 SfAtom scales; the loader still
                # writes block128 FP8 for the triton prefill fallback. _set_experts
                # consumes the uint8 FP4 weight with sfatom=True (scales pre-built).
                from sglang.srt.layers.quantization.ernie55_blockwise_fp8 import (
                    quant_experts_fp4,
                )

                w13d, s13 = quant_experts_fp4(w13.data)
                w2d, s2 = quant_experts_fp4(w2.data)
                s13, s2 = _mx_scale_from_group32(s13), _mx_scale_from_group32(s2)
            else:
                w13d, s13 = _quant_group32_experts(w13.data)
                w2d, s2 = _quant_group32_experts(w2.data)
            self._g32["experts"] = (w13d, s13, w2d, s2)
        else:
            wt = getattr(layer, "weight", None)
            if wt is None or wt.dtype != torch.bfloat16:
                return
            self._g32[key] = _quant_group32(wt.data)

    def _reuse_experts(self) -> None:
        """Use the model's own FP8 128x128 routed experts: ``w2`` and the scales as
        they are, with only ``w13`` pair-interleaved into the gate/up copy the kernel
        reads (built now, not inside a graph capture)."""
        g = self._g32.get("experts")
        if g is not None:  # native group32 FP8 weights + SfAtom scales
            w13d, s13, w2d, s2 = g
            self._set_experts(w13d, s13, w2d, s2, sfatom=True)
            self._experts_ready = True
            return
        ex = self.moe.experts
        w13, w2 = getattr(ex, "w13_weight", None), getattr(ex, "w2_weight", None)
        s13p = getattr(ex, "w13_weight_scale_inv", None)
        s2p = getattr(ex, "w2_weight_scale_inv", None)
        if w13 is None or w2 is None or s13p is None or s2p is None:
            return
        if w13.data.dtype != _FP8 or w2.data.dtype != _FP8:
            return  # not FP8 yet (or an unexpected format)
        E = _NUM_EXPERTS
        if tuple(w13.shape) != (E, 1024, 2048) or tuple(w2.shape) != (E, 2048, 512):
            logger.warning(
                "ERNIE5.5 mega MoE: layer %s expert shapes %s/%s do not fit the "
                "kernel; disabled",
                self.moe.layer_id,
                tuple(w13.shape),
                tuple(w2.shape),
            )
            self.ready = self.only = False
            return
        self._set_experts(
            w13.data,
            _scale_fp32(w13.data, s13p.data),
            w2.data,
            _scale_fp32(w2.data, s2p.data),
        )
        self._experts_ready = True

    def _set_experts(self, raw13, s13, w2, s2, sfatom=False) -> None:
        """Keep the kernel's routed-expert operands: the gate/up PAIR-packed ``w13``,
        ``w2`` as it is, and the scales. For the MX recipe ``s13``/``s2`` are the SfAtom
        UE8M0 tables when ``sfatom`` (native group32), else fp32 power-of-two 128x128
        block scales turned into SfAtom tables here."""
        from halops import transform_weights_for_moe

        mx = self.quant_mode != "w8a8-blockwise"
        if mx and not sfatom:
            s13, s2 = _mx_scale_from_block(s13), _mx_scale_from_block(s2)
        packed = transform_weights_for_moe(
            None, raw13, None, expert_up_weights_scale=s13 if mx else None
        )
        self._w13 = packed["expert_up_weights"]
        self._s13 = packed["expert_up_weights_scale"] if mx else s13
        self._w2, self._s2 = w2, s2

    def _mx_linear(self, key: str, layer: torch.nn.Module):
        """A mega linear operand for the MX recipe: ``(FP8 weight, SfAtom group32 scale)``
        -- the captured native group32 quant when present, else the model's block128 FP8
        weight with the SfAtom scale derived from its block scale."""
        g = self._g32.get(key)
        if g is not None:
            return g
        w = layer.weight.data
        return w, _mx_scale_from_block(_scale_fp32(w, layer.weight_scale_inv.data))

    def try_prepare(self) -> None:
        """Build the dense/router/shared kernel operands once every module is FP8.
        No-op until the experts and the dense modules have been quantised."""
        moe = self.moe
        fc1, fc2 = moe.fc1_latent_proj, moe.fc2_latent_proj
        gu, dn = moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj
        if not self._experts_ready:
            return
        for owner in (fc1, fc2, gu, dn):
            wt = getattr(owner, "weight", None)
            if (
                wt is None
                or wt.dtype != _FP8
                or getattr(owner, "weight_scale_inv", None) is None
            ):
                return  # a dense module is not quantised yet
        from halops import transform_weights_for_moe

        E = _NUM_EXPERTS
        mx = self.quant_mode != "w8a8-blockwise"
        # Shared expert (TP shard): gate_up [2*512, H] gate|up, down [H, 512]; the up
        # weight takes the gate/up pair layout. Latent fc1 [L, H] / fc2 [H, L] stay.
        if mx:
            self._lat_w, self._lat_s = self._mx_linear("fc1", fc1)
            sh_up_raw, sh_up_sf = self._mx_linear("shared_up", gu)
            self._sh_dn, self._sh_dn_s = self._mx_linear("shared_dn", dn)
            packed = transform_weights_for_moe(
                None, None, sh_up_raw, shared_up_weight_scale=sh_up_sf
            )
            self._sh_up = packed["shared_up_weight"]
            self._sh_up_s = packed["shared_up_weight_scale"]
        else:
            sh_up_raw = gu.weight.data
            sh_up_s = _scale_fp32(sh_up_raw, gu.weight_scale_inv.data)
            packed = transform_weights_for_moe(
                None, None, sh_up_raw, shared_up_weight_scale=None
            )
            self._lat_w = fc1.weight.data
            self._lat_s = _blockwise_scale_fp32(fc1)
            self._sh_up = packed["shared_up_weight"]
            self._sh_up_s = sh_up_s
            self._sh_dn = dn.weight.data
            self._sh_dn_s = _scale_fp32(self._sh_dn, dn.weight_scale_inv.data)
        # fc2 stays block128 in both recipes (its kernel is not on group32 yet).
        self._fc2_w = fc2.weight.data
        self._fc2_s = _blockwise_scale_fp32(fc2)
        # Router: raw gate weight [2E, H] (the two gates) -> [2, H, E], packed per
        # kernel variant (narrow m <= 8, wide m <= 16).
        w3 = (
            moe.gate_weight.data.to(torch.bfloat16)
            .view(2, E, moe.config.hidden_size)
            .transpose(1, 2)
            .contiguous()
        )
        self._router = {
            wide: transform_weights_for_moe(
                w3, None, None, num_tokens=_MAX_M if wide else 1
            )["router_weight"]
            for wide in (False, True)
        }
        cfg = moe.config
        self._situ = (
            float(cfg.activation_situ_beta),
            float(cfg.activation_situ_linear_beta),
        )
        self.prepared = True

    # ---- gate + launch -------------------------------------------------------

    def skip_reason(self, num_tokens: int) -> Optional[str]:
        if not mega_moe_enabled():
            return "switch-off"
        if not self.ready:
            return "kernel-unavailable"
        if not self.prepared:
            return "weights-not-prepared"
        if not 1 <= num_tokens <= _MAX_M:
            return "m-out-of-range"
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch

        if get_is_extend_in_batch():
            return "not-decode"
        return None

    def should_run(self, num_tokens: int) -> bool:
        reason = self.skip_reason(num_tokens)
        if reason is not None and self.only:
            raise RuntimeError(
                f"ERNIE5.5 mega MoE (MEGA_MOE_ONLY): layer {self.moe.layer_id} cannot "
                f"run the fused kernel for num_tokens={num_tokens} ({reason}); its "
                "other expert weights were not kept, so there is no fallback"
            )
        key = (reason, num_tokens)
        if key not in _logged and len(_logged) < 64:
            _logged.add(key)
            logger.info(
                "ERNIE5.5 mega MoE: %s (m=%d, layer_id=%s, quant=%s)",
                "taken" if reason is None else f"not taken ({reason})",
                num_tokens,
                self.moe.layer_id,
                self.quant_mode,
            )
        return reason is None

    def run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The whole MoE forward of the layer (router .. tail) in one launch; returns
        ``[m, H]`` bf16, already including the shared expert and the cross-rank
        AllReduce + RMSNorm + fc2 tail."""
        from halops import low_latency_moe

        moe = self.moe
        m, hidden = hidden_states.shape
        x = hidden_states.contiguous()
        device = x.device
        wide = m > _NARROW_M
        out = torch.empty(m, hidden, dtype=torch.bfloat16, device=device)
        low_latency_moe(
            x,
            self._router[wide],
            self._w13,
            self._s13,
            self._w2,
            self._s2,
            self._sh_up,
            self._sh_up_s,
            self._sh_dn,
            self._sh_dn_s,
            self._lat_w,  # latent_down_weight (fc1_latent_proj)
            self._lat_s,
            self._fc2_w,  # latent_up_weight (fc2_latent_proj, the tail)
            self._fc2_s,
            moe.latent_norm.weight,  # norm_weight
            out,
            _WORKSPACES[(device, wide)],
            moe.topk.topk_config.top_k,
            scoring_func="sigmoid",
            renormalize=bool(moe.topk.topk_config.renormalize),
            activation="situ",
            situ_beta=self._situ[0],
            situ_linear_beta=self._situ[1],
            eps=float(moe.config.rms_norm_eps),
            quant_mode=self.quant_mode,
        )
        return out






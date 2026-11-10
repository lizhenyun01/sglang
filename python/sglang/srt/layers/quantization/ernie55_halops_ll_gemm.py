"""Route eligible BF16 dense weights to HALOps' group-32 LL kernel.

Open-tree port of the internal ``halops_ll_gemm_patch`` (ERNIE5.5 single-batch
LL dense-GEMM backend). Behaviour is bitwise the internal one; only the four
internal-only imports are swapped for their open equivalents:

  SGLANG_INTERNAL_EB5_SINGLE_BATCH_LL -> envs.SGLANG_ERNIE55_SINGLE_BATCH_LL
  ll_threshold.LL_MAX_TOKENS          -> the module constant below (16)
  utils.common.is_sm100_supported     -> _is_sm100() (current_arch().is_sm(10))
  plugins.hook_registry               -> same path in open

Gated behind ``SGLANG_ERNIE55_SINGLE_BATCH_LL`` (default off): with the flag off
``register_ernie55_halops_ll_gemm_patch`` installs nothing and the validated
W4A8 / block-quant decode path is untouched.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, NamedTuple, Optional

import torch

from sglang.kernels.arch import current_arch
from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)

# One-shot dedup for the non-BF16 dense-weight warning (many layers would
# otherwise each log it at load).
_NON_BF16_WARNED = False


def _set_non_bf16_warned() -> None:
    global _NON_BF16_WARNED
    _NON_BF16_WARNED = True

# The HALOps low_latency_mma_gemm M ceiling (internal ll_threshold.LL_MAX_TOKENS).
LL_MAX_TOKENS = 16

_PLAN_ATTR = "_ll_gemm_plan"
_WEIGHT_ATTR = "_ll_gemm_weight_g32"
_SCALE_ATTR = "_ll_gemm_sfb_g32"
_UNSET = object()
_HALOPS = _UNSET
_REGISTERED = False
_MOE_PREFIX_MARKERS = (".experts",)
LL_GEMM_SM_COUNT_ATTR = "_ll_gemm_sm_count"
# HALOps low_latency_mma_gemm launcher preconditions on an [N, K] weight.
LL_GEMM_N_ALIGN = 8
LL_GEMM_K_ALIGN = 256
# The fused (dual-output / concat-K) callers pass each activation scale unpadded,
# [M, K / 32], so every fused K must keep scale rows on 16 bytes; concat-K K1 is
# also a launcher precondition.
LL_GEMM_FUSED_K_ALIGN = 512
# The kernel copies each scale row whole, on a 16-byte pitch.
_SCALE_ROW_ALIGN = 16


@functools.cache
def _is_sm100() -> bool:
    return current_arch().is_sm(10)


def _single_batch_ll() -> bool:
    """The ``SGLANG_ERNIE55_SINGLE_BATCH_LL`` switch, read at call time so the
    flag is honoured regardless of import order (open EnvBool is set at start)."""
    return envs.SGLANG_ERNIE55_SINGLE_BATCH_LL.get()


class HALOpsLLOps(NamedTuple):
    gemm: Callable
    quant: Callable


class _Plan(NamedTuple):
    weight: torch.Tensor
    sfb: torch.Tensor
    n: int


def load_halops() -> Optional[HALOpsLLOps]:
    """HALOps low_latency_mma_gemm and group_wise_quant, imported once; None when
    HALOps is not installed."""
    global _HALOPS
    if _HALOPS is _UNSET:
        try:
            from halops import group_wise_quant, low_latency_mma_gemm
        except Exception as exc:  # pragma: no cover - deployment dependent
            logger.warning("HALOps LL GEMM is unavailable (%s)", exc)
            _HALOPS = None
        else:
            _HALOPS = HALOpsLLOps(low_latency_mma_gemm, group_wise_quant)
    return _HALOPS


def ll_gemm_shape_ok(n: int, k: int, *, n2: int = 0, k2: int = 0) -> bool:
    """The one shape gate every LL GEMM caller uses, on the [n, k] weight:
    N % LL_GEMM_N_ALIGN == 0 and K % LL_GEMM_K_ALIGN == 0. A dual output adds an
    [n2, k] weight2 (``n2``), a concat-K an [n, k2] weight2 (``k2``); both fused
    forms need each K on LL_GEMM_FUSED_K_ALIGN."""
    if n <= 0 or n % LL_GEMM_N_ALIGN or k <= 0 or k % LL_GEMM_K_ALIGN:
        return False
    if not (n2 or k2):
        return True
    if (n2 and k2) or k % LL_GEMM_FUSED_K_ALIGN:
        return False
    if n2:
        return n2 > 0 and n2 % LL_GEMM_N_ALIGN == 0
    return k2 > 0 and k2 % LL_GEMM_FUSED_K_ALIGN == 0


def _scale_group(x: torch.Tensor, scale: torch.Tensor) -> int:
    """Quant group along K, read off the operand and its scale the way flashinfer
    does (``sf_vec_size = x.shape[-1] // scale.shape[-1]``) instead of assuming
    the w8a8-mxfp8 group of group_wise_quant."""
    return int(x.shape[-1]) // int(scale.shape[-1])


def _quant_group32(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 + uint8 UE8M0 scales in the plain row-major layout the LL kernel reads."""
    with torch.no_grad():
        return load_halops().quant(x.contiguous(), mode="w8a8-mxfp8", use_swizzle=False)


def _align_scale_rows(sf: torch.Tensor) -> torch.Tensor:
    """``sf`` ([rows, K/32] uint8) itself when its rows already sit on a 16-byte
    pitch (K % 512 == 0), else its [rows, K/32] view in a padded copy."""
    if (
        sf.stride(1) == 1
        and sf.stride(0) % _SCALE_ROW_ALIGN == 0
        and sf.data_ptr() % _SCALE_ROW_ALIGN == 0
    ):
        return sf
    rows, cols = sf.shape
    pitch = -(-cols // _SCALE_ROW_ALIGN) * _SCALE_ROW_ALIGN
    padded = sf.new_empty((rows, pitch))[:, :cols]
    padded.copy_(sf)
    return padded


def g32_side_copy(
    layer: torch.nn.Module,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """The (fp8 weight, uint8 sfb) group-32 copy cached at load, or None."""
    weight = getattr(layer, _WEIGHT_ATTR, None)
    sfb = getattr(layer, _SCALE_ATTR, None)
    if weight is None or sfb is None:
        return None
    return weight, sfb


def set_g32_side_copy(
    layer: torch.nn.Module, weight: torch.Tensor, sfb: torch.Tensor
) -> None:
    """Replace the cached group-32 copy (e.g. with row views of a stacked buffer)
    and drop the cached plan so the next call rebuilds it from the new copy."""
    setattr(layer, _WEIGHT_ATTR, weight)
    setattr(layer, _SCALE_ATTR, sfb)
    setattr(layer, _PLAN_ATTR, None)


def _is_moe_expert(layer: torch.nn.Module) -> bool:
    prefix = getattr(layer, "prefix", None)
    if not isinstance(prefix, str):
        return False
    return any(marker in prefix.lower() for marker in _MOE_PREFIX_MARKERS)


def _shape_reason(
    method, layer: torch.nn.Module, weight: Optional[torch.Tensor]
) -> Optional[str]:
    if not method.block_quant:
        return "not block-quantized"
    if method.use_marlin or method.use_mxfp8:
        return "incompatible FP8 backend"
    if weight is None or weight.dim() != 2:
        return "weight is not 2D"
    if _is_moe_expert(layer):
        return "routed MoE expert path"
    n, k = (int(v) for v in weight.shape)
    if not ll_gemm_shape_ok(n, k):
        return f"N={n}, K={k} fails the LL GEMM shape preconditions"
    return None


def _process_group32_weights(original_fn, self, layer):
    """Cache a group-32 copy of the BF16 weight, then run the native group-128
    quantization.

    The group-32 pass must come first: original_fn replaces the BF16 weight with
    its group-128 FP8 copy."""
    if _single_batch_ll() and _shape_reason(self, layer, layer.weight) is None:
        if layer.weight.dtype != torch.bfloat16:
            # Open's ``--quantization fp8`` serializes dense weights to FP8
            # before this hook runs, so there is no BF16 source to build the
            # group-32 side copy from. Internal runs ``blockwise_fp8`` where
            # dense stays BF16 into this hook. Degrade instead of crashing:
            # skip the side copy, the layer keeps its native path, and the LL
            # GEMM simply never engages for it (g32_prequant_ready stays False).
            if not _NON_BF16_WARNED:
                logger.warning(
                    "ERNIE5.5 single-batch LL: skipping group-32 side copy for "
                    "a non-BF16 (%s) dense weight; such layers stay on their "
                    "native path. Run with BF16-at-load dense (blockwise_fp8) "
                    "to engage the HALOps LL GEMM. (warned once)",
                    layer.weight.dtype,
                )
                _set_non_bf16_warned()
            return original_fn(self, layer)
        weight, sfb = _quant_group32(layer.weight.data)
        # Aligned here, once: the kernel reads sfb before its PDL wait, so a
        # copy kernel right ahead of it in the forward would race.
        set_g32_side_copy(layer, weight, _align_scale_rows(sfb))
    return original_fn(self, layer)


def _layer_name(layer: torch.nn.Module) -> str:
    prefix = getattr(layer, "prefix", None)
    return prefix if isinstance(prefix, str) else type(layer).__name__


def _prepare_layer(method, layer: torch.nn.Module) -> tuple[Optional[str], object]:
    """(reason, plan), cached on the layer: plan is a _Plan when reason is None."""
    cached = getattr(layer, _PLAN_ATTR, None)
    if cached is not None:
        return cached

    copy = g32_side_copy(layer)
    weight = copy[0] if copy is not None else layer.weight
    reason = _shape_reason(method, layer, weight)
    if reason is None and not _is_sm100():
        reason = "device is not Blackwell"
    if reason is None and (
        copy is None
        or copy[0].dtype != torch.float8_e4m3fn
        or copy[1].dtype != torch.uint8
    ):
        reason = "group-32 weight quantization was not prepared"
    plan = None if reason is not None else _Plan(*copy, int(copy[0].shape[0]))
    setattr(layer, _PLAN_ATTR, (reason, plan))
    if weight is not None and weight.dim() == 2:
        n, k = (int(v) for v in weight.shape)
        logger.info(
            "LL GEMM group32: %-34s N=%-6d K=%-6d %s",
            _layer_name(layer),
            n,
            k,
            "USE_GROUP32_LL" if plan else f"keep-native ({reason})",
        )
    return reason, plan


def _run_plan(
    plan: _Plan, q: torch.Tensor, sfa: torch.Tensor, sm_count: int
) -> torch.Tensor:
    output = torch.empty((q.shape[0], plan.n), dtype=torch.bfloat16, device=q.device)
    load_halops().gemm(q, plan.weight, sfa, plan.sfb, output, sm_count=sm_count)
    return output


def _stacked_ll_usable(n: int, k: int) -> bool:
    return (
        _single_batch_ll()
        and _is_sm100()
        and load_halops() is not None
        and ll_gemm_shape_ok(n, k)
    )


def _stacked_plan(q_weight: torch.Tensor, sfb: torch.Tensor) -> _Plan:
    n, k = (int(v) for v in q_weight.shape)
    logger.info("LL GEMM group32: stacked N=%d K=%d", n, k)
    return _Plan(q_weight, _align_scale_rows(sfb), n)


def prepare_stacked_group32(weight_bf16: torch.Tensor) -> Optional[_Plan]:
    """Group-32 LL plan for a stacked BF16 weight that bypasses
    Fp8LinearMethod.apply (DSpark CommitKvProj). None when LL is unusable."""
    if weight_bf16.dim() != 2 or weight_bf16.dtype != torch.bfloat16:
        return None
    if not _stacked_ll_usable(*weight_bf16.shape):
        return None
    return _stacked_plan(*_quant_group32(weight_bf16))


def stack_group32_side_copies(linears: list) -> Optional[_Plan]:
    """Stack the per-linear group-32 side copies cached at load along N; None
    when any linear lacks one or LL is unusable."""
    copies = [g32_side_copy(linear) for linear in linears]
    if any(copy is None for copy in copies):
        return None
    q_weight = torch.cat([w for w, _ in copies], dim=0)
    if not _stacked_ll_usable(*q_weight.shape):
        return None
    return _stacked_plan(q_weight, torch.cat([s for _, s in copies], dim=0))


def stacked_group32_gemm(
    plan: Optional[_Plan], x: torch.Tensor
) -> Optional[torch.Tensor]:
    """x [M, K] bf16 @ plan.weight.T on LL when 1 <= M <= the LL ceiling;
    None tells the caller to keep its native path."""
    if plan is None or x.dtype != torch.bfloat16 or x.dim() != 2:
        return None
    if not 1 <= int(x.shape[0]) <= LL_MAX_TOKENS:
        return None
    return _run_plan(plan, *_quant_group32(x), sm_count=-1)


def g32_prequant_ready(layer: torch.nn.Module, m: int) -> bool:
    """Whether a fused producer should hand `layer` a group-32 (fp8, uint8
    [m, K/32]) tuple: 1 <= m <= the LL ceiling and the layer has a group-32 LL
    plan. The tuple has no native fallback (DeepGEMM reads group-128 scales
    only), so a producer emits group 32 exactly when this holds."""
    if not _single_batch_ll() or not 1 <= m <= LL_MAX_TOKENS:
        return False
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    method = layer.quant_method
    if not isinstance(method, Fp8LinearMethod):
        return False
    return _prepare_layer(method, layer)[1] is not None


def _activation_shape(x) -> tuple[int, int]:
    """(M, K) of a bf16 input or of the fp8 half of a pre-quantized tuple."""
    a = x[0] if isinstance(x, tuple) else x
    return int(a.numel() // a.shape[-1]), int(a.shape[-1])


def _ll_gemm(method, layer, x, bias) -> torch.Tensor:
    """Fp8LinearMethod.apply on the LL kernel, or raise -- never the DeepGEMM
    fallback. A bf16 input is quantized to group 32 here; an (fp8, uint8) tuple
    from an upstream fused quant is used as is, its scale rows aligned if needed."""
    m, k = _activation_shape(x)

    def error(why: str) -> RuntimeError:
        n = int(layer.weight.shape[-2])
        return RuntimeError(f"LL GEMM: {_layer_name(layer)} N={n} K={k} M={m}: {why}")

    reason, plan = _prepare_layer(method, layer)
    if plan is None:
        raise error(f"no group-32 plan ({reason})")
    if isinstance(x, tuple):
        q, sfa = x[0].reshape(m, k), x[1].reshape(m, -1)
        if q.dtype != torch.float8_e4m3fn:
            raise error(f"tuple input {q.dtype}/{sfa.dtype} is not fp8 + uint8")
    elif x.dtype == torch.bfloat16:
        q, sfa = _quant_group32(x.reshape(m, k))
    else:
        raise error(f"input dtype {x.dtype} is not bf16")
    if _scale_group(q, sfa) != _scale_group(plan.weight, plan.sfb):
        raise error(
            f"activation scale {tuple(sfa.shape)} does not match "
            "the weight's quant group"
        )
    try:
        output = _run_plan(
            plan, q, _align_scale_rows(sfa), getattr(layer, LL_GEMM_SM_COUNT_ATTR, -1)
        )
    except RuntimeError as exc:
        raise error(f"LL kernel raised: {exc}") from exc
    if bias is not None:
        output = output + bias
    lead = (x[0] if isinstance(x, tuple) else x).shape[:-1]
    return output.view(*lead, plan.n)


def _apply_with_ll_gemm(original_fn, self, layer, x, bias=None):
    if not self.block_quant or self.use_marlin or self.use_mxfp8:
        return original_fn(self, layer, x, bias)
    if isinstance(x, tuple):
        # A group-32 tuple has no native fallback (DeepGEMM reads group 128);
        # any other pre-quantized tuple is DeepGEMM's own layout.
        if x[1].dtype == torch.uint8:
            return _ll_gemm(self, layer, x, bias)
        return original_fn(self, layer, x, bias)
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    # Decode-sized M in capture runs LL or raises; above the LL ceiling (bs > 1
    # verify) and outside capture the native DeepGEMM path runs.
    if (
        get_is_capture_mode()
        and not _is_moe_expert(layer)
        and 1 <= _activation_shape(x)[0] <= LL_MAX_TOKENS
    ):
        return _ll_gemm(self, layer, x, bias)
    return original_fn(self, layer, x, bias)


def register_ernie55_halops_ll_gemm_patch() -> None:
    """Install the group-32 LL GEMM AROUND hooks on Fp8LinearMethod. No-op unless
    SGLANG_ERNIE55_SINGLE_BATCH_LL is set and HALOps is importable; the once-guard
    keeps a double call (e.g. two ERNIE5.5 model builds) to a single install."""
    global _REGISTERED
    if _REGISTERED or not _single_batch_ll() or load_halops() is None:
        return
    _REGISTERED = True
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading_block_quant",
        _process_group32_weights,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.apply",
        _apply_with_ll_gemm,
        HookType.AROUND,
    )
    logger.info(
        "LL GEMM: eligible dense layers use HALOps group-32 quantization "
        "and low_latency_mma_gemm; routed MoE paths remain native."
    )

# [Feature] Add HALOps as an opt-in MXFP8 GEMM backend for Blackwell

## Motivation

[HALOps](https://github.com/PaddlePaddle/HALOps) is the PaddlePaddle team's high-performance kernel library for NVIDIA Blackwell (SM100). This PR integrates its W8A8 MXFP8 dense GEMM and fused `group_wise_quant` activation quantizer.

SGLang already serves MXFP8 dense linears through DeepGEMM and FlashInfer. HALOps is added as a third backend, selected explicitly with `--fp8-gemm-backend halops`. Defaults are unchanged: `auto` never resolves to HALOps, and the package is not imported unless the flag is set.

The change also completes the MXFP8 path used in the benchmark below. MXFP8 experts are served by the existing DeepGEMM MegaMoE W8A8 (`fp8xfp8`) kernel, and the MoE scale layout stays canonical so MegaMoE can derive its own layout from it.

The `halops` wheel is published on PaddlePaddle's CUDA 13.2 package page rather than PyPI and ships cp312/Linux wheels only, so it is an optional extra:

```bash
pip install "sglang[halops]" \
  -f https://www.paddlepaddle.org.cn/packages/stable/cu132/halops/
```

## Modifications

- `layers/quantization/halops_mxfp8.py`: lazy `halops` loader with an install hint, SfAtom scale swizzle, and the load-time weight quantizer.
- `layers/quantization/halops_mxfp8_linear.py`: W8A8 MXFP8 dense linear runner. Shapes HALOps has no tile for, K not 128-aligned, non-bf16/fp16 activations, and non-SM100 devices fall back to FlashInfer CUTLASS using the canonical scale kept on the layer.
- `layers/quantization/fp8_utils.py`: `HALOPS` entries in the FP8/MXFP8 backend enums, resolution, and dispatch. Selecting `halops` for a block-FP8 checkpoint raises instead of silently running a different kernel.
- `layers/quantization/fp8.py`: `weight_scale_inv_halops` prepared at load, apply-path wiring, and canonical MXFP8 MoE quantization for MegaMoE.
- `layers/moe/mega_moe.py`: pick the `fp8xfp8` MMA type from E4M3 expert weights, turn UE8M0 scale bytes into fp32, and read K as un-packed (one element per byte) for that type.
- `arg_groups/choices.py`, an optional `halops` extra in `pyproject.toml`, and a documentation entry for the new backend.
- `test/registered/unit/layers/quantization/test_halops_mxfp8.py` (CPU): SfAtom layout, arguments passed to HALOps, dtype handling, and the FlashInfer fallback.

## Usage

Serving DeepSeek-V4-Flash with online MXFP8 quantization and HALOps dense GEMM on 4×GB200:

```bash
#!/bin/bash
# Attention HALOps MXFP8 + MoE MegaMoE MXFP8
set -e

export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_MHC_POST_PRE=1
export SGLANG_OPT_USE_ONLINE_COMPRESS=1
export SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1

export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096

MODEL_PATH=/path/to/DeepSeek-V4-Flash-bf16
PORT="${PORT:-30000}"

sglang serve \
  --trust-remote-code \
  --model-path "$MODEL_PATH" \
  --tp 4 \
  --dp 4 \
  --ep 4 \
  --enable-dp-attention \
  --moe-a2a-backend megamoe \
  --moe-runner-backend deep_gemm \
  --quantization mxfp8 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --nccl-port 18201 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --chunked-prefill-size 16384 \
  --mem-fraction-static 0.8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --context-length 131072 \
  --allow-auto-truncate \
  --enable-cache-report \
  --enable-metrics \
  --fp8-gemm-backend halops
```

## Accuracy Tests

DeepSeek-V4-Flash on GSM8K, full 1314 samples, 5-shot, greedy, `--max-tokens 4096`. The three arms differ only in `--fp8-gemm-backend`; MoE is MXFP8 MegaMoE in all of them.

| arm | `--fp8-gemm-backend` | gsm8k |
|---|---|---:|
| HALOps | `halops` | 0.895 |
| DeepGEMM | `deep_gemm` | 0.897 |
| FlashInfer | `flashinfer_cutlass` | 0.879 |

## Speed Tests and Profiling

4×GB200 (SM100). DeepSeek-V4-Flash, TP4 / attention-DP4 / EP4, EAGLE, `--moe-a2a-backend megamoe --moe-runner-backend deep_gemm`, `--quantization mxfp8`. Attention dense MXFP8 GEMM is the only variable: MoE is MegaMoE in all three arms and the arms differ only in `--fp8-gemm-backend`.

Dataset level (`bench_serving`, 512 random prompts, mean input 1024 / mean output 128, concurrency 128):

| backend | req/s | out tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|
| **HALOps** | **23.94** | **1576** | **592 ms** | **74.2 ms** |
| DeepGEMM | 20.23 | 1332 | 1212 ms | 81.2 ms |
| FlashInfer | 19.51 | 1285 | 865 ms | 89.2 ms |

> NOTE: The difference is on the host side, not in the kernels. On the same prefill traces all three backends execute the same 8296 kernels:

| prefill, 16384 in | HALOps | DeepGEMM | FlashInfer |
|---|---:|---:|---:|
| GPU kernel total | 421.1 ms | 417.3 ms | 424.3 ms |
| of which MXFP8 dense GEMM | 58.5 ms | 56.9 ms | 61.5 ms |
| CPU-op busy | 1528.8 ms | 1812.6 ms | 2956.7 ms |
| CUDA runtime calls | 22,955 | 17,127 | 51,213 |

GPU time is flat within ±2%, so this PR does not claim a GPU-level speedup over the other two backends. HALOps has the lowest CPU-op busy time (breakable CUDA graph not enabled); FlashInfer's host path is the heaviest.

### CPU dispatch overhead

Host dispatch cost vs kernel time at N=8192 / K=4096 on a single GPU, with the benchmark script and results: [HALOps vs DeepGEMM host-overhead benchmark](https://docs.google.com/document/d/1LsvPMVH1KmNcxfKUECui-42uOKnVay9ourd9_BZTxWw/edit).

## Checklist

- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

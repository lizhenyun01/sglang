#!/bin/bash
# Exmaple of attn infer_rock mxfp8 + moe megamoe mxfp8
set -e

unset http_proxy https_proxy

export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_MHC_POST_PRE=1
export SGLANG_OPT_USE_ONLINE_COMPRESS=1
export SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1

# MegaMoE per-rank token budget: must cover the decode cuda-graph batch
# (max_bs x num_draft_tokens) and the per-rank chunked-prefill budget.
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096

VENV=/root/paddlejob/inference-public/wanglongzhi/SGLangGithubInternal/sglang/.venv
source "$VENV/bin/activate"

MODEL_PATH="/root/paddlejob/inference-public/inference_models/DeepSeek-V4/DeepSeek-V4-Flash-bf16-1392"
PORT="${PORT:-30082}"

sglang serve \
  --trust-remote-code \
  --model-path "$MODEL_PATH" \
  --tp 4 \
  --dp 4 \
  --ep 4 \
  --disable-radix-cache \
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
  --fp8-gemm-backend inferrock \
  2>&1 | tee server_mxfp8_inferrock_megamoe.log

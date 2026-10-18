#!/usr/bin/env bash
# Local Qwen3-VL-8B server (OpenAI-compatible API) on 127.0.0.1:8001 for the pipeline, on the GPU host.
# Offline: weights come from ~/hf; vLLM usage statistics are off, so nothing leaves the host during processing.
# Start (once, shared by every pipeline process):
#   tmux new-session -d -s vllm "bash ~/forte/tools/vllm_serve.sh 2>&1 | tee -a ~/forte/logs/vllm.log"
# Ready when: curl -s 127.0.0.1:8001/v1/models
set -euo pipefail
export HF_HOME="$HOME/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_DO_NOT_TRACK=1
# The FlashInfer sampler JIT-compiles CUDA kernels and needs nvcc, which the host does not have.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec "$HOME/vllm-venv/bin/vllm" serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3vl \
  --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.65}" \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt '{"image": 32, "video": 0}'

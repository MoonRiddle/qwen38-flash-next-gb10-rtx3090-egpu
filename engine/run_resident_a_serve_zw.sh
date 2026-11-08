#!/usr/bin/env bash
set -euo pipefail
BASE="${RECIPE_HOME:-/home/user/OMP_Workspace/qwen38-gb10-3090-20261029}"
NAME=${1:-resident-a-serve-zw}
if [[ ! "$NAME" =~ ^resident-a-[A-Za-z0-9_-]{1,60}$ ]]; then
    printf 'Invalid AutoRound resident artifact basename\n' >&2
    exit 2
fi
finish_early() {
    status=$?
    printf '%s\n' "$status" > "$BASE/$NAME.exit"
    tmux wait-for -S "qwen38-$NAME-done"
}
trap finish_early EXIT
export QWEN38_RESIDENT=1
unset QWEN38_API_BENCH
export QWEN38_ENGINE_IMAGE=qwen38-flash-dgx:a5b-resident-20261030-q9
export QWEN38_VISIBLE_DEVICES=0,1
export QWEN38_MODEL_PATH=/work/target-no-mtp
export QWEN38_DRAFT_PATH=/work/draft-k10-int4
export QWEN38_ASYNC_TRANSFERS=1
export QWEN38_HELPER_GRAPHS=io
export QWEN38_SHARED_OVERLAP=0
export QWEN38_LOCAL_EXPERT_LAYERS=0
export VLLM_MTP_DRAFT_HEAD=int4
export VLLM_MTP_DRAFT_VOCAB=/work/draft-vocab-generic-65536.npy
export VLLM_MTP_DRAFT_VOCAB_CKPT=/work/draft-k10-int4
export VLLM_MTP_DRAFT_SCALE=1
export VLLM_PLE_MMAP_MADV_RANDOM=0
export VLLM_PLE_MMAP_PREWARM=1
export VLLM_PLE_MMAP_STATS_SEC=15
export QWEN38_INDEX_SHARE=1
export QWEN38_MAMBA_DTYPE=bfloat16
export QWEN38_CACHE_PROBE_PATH=/work/resident-a-dual-q32b-cache-plan.json
export QWEN38_SCHED_DIAGNOSTIC=1
export QWEN38_INIT_TIMEOUT_SECONDS=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QWEN38_ZW_PREBAKE=1
exec /bin/bash "$BASE/run_engine_hostnet.sh" "$NAME" 15 PIECEWISE --mode serve --state-file "/work/$NAME.json" --served-model-name qwen38-mia-a5b --protocol-profile final --max-model-len 17408 --max-num-seqs 1 --kv-mib 4096 --block-size 160 --max-num-batched-tokens 4096 --draft-sample-method probabilistic --rejection-sampling standard --reasoning-parser qwen3 --server-start-timeout 1800

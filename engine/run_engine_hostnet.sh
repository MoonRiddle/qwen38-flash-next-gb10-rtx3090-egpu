#!/usr/bin/env bash
set -euo pipefail
BASE="${RECIPE_HOME:-/home/user/OMP_Workspace/qwen38-gb10-3090-20261029}"
MODELS=/srv/vault/int4-models
MODEL=${QWEN38_MODEL_PATH:-/work/target-no-mtp}
DRAFT=${QWEN38_DRAFT_PATH:-/work/draft-k10-compact}
ENGINE_IMAGE=${QWEN38_ENGINE_IMAGE:-qwen38-flash-dgx:a5b-int4}
BENCH=/work/bench_engine.py
DEFAULT_TIMEOUT_SECONDS=600
DOCKER_LIMITS=()
AZ_PLE_MOUNT=()
if [[ -n "${QWEN38_AZ_PLE_HOST_DIR:-}" ]]; then
    AZ_PLE_MOUNT=(-v "${QWEN38_AZ_PLE_HOST_DIR}:/work/az-download-stage-q1:ro")
fi
if [[ "${QWEN38_API_BENCH:-0}" == "1" ]]; then
    BENCH=/work/mia_api_bench.py
fi
if [[ "${QWEN38_RESIDENT:-0}" == "1" ]]; then
    BENCH=/work/resident_bench.py
    DEFAULT_TIMEOUT_SECONDS=0
    DOCKER_LIMITS=(--ulimit memlock=-1)
fi
NAME=${1:?run name required}
MTP=${2:-3}
GRAPHS=${3:-PIECEWISE}
VISIBILITY=${QWEN38_VISIBLE_DEVICES:-GPU-f2b51a5e-26fe-622b-2c31-0885c660a5df}
CHANNEL=qwen38-$NAME-done
finish() {
    status=$?
    if [[ "$status" != 0 ]]; then
        docker stop --timeout 10 "qwen38-$NAME" > "$BASE/$NAME.cleanup.log" 2>&1 || true
    fi
    printf '%s\n' "$status" > "$BASE/$NAME.exit"
    tmux wait-for -S "$CHANNEL"
}
trap finish EXIT
exec 9>"$BASE/benchmark.lock"
flock 9
bash "$BASE/check_egpu_link.sh" > "$BASE/$NAME.egpu.log" 2>&1
exec_log="$BASE/$NAME.log"
timeout --signal=TERM --kill-after=20s "${QWEN38_TIMEOUT_SECONDS:-$DEFAULT_TIMEOUT_SECONDS}s" docker run --rm --name "qwen38-$NAME" --network host --ipc=host --gpus all --cpuset-cpus=5-9,15-19 "${DOCKER_LIMITS[@]}" \
    -v "$BASE:/work" -v "$MODELS:$MODELS:ro" \
    "${AZ_PLE_MOUNT[@]}" \
    -v "$BASE/vllm-cache:/root/.cache/vllm" \
    -v "${QWEN38_PLE_HOST_DIR:-/mnt/nvme20/ple-table-int4}:/home/user/ple-table-int4:ro" \
    -v "$BASE/peer_worker.py:/usr/local/lib/python3.12/dist-packages/peer_worker.py:ro" \
    -v "$BASE/benchmark_worker.py:/usr/local/lib/python3.12/dist-packages/benchmark_worker.py:ro" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e "CUDA_VISIBLE_DEVICES=$VISIBILITY" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_PLE_MMAP=1 -e "VLLM_PLE_MMAP_WORKERS=${VLLM_PLE_MMAP_WORKERS:-32}" -e "VLLM_PLE_MMAP_PREWARM=${VLLM_PLE_MMAP_PREWARM:-0}" \
    -e "VLLM_PLE_MMAP_MADV_RANDOM=${VLLM_PLE_MMAP_MADV_RANDOM:-1}" -e VLLM_PLE_MMAP_FAST_ROWS=512 \
    -e VLLM_DISABLE_COMPILE_CACHE=1 \
    -e "VLLM_PLE_MMAP_DIR=${VLLM_PLE_MMAP_DIR:-/home/user/ple-table-int4}" \
    -e "VLLM_MARLIN_USE_ATOMIC_ADD=${VLLM_MARLIN_USE_ATOMIC_ADD:-1}" -e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0 \
    -e "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-1}" -e "VLLM_MTP_DRAFT_SCALE=${VLLM_MTP_DRAFT_SCALE:-2}" \
    -e VLLM_MTP_DRAFT_VOCAB \
    -e VLLM_MTP_DRAFT_HEAD \
    -e VLLM_MTP_DRAFT_VOCAB_CKPT \
    -e VLLM_MTP_DRAFT_SCALE_PER_POS \
    -e VLLM_MTP_INPUT_GAMMA \
    -e CUDA_DEVICE_MAX_CONNECTIONS \
    -e CUDA_MODULE_LOADING \
    -e QWEN38_COMPACT_DRAFT_VOCAB \
    -e QWEN38_COMPACT_DRAFT_TOKENIZER_SHA256 \
    -e QWEN38_LOCAL_EXPERT_LAYERS \
    -e QWEN38_HELPER_GRAPHS \
    -e QWEN38_ASYNC_TRANSFERS \
    -e QWEN38_HOT_CACHE_PROFILE \
    -e QWEN38_SHARED_OVERLAP \
    -e QWEN38_PEER_DIAGNOSTICS \
    -e QWEN38_PROFILER_CONFIG \
    -e QWEN38_ROUND_PROFILE \
    -e QWEN38_CODE_DRAFT_TOKENS \
    -e QWEN38_ROUND_RECORD_FUNCTION \
    -e QWEN38_INDEX_SHARE \
    -e QWEN38_MAMBA_DTYPE \
    -e QWEN38_QAD_NATIVE \
    -e QWEN38_NVFP4_BRIDGE_LIBRARY \
    -e CUDA_LAUNCH_BLOCKING \
    -e QWEN38_SCHED_DIAGNOSTIC \
    -e QWEN38_CACHE_PROBE_PATH \
    -e VLLM_DEBUG_WORKSPACE \
    -e QWEN38_MTP_FORCE_TRITON \
    -e VLLM_BATCH_INVARIANT \
    -e VLLM_ENABLE_V1_MULTIPROCESSING \
    -e VLLM_QSA_EXACT_TOPK \
    -e QWEN38_QSA_STABLE_TOPK \
    -e QWEN38_MOE_STABLE_ALIGN \
    -e OMP_NUM_THREADS=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e "QWEN38_INIT_TIMEOUT_SECONDS=${QWEN38_INIT_TIMEOUT_SECONDS:-420}" \
    -e TRITON_CACHE_DIR=/work/triton-cache --entrypoint python3 \
    "$ENGINE_IMAGE" "$BENCH" \
    --model "$MODEL" --draft "$DRAFT" --mtp "$MTP" --graphs "$GRAPHS" \
    --out "/work/$NAME.json" "${@:4}" > "$exec_log" 2>&1

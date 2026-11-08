# 409,600 with concurrency 2.0x

This page records how the 409,600 window moved from one resident request to two on the
GB10 + RTX 3090 pair, and what each lever actually bought.

The decode cost of this configuration is analysed in `docs/12-decoding-regression.md`, which
lists every stage from 204,800 to 340,000 with its flags and measured rates.

## Starting point

The route A engine (`sep-e`) served 409,600 with a 14,336 MiB KV pool:

| Item | Value |
|---|---|
| `GPU KV cache size` | 427,468 tokens |
| `Maximum concurrency for 409,600 tokens per request` | 1.04x |
| RTX 3090 resident | 22,985 MiB of 24,176 MiB |
| Free on the 3090 at that point | 1,191 MiB |

A 409,600 request needs 2,560 blocks at `--block-size 160` (409,600 / 160). Two requests
need 5,120 blocks, so concurrency 2.0x requires 819,200 tokens of KV, which does not fit
the 427,468-token pool on this card.
The only way to hold two full 409,600 requests is a narrower cache element, not a larger
allocation.

## Where the window comes from

The checkpoint's `text_config.max_position_embeddings` is 262,144. Everything above that is a
YaRN stretch, not native support. Each view carries its own `rope_parameters`:

| View | `max_position_embeddings` | YaRN factor |
|---|---:|---:|
| `target-409k` | 409,600 | 1.5625 |
| `target-524k` | 524,288 | 2.0 |
| `target-819k` | 819,200 | 3.125 |

Each factor is exactly window / 262,144. `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` lets vLLM accept the
longer `--max-model-len`. The 262,209-token boundary pass proves YaRN runs, but quality beyond
262,144 is not guaranteed to match the native range.

## Step 1: raise the BF16 pool

`sep-f16` asked for `--kv-mib 16000` (15.62 GiB) with no code change. Admission computed
`476,950 tokens, 1.16x`, then failed on a 38 MiB allocation with 32.19 MiB free.

| Attempt | Admission | Outcome |
|---|---|---|
| `--kv-mib 14336` | 427,468 tokens, 1.04x | ready |
| `--kv-mib 16000` | 476,950 tokens, 1.16x | OOM, 38 MiB request, 32.19 MiB free |

The 1,664 MiB increase exceeds the 1,191 MiB headroom by 473 MiB, so this is an absolute
shortage rather than fragmentation.

## Step 2: allocator changes

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8` slowed
checkpoint loading badly (shard 40 onward dropped to 12.86 s/it) and pushed host RAM to
121 GiB used with `MemAvailable 0`, which stalled SSH for tens of seconds. Both options were
removed and only an extra `torch.cuda.empty_cache()` was kept in `peer_worker.py`.

The retry reproduced the same 38 MiB OOM, which confirmed the shortage reading.

## Step 3: draft embedding table

The draft MTP predictor allocates its own `VocabParallelEmbedding` on the 3090. Moving that
table to CPU frees 1.19 GiB in practice: the draft `primary_bytes` fell from 9,757,622,784
to 8,567,792,128. `qwen38_mtp.py` now gathers on the table's device and moves the result
back to the caller's device, so the forward pass stays unchanged.

The target model reported 6,729,151,488 bytes in the same run, down from the 9,757,622,784
seen before the `--language-model-only` exclusion was in effect.

## Step 4: vision tower

`mia_api_bench.build_api_server_command` already passes `--language-model-only`, so the
vision tower is excluded and the engine reports `free memory 23.34 GiB` at start. The
`resident_bench.py` argument parser does not expose this flag, which is why it must not be
passed on the command line.

## Step 5: 8-bit KV cache

Four places in the QSA fork rejected or misread `fp8_e5m2`:

| Location | Defect |
|---|---|
| `_fp8` tuple in `qsa.py` | `fp8_e5m2` missing, so the parent guard raised |
| `forward_qsa` uint8 view | always viewed as `float8_e4m3fn` |
| storage dtype check | `float8_e5m2` rejected |
| `qsa_mqa_paged` uint8 view | always viewed as `float8_e5m2` |

The patched files are mounted as `qwen38_qsa_v2.py` and `qwen38_ops_qsa_v2.py` because the
originals are root-owned and cannot be overwritten over scp.

With `--kv-cache-dtype fp8_e5m2 --kv-mib 14336`, admission reported:

```
GPU KV cache size: 677,415 tokens, Maximum concurrency for 409,600 tokens per request: 1.65x
```

This is the first time the 8-bit path produced a working engine in this fork.

## Step 6: concurrency 2.0x

`--kv-mib 17408` (17 GiB) with `--max-num-seqs 2` produced:

```
GPU KV cache size: 823,701 tokens, Maximum concurrency for 409,600 tokens per request: 2.01x
```

Admission meets the target, but the engine still fails during `_allocate_kv_cache`
(`attn_utils.py:214`, `torch.zeros(..., dtype=torch.int8)`), which asks for a 1.23 GiB tensor
while only 570 MiB is free. At that point PyTorch already holds 22.63 GiB and the process
occupies 23.03 GiB of 23.61 GiB.

Moving the draft embedding table to CPU did reduce the draft resident bytes from
9,757,622,784 to 8,567,792,128, but the KV allocation happens with the same 570 MiB free, so
the saving does not reach the failing allocation. The model plus non-PyTorch overhead leaves
at most about 15.8 GiB for KV, and 2.0x needs 17.0 GiB.

The measured ceiling therefore sits below 2.01x: the 17,408 MiB request cannot be satisfied
in one process, and the highest KV budget that still starts is probed at 15,616 MiB
(`sep-g2c`).

## What the numbers say

| Window | KV dtype | `--kv-mib` | Admission | Concurrency |
|---|---|---|---|---|
| 409,600 | bf16 | 14,336 | 427,468 tokens | 1.04x |
| 409,600 | bf16 | 16,000 | 476,950 tokens | 1.16x, OOM |
| 409,600 | fp8_e5m2 | 14,336 | 677,415 tokens | 1.65x |
| 409,600 | fp8_e5m2 | 17,408 | 823,701 tokens | 2.01x, OOM at 1.23 GiB |
| 409,600 | fp8_e5m2 | 15,616 | 738,180 tokens | 1.80x, ready |

The 8-bit cache multiplies the token count by 1.586 at the same byte budget (677,415 /
427,468), which matches the 2x element density minus the fixed per-block overhead.

## Step 7: the KV budget that actually starts

Lowering the pool to `--kv-mib 15616` (15.25 GiB) let the engine reach ready at 409,600:

```
GPU KV cache size: 738,180 tokens, Maximum concurrency for 409,600 tokens per request: 1.80x
```

Engine load took 276.17 s. That budget survives the peak but only fits one 409,600 request.

A 340,000 window with `--kv-mib 15360` also reaches ready, at 296.40 s:

```
GPU KV cache size: 677,901 tokens, Maximum concurrency for 340,000 tokens per request: 1.99x
```

One 70 MB allocation failed on the way with 56 MiB free, but the engine recovered and served.
The attention page size is 3,520 tokens for the attention group, but hybrid KV capacity is not
that number divided into the pool: upstream `kv_cache_utils.py` sizes capacity from each KV
group's per-block demand. The exact block count needs `KVCacheConfig` inspection rather than
this division, so no concurrent total is asserted here.

With `--kv-mib 14336` the pool admits `631,728 tokens, 1.86x`. A long-prompt probe sent two
requests at 4K, 16K, 32K and 315,000 (prompt tokens 4,244, 16,975, 33,940 and 326,260 per
stream) and all completed, but it was non-streaming and recorded only wall time, so TTFT
overlap and simultaneous residency are unverified.

At 409,600 a real two-request run failed. The sparkDash matrix at `C=1,2` reported
`C=2 had failed streams` on the first prose case, because the 409,600 window needs more KV
than the pool holds for two requests. The admission ratio describes how much of one 409,600
request fits, not how many requests fit, so 1.80x is a single-request ceiling rather than a
route to 2.0x at that window.

## Measured at 340,000

Decode, 400 forced tokens, sparkDash native runner at commit `7c2b9638`:

| Case | C=1 | C=2 per request | C=2 aggregate | Failed streams |
|---|---:|---:|---:|---:|
| code 1 | 114.11 | 86.79 | 163.46 | 0 |
| code 2 | 108.13 | 83.83 | 156.11 | 0 |
| code 3 | 115.32 | 90.60 | 177.42 | 0 |
| prose 1 | 34.74 | 37.70 | 73.25 | 0 |
| prose 2 | 45.85 | 41.87 | 78.35 | 0 |
| prose 3 | 59.28 | 40.16 | 79.66 | 0 |

Every stream succeeded at both concurrency levels, which is the first real proof that two
requests run at once on this pair. Code aggregate moves from a 108-115 range at C=1 to
156-177 at C=2, and prose aggregate from 35-59 to 73-80, while per-request rates fall as
expected. The C=2 per-request spread is tight (code 83.83-90.60, prose 37.70-41.87), which
is the more stable side of the measurement.

Prefill on the same engine failed at every size: all three rows returned `prefillTps 0`
and `samples 0` (`fetch failed` on `a-sep-c2`, HTTP 500 on `a-sep-c2s48`). The earlier
prefill table in this page quoted 2,289.39 / 2,209.43 / 2,149.69, which no `a-sep-c2`
record contains; those numbers are withdrawn.

Prefill, `--max-num-batched-tokens 1024`: withdrawn. The `a-sep-c2` and `a-sep-c2s48`
records show 0/3 samples at 4K, 16K and 32K, so no prefill rate is reported for the
340,000 window on this pair.

The first request in each job is cold; later repeats share the warm engine.

## Where the limit sits

The binding constraint is the KV allocation peak, not admission. Two levers remain untried:

1. Lower `--kv-mib` until `_allocate_kv_cache` succeeds. The ceiling is about 15.8 GiB, which
   yields roughly 1.8x, so this closes the loop on the true maximum rather than reaching 2.0x.
2. Lower `--max-model-len` so two requests fit the same block count. The 360,000 window
   admits 702,857 tokens at 1.95x and the 348,000 window admits `696,000 tokens, 2.00x`, but
   the start-up peak still fails on 75 MB with 58 MiB free, so the binding limit is the peak
   itself rather than the admission math. The next probe drops the pool to `--kv-mib 15360`.
3. Move the whole draft MTP layer to the GB10 helper. That frees the remaining 1.71 GiB of
   draft resident bytes and would put KV near 17.5 GiB, which does fit 2.0x. It costs a
   cross-device hop per decode step, so it trades decode speed for concurrency.

Both are outside the original plan scope and are recorded here as the next decision point.

## Measurement plan

Once a concurrency-2 engine reaches ready, the sparkDash native runner measures both
concurrency levels with the same pinned commit:

| Case | C=1 | C=2 |
|---|---|---|
| code decode | 3 jobs, 400 tokens | 3 jobs, 400 tokens |
| prose decode | 3 jobs, 400 tokens | 3 jobs, 400 tokens |
| prefill | 3 depths | 3 depths |

Prefill depths are 4,096, 16,384 and 32,768 tokens. Deeper depths (131,072, 200,000, 261,000)
exceed the concurrency-2 KV budget and kill the core during allocation.

## Reproduce

```bash
export QWEN38_MODEL_PATH=/work/target-409k
export QWEN38_DRAFT_PATH=/work/draft-409k
export VLLM_MTP_DRAFT_HEAD=int4
export VLLM_PLE_MMAP_MADV_RANDOM=0
export VLLM_PLE_MMAP_PREWARM=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32
/bin/bash run_engine_hostnet.sh sep-c2 15 NONE \
    --mode serve --state-file /work/sep-c2.json \
    --protocol-profile final --max-model-len 340000 \
    --max-num-seqs 2 --kv-mib 15360 --kv-cache-dtype fp8_e5m2 \
    --block-size 160 --max-num-batched-tokens 1024
```

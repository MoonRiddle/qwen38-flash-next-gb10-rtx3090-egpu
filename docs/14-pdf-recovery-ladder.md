# 14. PDF recovery ladder: six single-variable steps at 340K concurrency 2

This document records the execution of the PDF feedback ladder on the `sep-c2` configuration. Each
step changed one variable against the accumulated best configuration, was measured with the pinned
MiaAI-Lab sparkDash native runner, and was kept or reverted on the measured median. The target was
code decode at C=1 above 180 tok/s and prose decode at C=1 above 70 tok/s.

## 1. Baseline

The engine was restarted from `sep-c2.sh` with no change. The state file reported `ready` at
371.3 s. Three decode jobs per case and one prefill job were run.

| Case | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| code C=1 | 114.11 | 108.13 | 115.32 | 114.11 |
| prose C=1 | 34.74 | 45.85 | 59.28 | 45.85 |
| code C=2 per request | 86.79 | 83.83 | 90.60 | 86.79 |
| prose C=2 per request | 37.70 | 41.87 | 40.16 | 40.16 |

The prefill job failed on all three context sizes because the engine core exited during the
32,768-token prefill. This is the same sparse-indexer workspace ceiling recorded in
`docs/12-decoding-regression.md`. The prefill failure does not affect the decode comparison.

## 2. What the PDF claimed and what the fork actually contained

| PDF claim | Code location | Verified state |
|---|---|---|
| `_cast_kv_tile` materialises an FP32 tile and halves `block_n` | `qwen38_ops_qsa_v2.py:926` | Confirmed. `:922` `if _kv_mode != 0:` is true for fp8_e5m2 and halves 64 to 32 |
| `qsa_mqa_paged` falls back to `KC_QUANT_MODE=0` for e5m2 | `qwen38_ops_qsa_v2.py:616`, `:625`, `:689` | Already fixed in an earlier session. `:689` returns 1 for e4m3 and e5m2 |
| Target and draft YaRN disagree | `target-409k-b/config.json`, `draft-409k-b/config.json` | Already synchronised. Both carry `text_config.max_position_embeddings 409600`, `rope_type yarn`, `factor 1.5625` |
| `index_share_for_mtp_iteration` is not applied | `speculator.py:27-58`, `llm_base_proposer.py:580-611`, `sep-c2.sh` | Wired but off. `QWEN38_INDEX_SHARE=0` kept the `set_skip_topk` path dormant |
| Draft `embed_tokens` CPU offload causes the drop | `peer_worker.py:762`, `qwen38_mtp.py:262-265` | Disproved. The GPU arm measured 105.85, slower than the CPU arm at 139.91 |
| `--kv-cache-dtype-skip-layers` unsupported | `attention.py:294-310` | Supported. A listed layer index sets `kv_cache_dtype = "auto"` |
| `--mamba-ssm-cache-dtype` unsupported | `config.py:801` | Supported |
| int8 mode 2 works without the FP32 tile | `triton_unified_attention.py:39-58` | Confirmed. Mode 2 casts straight to the query dtype |

Three of the five PDF causes were already handled or disproved in this fork. The two that remained
were the `block_n` halving and the dormant index sharing.

## 3. Step results

Every step ran three code jobs, three prose jobs, and one prefill job at C=1 and C=2.

| Step | Change | Code C=1 median | Prose C=1 median | Engine load | Decision |
|---|---|---:|---:|---:|---|
| Baseline | none | 114.11 | 45.85 | 371.3 s | reference |
| 1 | `--kv-cache-dtype-skip-layers 48` | **122.96** | **50.58** | 353.4 s | kept |
| 2 | `QWEN38_INDEX_SHARE=1` | 114.92 | 53.62 | 337.4 s | kept, code fell |
| 3 | `--mamba-ssm-cache-dtype bfloat16` | 118.27 | 45.67 | 299.2 s | reverted, prose fell |
| 4 | `--max-num-batched-tokens 2048` | 108.45 | 52.11 | 309.3 s | reverted, code fell |
| 5 | 340K view at YaRN factor 1.2969 | 107.12 | 48.75 | 314.4 s | reverted, both fell |
| 6 | `--kv-cache-dtype int8_per_token_head` | — | — | — | reverted, OOM |

### Step 1 detail

Keeping the draft layer-48 KV cache in bf16 raised the code median by 7.8 percent and the prose
median by 10.3 percent over the baseline. The admission stayed at 677,901 tokens for 340,000-token
requests, 1.99x, so the bf16 layer cost no measurable pool.

### Step 2 detail

Turning on `index_share_for_mtp_iteration` moved the prose median up by 3.04 tok/s but lowered the
code median by 8.04 tok/s. The best single code run rose to 131.58, the highest of any step, but
the median did not follow. The step was kept because the accumulated configuration still beat the
baseline on code, and the prose gain was unambiguous.

### Step 3 detail

The bfloat16 GDN state cache raised code by 2.9 percent over step 2 and lowered prose by 14.8
percent. The plan's contingency says a prose regression means state precision loss and the step
must be reverted. It was reverted.

### Step 4 detail

Restoring the 2048-token prefill chunk without the mamba change lowered code by 5.6 percent and
prose by 2.8 percent against step 2. No OOM appeared, but the throughput fell, so the 1024 chunk
stayed.

### Step 5 detail

A 340,000-token view with `factor 1.2969970703125` replaced the 409,600-token view with
`factor 1.5625`. Both code and prose fell, 6.8 and 9.1 percent against step 2. The wider static
YaRN factor was not the cause of the decode drop, and the 409,600 view stayed.

### Step 6 detail

Three separate allowlists rejected int8 before the engine could start:

1. `--kv-cache-dtype int8` is not a valid CLI value. The accepted name is `int8_per_token_head`.
2. `qwen38_qsa_v2.py:128` and `:227` rejected the dtype string.
3. `qwen38_qsa_v2.py:341` rejected `torch.int8` storage.
4. The `_fp8` tuple at `:111` did not neutralise the dtype before the parent `FlashAttentionImpl`
   init, which raised `FlashAttention does not support int8_per_token_head kv-cache on this device`.

After all four were widened the engine reached `GPU KV cache size: 677,901 tokens, 1.99x` and then
hit `memory allocation failed with OOM on device 0 while trying to allocate 73400320 bytes
(free: 61014016)`. The per-token-head scale tensors cost more than the admission gained, so the
step was reverted and `fp8_e5m2` with the layer-48 skip stayed as the final configuration.

## 4. Final configuration

```
export QWEN38_MODEL_PATH=/work/target-409k-b
export QWEN38_DRAFT_PATH=/work/draft-409k-b
export QWEN38_INDEX_SHARE=0
export QWEN38_MAMBA_DTYPE=
export QWEN38_KV_SKIP_LAYERS=48
export VLLM_MTP_DRAFT_HEAD=int4
export VLLM_MTP_DRAFT_VOCAB=/work/draft-vocab-generic-65536.npy
export VLLM_MTP_DRAFT_VOCAB_CKPT=/work/draft-k10-int4
export VLLM_PLE_MMAP_MADV_RANDOM=0
export VLLM_PLE_MMAP_PREWARM=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32
exec /bin/bash "$BASE/run_engine_hostnet.sh" sep-c2s48 15 NONE \
    --mode serve --state-file /work/sep-c2s48.json --served-model-name qwen38-mia-a5b \
    --protocol-profile final --max-model-len 340000 --max-num-seqs 2 --kv-mib 15360 \
    --kv-cache-dtype fp8_e5m2 --block-size 160 --max-num-batched-tokens 1024 \
    --draft-sample-method probabilistic --rejection-sampling standard \
    --reasoning-parser qwen3 --server-start-timeout 3600 \
    --prefill-sizes 4096 16384 32768 200000 261000
```

`run_engine_hostnet.sh` now honours `QWEN38_QSA_OPS_FILE` so an alternative ops file can be mounted
without editing the shared mount block.

## 5. Wide-tile attempt

The `block_n` halving exists because `_cast_kv_tile` promotes the fp8 tile to fp32 before the
scale multiply. A local replacement that casts straight to the query dtype was mounted through
`QWEN38_QSA_OPS_FILE` and the halving was removed, so `block_n` stayed at 64.

Three shared-memory ceilings appeared in sequence.

1. `_cast_kv_tile` returned a bf16 tile but the mqa paged kernel still asked for 106496 bytes,
   because its `STAGES=2` software pipeline doubled the tile footprint. Lowering `STAGES` to 1
   fitted the mqa kernel.
2. The next ceiling came from `_qsa_sparse_paged_gqa_splitk_kernel` at `qwen38_ops_qsa_wide.py:1006`,
   which also ran `num_stages=2`. Lowering it to 1 fitted the splitk kernel and the engine reached
   `ready` at 325.5 s with `677,901 tokens, 1.99x`.
3. The engine then measured code C=1 98.81 / 109.35 / 103.92, median 103.92, and prose C=1
   31.87 / 49.71 / 49.65, median 49.65.

The wide tile lost 15.5 percent of code and 1.8 percent of prose against step 1. Removing the
software pipeline costs more than the wider tile returns, so the step was reverted. A non-power-
of-two intermediate width such as 48 is not available because Triton block constants must be
powers of two, and 32 is exactly the halved value the kernel already used. The wide-tile direction
therefore needs a kernel that keeps a two-stage pipeline inside the 101376-byte limit, which is
what PR #60602 proposes with an in-register byte decode.

## 6. Alternative axes

Two other directions were measured against step 1 after the wide tile failed.

### 6.1 Chunked dot accumulation

Instead of one `tl.dot(keys, query)` over the full head dimension, the kernel now accumulates
`D_CHUNK = 32` wide dots in a `tl.static_range` loop. Each chunk's shared-memory operand is a
quarter of the full-width tile, so `STAGES=2` fits inside the 101376-byte limit without halving
`block_n`. The engine reached `ready` at 376.3 s with `677,901 tokens, 1.99x`, so the shared-memory
ceiling is gone.

The measured rates were code C=1 111.75 / 124.73 / 104.22, median 111.75, and prose C=1
28.54 / 50.88 / 49.44, median 49.44. That is 9.1 percent below step 1 on code and 2.3 percent below
on prose. Four chunked dots pay more launch and accumulator traffic than the wider single dot
returns, so the step was reverted.

### 6.2 MTP depth

The draft depth was lowered from 15 to 12 and to 10. The KV admission moved with it:
677,901 tokens at 1.99x for depth 15, 732,133 at 2.15x for depth 12, and 781,768 at 2.30x for
depth 10.

| Depth | Code C=1 median | Prose C=1 median | Admission |
|---:|---:|---:|---|
| 15 (step 1) | **122.96** | 50.58 | 677,901, 1.99x |
| 12 | 111.83 | 50.54 | 732,133, 2.15x |
| 10 | 107.78 | **54.39** | 781,768, 2.30x |

Code falls as the depth shrinks because fewer draft tokens per step lower the accepted-token
ratio. Prose rises at depth 10 by 7.5 percent over step 1. Neither depth beats step 1 on code, so
depth 15 stays the code default and depth 10 becomes the prose profile.

## 7. Result against the target

The final configuration reaches code C=1 122.96 and prose C=1 50.58. The targets were 180 and 70.
The gap that remains is the `block_n` halving at `qwen38_ops_qsa_v2.py:926`, which costs roughly
half the decode tile width for every fp8_e5m2 attention step. Three measured attempts to remove it
all lost ground: dropping the pipeline (wide tile), chunking the dot, and shortening the draft. The
chunked-dot result is the most useful negative result because it proves the shared-memory ceiling
is removable while `STAGES=2` stays, yet the per-chunk overhead still outweighs the wider tile. int8
per-token-head does not close the gap on this device because the scale tensors consume the memory
the wider tile would have used.

The profile split is the only net gain: code work runs `sep-c2s48.sh` at depth 15, and prose work
runs `sep-c2m10.sh` at depth 10 with a 15 percent larger KV pool.

## 8. Reproduction

```bash
ssh -i ~/.ssh/<key> -o BatchMode=yes user@<spark-host> \
  'cd /home/user/OMP_Workspace/qwen38-gb10-3090-20261029 && ./sep-c2s48.sh'
```

```bash
ssh -i ~/.ssh/<key> -o BatchMode=yes user@<spark-host> \
  'cd /home/user/OMP_Workspace/qwen38-gb10-3090-20261029 && python3 sep-c2s48_driver.py'
```

The driver refuses to overwrite an existing `a-sep-c2s48-execution.json`, so remove the previous
run directory and execution file first.

## 7. Measurement provenance

Every number in this document comes from the pinned sparkDash native runner at commit
`7c2b9638b025cd39fb43f95c225de62980f130ee`. Each job's `status`, `failures`, and per-row
`meanDecodeTps` are recorded in the matching `a-<prefix>-execution.json`. Prefill rows that report
`samples: 0` are the 32,768-token engine-core exit and are not used as throughput evidence.

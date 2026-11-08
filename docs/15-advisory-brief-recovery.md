# 15. Single-stream decoding recovery: advisory brief

This file is written for an outside AI or engineer who has no prior context. Reading only this document should
be enough to understand the hardware, the model, the measurement contract, what has been measured, what failed
with which exact errors, and where the work is now blocked.

- Section 0 is background.
- Sections 1-6 are the measured record.
- Sections 7-9 are the open questions, the candidates that remain, and the artifacts.

Every number below is copied from the sparkDash native artifacts and from `docs/14-pdf-recovery-ladder.md`.
Nothing is modelled. Unconfirmed claims are marked at the point they appear.

---

## 0. Background

### 0.1 One-sentence summary

This is an experiment to serve `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` on a GB10 unified-memory
box plus an RTX 3090 eGPU, at a 340,000-token window and concurrency 2, while recovering the single-stream
decode rate that concurrency 2 cost.

### 0.2 The two targets

| Workload | Target | Current best | Gap |
|---|---:|---:|---:|
| code decode, C=1 | 180 tok/s | 122.96 | 57.04 |
| prose decode, C=1 | 70 tok/s | 54.39 | 15.61 |

Code and prose are measured separately because they behave differently on this recipe. The code path uses the
full MTP depth; the prose path's best configuration is a different depth.

### 0.3 Hardware

| Device | Character | Role |
|---|---|---|
| GB10 (SM121) | CPU and GPU share one 121.7 GiB unified pool, 246 GB/s | Model weights, PLE page cache |
| RTX 3090 (SM86) | Separate 24 GiB GDDR6X, 833 GB/s, eGPU | Hot-expert cache, KV cache. The bottleneck |

The link is PCIe Gen4 x4 and Non-P2P; transfers go through host RAM. Measured round trip is 33.14 us at 20 KB,
and copy bandwidth is 6.20 GB/s host to device and 6.59 GB/s device to host. The 3090 carries a software unlock
that makes BF16 compute 70-80 percent faster than stock. This recipe did not create that state and does not
change it.

### 0.4 Model

| Item | Value |
|---|---|
| Total parameters | 125B |
| Active parameters per token | 4.8B |
| Structure | MoE, all 512 routed experts retained, 5 selected per token |
| Quantization | INT4 AutoRound (W4A16) |
| lm_head | int4 GPTQ-Marlin g128, 97.2 percent top-1 agreement with BF16 |
| Draft head | BF16 top-10 |
| PLE table | 51 GiB int4, mmapped from USB, used in Layer 1 only |

The PLE table does not fit beside the weights on one device. It lives on a 20 Gbps USB NVMe and is read through
mmap, with a 10 Gbps copy also present.

### 0.5 Measurement contract

All rates come from the MiaAI-Lab sparkDash native runner pinned at commit
`7c2b9638b025cd39fb43f95c225de62980f130ee`. The rules this work follows:

- A rate is reported only when the execution status is `completed` and `failures` is empty.
- Three runs are taken per case and the median is reported. Peak values and summed concurrent rates are never
  used as evidence of a target being met.
- The engine is started once and all cases run against that one engine; a restart per case is not used.
- Warm-up, repetitions and EOS handling follow the upstream collector unchanged.

### 0.6 The serving configuration

The engine serves a 340,000-token window with `--max-num-seqs 2`. Two resident views are built for this window:

| View | Path | Notes |
|---|---|---|
| target-340k | `/work/target-340k` | YaRN factor 1.2969970703125 |
| draft-340k | `/work/draft-340k` | Same factor, synchronized |

`max_position_embeddings` is 340000, `rope_type` yarn, `original_max_position_embeddings` 262144,
`mrope_section [11,11,10]`, `partial_rotary_factor 0.25`, `rope_theta 10000000`.

The KV cache is `fp8_e5m2`, `block_size 160`, `kv-mib 15360`. Admission reports `677,901 tokens, 1.99x` for
340,000 tokens per request at concurrency 2.

---

## 1. What concurrency 2 cost

Concurrency 2 was required to serve the 340K window on this card pair. The side effect was a drop in
single-stream decode. The measured baseline before any recovery work was code C=1 114.11 and prose C=1 45.85.
The targets are 180 and 70, so the recovery work began from a position already below target.

---

## 2. The six-step ladder

Six single-variable steps were measured against that baseline. Each step kept or reverted its change based on
the measured rates, not on expectation.

| Step | Change | Code C=1 | Prose C=1 | Admission | Verdict |
|---|---|---:|---:|---|---|
| baseline | — | 114.11 | 45.85 | 677,901, 1.99x | reference |
| 1 | `--kv-cache-dtype-skip-layers 48` | **122.96** | **50.58** | 677,901, 1.99x | kept |
| 2 | `QWEN38_INDEX_SHARE=1` | 114.92 | 53.62 | 677,901, 1.99x | mixed |
| 3 | mamba SSM cache bf16 | 118.27 | 45.67 | — | reverted |
| 4 | batch 2048 | 108.45 | 52.11 | — | reverted |
| 5 | 340K view, factor 1.2969 | 107.12 | 48.75 | — | reverted |
| 6 | int8 per-token-head KV | — | — | OOM | reverted |

Step 1 is the only clear net gain: code +7.8 percent, prose +10.3 percent, with no admission loss. Step 2
trades code for prose and is kept only as part of the cumulative configuration. Steps 3, 4 and 5 each lose more
than they gain and are reverted.

Step 6 deserves its own note. `int8_per_token_head` was wired into four allow-lists
(`qwen38_qsa_v2.py:111` `_fp8` tuple, `:128`, `:227`, `:341` `torch.int8`). The engine started, but admission
reported the same `677,901 tokens, 1.99x` as fp8 and then failed:

```
memory allocation failed with OOM on device 0 while trying to allocate 73400320 bytes (free: 61014016)
```

The scale tensors consume the memory the wider tile would have used, so int8 per-token-head does not buy
admission on this device. Reverted.

---

## 3. The root cause of the remaining gap

The gap that remains is a tile-width halving inside the attention kernel:

```
qwen38_ops_qsa_v2.py:926   _cast_kv_tile casts the fp8 tile through FP32, then the kernel halves block_n
```

Every fp8_e5m2 attention step loses roughly half its decode tile width because the fp8-to-bf16 cast needs a
wider shared-memory operand than the 101376-byte hardware limit allows. Three attempts to remove this halving
were measured, and all three lost ground.

### 3.1 Wide tile

The kernel `qwen38_ops_qsa_wide.py` casts the KV tile directly to bf16, removing the halving. It required three
rounds to fit shared memory:

```
OutOfResources: shared memory (Required: 106496, Hardware limit: 101376)
```

The first cause was `_qsa_mqa_paged_kernel` with `STAGES=2`; setting `STAGES=1` resolved it. The second cause
was `_qsa_sparse_paged_gqa_splitk_kernel` with `num_stages=2` at `qwen38_ops_qsa_wide.py:1006`; setting 1
resolved it.

The engine reached `ready` at 325.5 s with `677,901 tokens, 1.99x`. Measured rates were code C=1 median 103.92
and prose C=1 median 49.65. That is 15.5 percent below step 1 on code and 1.8 percent below on prose. The
`num_stages=1` pipeline loss outweighs the wider tile's gain, so the step was reverted.

### 3.2 Chunked dot accumulation

Instead of one `tl.dot(keys, query)` over the full head dimension, `qwen38_ops_qsa_chunk.py` accumulates
`D_CHUNK = 32` wide dots in a `tl.static_range` loop, with `STAGES=2` restored:

```python
        scores = tl.zeros((BLOCK_N, MAX_N), dtype=tl.float32)
        for d_start in tl.static_range(0, BLOCK_D, D_CHUNK):
            d_idx = d_start + tl.arange(0, D_CHUNK)
            d_valid = d_idx < HEAD_DIM
            k_chunk = tl.load(
                k_cache_ptr
                + safe_physical_page[:, None] * stride_cache_block
                + page_offset[:, None] * stride_cache_token
                + d_idx[None, :] * stride_cache_dim,
                mask=page_valid[:, None] & d_valid[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            q_chunk = tl.load(
                q_ptr
                + row * stride_q_row
                + heads[None, :] * stride_q_head
                + d_idx[:, None] * stride_q_dim,
                mask=(heads[None, :] < NUM_HEADS) & d_valid[:, None],
                other=0.0,
            )
            k_chunk = _cast_kv_tile(k_chunk, q_chunk, kc_scale_ptr, KC_QUANT_MODE)
            scores += tl.dot(k_chunk, q_chunk, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
```

The engine reached `ready` at 376.3 s with `677,901 tokens, 1.99x`, so the shared-memory ceiling is gone while
`STAGES=2` stays. Measured rates were code C=1 111.75 / 124.73 / 104.22, median 111.75, and prose C=1
28.54 / 50.88 / 49.44, median 49.44. That is 9.1 percent below step 1 on code and 2.3 percent below on prose.
Four chunked dots pay more launch and accumulator traffic than the wider single dot returns. Reverted.

This is the most useful negative result: it proves the shared-memory ceiling is removable without dropping the
pipeline, yet the per-chunk overhead still outweighs the wider tile.

### 3.3 MTP depth

The draft depth was lowered from 15 to 12 and to 10. The KV admission moved with it.

| Depth | Code C=1 median | Prose C=1 median | Admission | Load s |
|---:|---:|---:|---|---:|
| 15 (step 1) | **122.96** | 50.58 | 677,901, 1.99x | 353.4 |
| 12 | 111.83 | 50.54 | 732,133, 2.15x | 325.3 |
| 10 | 107.78 | **54.39** | 781,768, 2.30x | 323.3 |

Code falls as depth shrinks because fewer draft tokens per step lower the accepted-token ratio. Prose rises at
depth 10 by 7.5 percent over step 1. Neither depth beats step 1 on code.

One implementation trap was found here: the driver hardcodes `prose_max_draft_tokens`. Depth 10 and 12 need
drivers with 10 and 12 respectively; a mismatch fails `restoration_verified`.

---

## 4. Cumulative configuration that stands

`sep-c2s48.sh` is the code profile and is the best measured code configuration:

```bash
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

`sep-c2m10.sh` is identical except for `sep-c2m10 10 NONE` and is the prose profile.

---

## 5. What was checked and ruled out

Several suspected causes were checked against the source and ruled out before measuring:

| Suspect | Finding |
|---|---|
| `qsa_mqa_paged` KC_QUANT_MODE=0 fallback | Already fixed at `qwen38_qsa_v2.py:616`, `:625`, `:689` |
| Target/draft YaRN mismatch | Both views carry factor 1.5625; already synchronized |
| `index_share_for_mtp_iteration` unwired | Wired, but `QWEN38_INDEX_SHARE=0` in the code profile |
| Draft embedding on CPU | Disproven: GPU arm 105.85 is below CPU arm 139.91 |
| `--kv-cache-dtype-skip-layers` unsupported | Supported at `attention.py:294-310` |
| `--mamba-ssm-cache-dtype` unsupported | Supported at `config.py:801` |

---

## 6. Where the work stands

The best measured configuration is code C=1 122.96 and prose C=1 54.39, using different profiles for the two
workloads. Both targets remain unmet. The remaining gap is the tile-width halving, and the three measured
attempts to remove it all lost ground because each traded away something the halving's cost was smaller than:

| Attempt | What it traded | Loss |
|---|---|---|
| wide tile | software pipeline (`num_stages=1`) | code -15.5 percent |
| chunked dot | per-chunk launch and accumulation | code -9.1 percent |
| shorter draft | accepted-token ratio | code -9.1 percent at depth 12 |

---

## 7. Open questions

1. Is there a way to keep `block_n` wide and `STAGES=2` at the same time without chunking the dot? The
   chunked result shows the shared-memory ceiling is removable, so the question is whether the operand can be
   decoded in registers instead of materialized as a wide shared-memory tile. PR #60602 proposes an in-register
   byte decode; it has not been tested on this fork.
2. Does the FP32 intermediate in `_cast_kv_tile` have to exist? If the cast can go fp8 to bf16 directly in
   registers, the shared-memory operand may halve without any pipeline loss.
3. Is the accepted-token ratio at depth 15 already near its ceiling, or can the draft head's vocabulary
   (`draft-vocab-generic-65536.npy`, `draft-k10-int4`) be widened without admission loss?
4. Can the code and prose paths share one engine with a per-request depth, instead of two profiles?

---

## 8. Candidates in priority order

1. **In-register e5m2 byte decode (PR #60602 direction).** Bypass the shared-memory operand entirely: decode
   the fp8 bytes in registers and accumulate without a wide `tl.dot` operand. This is the only candidate that
   addresses the halving without trading the pipeline or the dot count.
2. **Direct fp8-to-bf16 cast.** Remove the FP32 intermediate in `_cast_kv_tile` and measure whether the shared
   memory operand shrinks enough to keep `block_n` wide at `STAGES=2`.
3. **Draft vocabulary widening.** Test whether a wider draft vocabulary raises the accepted-token ratio at
   depth 15 without admission loss.
4. **Per-request depth.** Wire the depth into the request rather than the profile, so code and prose share one
   engine.

Chunked accumulation is exhausted as a path; it should not be retried.

---

## 9. Artifacts

Remote working directory: `/home/user/OMP_Workspace/qwen38-gb10-3090-20261029`.

| Artifact | Content |
|---|---|
| `sep-c2.json` | baseline state |
| `sep-c2s48.json` | step 1 state, the code profile |
| `sep-c2is.json` | step 2 state |
| `sep-c2mb.json` | step 3 state |
| `sep-c2b2n.json` | step 4 state |
| `sep-c2ys.json` | step 5 state |
| `sep-c2wide.json` | wide-tile state |
| `sep-c2chunk.json` | chunked-dot state |
| `sep-c2m12.json` | depth 12 state |
| `sep-c2m10.json` | depth 10 state, the prose profile |
| `a-<prefix>-execution.json` | sparkDash execution record per prefix |
| `a-<prefix>-k6-open-<case>-0N.json` | per-run native artifact |

The ladder narrative is `docs/14-pdf-recovery-ladder.md`; the Korean edition is
`docs/14-pdf-recovery-ladder.ko.md`. The measurement method is `docs/01-measurement-method.md`.

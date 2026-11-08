# Recovery path measurements: int8 KV, draft embed pointer, YaRN 1.35, single-stream QSA

This note records the four untried paths from the external advisory PDF against
the step 1 configuration (`sep-c2s48.sh`). Each path is a single change on top of
that configuration, measured with the pinned sparkDash native runner
(`7c2b9638b025cd39fb43f95c225de62980f130ee`) at C=1 and C=2 with three decode
repetitions and one prefill job.

A run counts as a success only when the execution record says `status completed`,
`failures` is empty, and the three decode repetitions give a median.

## 1. int8 per-token-head KV cache with side scales

The previous int8 attempt stopped at the FlashAttention write path, which rejects
`int8_per_token_head` in `get_fp8_kv_cache_data_type`. This run replaced that write
with the Triton per-token-head reshape, allocated the float32 scale views inside the
KV padding, and taught the QSA kernel to multiply scores and values by the scales.

The engine started and admitted `677,901 tokens, 1.99x`, exactly the fp8 figure. The run
record is `failed`: the prefill job returned `prefill samples short` (only one of three sizes
completed) and `restoration_verified` is false.

| Configuration | code C=1 | prose C=1 | admission | load s |
|---|---:|---:|---|---:|
| step 1 `sep-c2s48` | **122.96** | **50.58** | 677,901, 1.99x | 353.4 |
| int8 `sep-c2i8s` | 18.47 | 16.90 | 677,901, 1.99x | 301.2 |

Code fell 83.8 percent and prose fell 63.2 percent. Two causes remain. The
`_cast_kv_tile` mode 2 branch still does a plain cast, so int8 bytes reach the dot
product as bf16 values without dequantization. The kernel also loads and multiplies
per-tile scales on every iteration, which collapses the MTP acceptance rate. The
int8 axis is not recoverable in this fork.

## 2. Draft embedding pointer

`peer_worker.load_model` moves the draft embedding to CPU explicitly. This run added
`QWEN38_DRAFT_EMBED_SHARE=1` to skip that move and bind the target embedding after
the runner exists (`_bind_pending_draft_embed` in `peer_inventory`).

| Configuration | code C=1 | prose C=1 | load s | status |
|---|---:|---:|---:|---|
| step 1 `sep-c2s48` | **122.96** | **50.58** | 353.4 | failed (prefill samples short) |
| draft embed `sep-c2e` | 115.84 | 48.13 | 361.5 | failed |

Code was 94.2 percent of step 1 and prose was 95.2 percent. The prefill job then
exhausted the RTX 3090: `Tried to allocate 128.00 MiB`, `23.50 GiB memory in use`.
The engine died and `restoration_verified` is false. Keeping the embedding resident
on the 24 GiB owner does not fit the VRAM budget.

## 3. Layer 48 routed experts in INT4

The published checkpoint already carries `quantization_config` with
`quant_method: gptq`, `bits: 4`, `group_size: 128` and a dynamic rule that excludes
only attention, shared experts, PLE, embeddings and gates. Layer 48 routed experts
are therefore already INT4, so this path needs no conversion and is dropped.

## 4. YaRN factor 1.35 views

`build_135_views.py` writes `/work/target-135` and `/work/draft-135` as symlink views
over the same weights with `factor: 1.35`, `max_position_embeddings: 340000`.

Note the mount contract: `run_engine_hostnet.sh` binds `$BASE` to `/work`, so views
must be created under `$BASE`, not under the host `/work` directory.

| Configuration | code C=1 | prose C=1 | admission | load s |
|---|---:|---:|---|---:|
| step 1 `sep-c2s48` | **122.96** | **50.58** | 677,901, 1.99x | 353.4 |
| YaRN 1.35 `sep-c2y35` | 96.18 | 49.70 | 677,901, 1.99x | 340.3 |

Code fell 21.8 percent and prose fell 1.7 percent. The engine then died during
prefill with a refused connection and `restoration_verified` is false. The 1.35 factor
produced a measured throughput drop and a prefill failure; the run does not establish why.

## 5. Single-stream QSA fast path

`qwen38_ops_qsa_fast.py` sets `target_splits = 1` when `q.shape[0] <= 16`, which is
the MTP draft batch bound for one stream. `num_splits == 1` compiles out the fp32
partial buffers and the merge kernel. C=2 doubles the row count and keeps the split
path.

| Configuration | code C=1 | prose C=1 | admission | load s |
|---|---:|---:|---|---:|
| step 1 `sep-c2s48` | **122.96** | **50.58** | 677,901, 1.99x | 353.4 |
| single stream `sep-c2f` | 106.93 | 49.62 | 677,901, 1.99x | 341.4 |

Code fell 13.0 percent and prose fell 1.9 percent, so split parallelism is worth more
than the merge overhead it costs. The prefill job again exhausted the RTX 3090 with a
128 MiB allocation.

## 6. Adopted combination

All four new paths are reverts, so the adopted combination stays `sep-c2s48.sh` for
code (draft depth 15) and `sep-c2m10.sh` for prose (draft depth 10). That is code
122.96 and prose 54.39, still short of the 180 and 70 targets. The remaining gap is
the `block_n` halving and the MTP acceptance rate.

## 7. Concurrent decode at the long window

The 409,600 window admits `427,468 tokens, 1.04x`, so two full requests do not fit. To
test concurrency at that window, `sep-g` (`sep-f` with `--max-num-seqs 4`) was run at C=2
and C=4 with three repetitions.

| Case | Workload | C | per request | aggregate | streams failed |
|---|---|---:|---:|---:|---:|
| C=2 | code | 2 | 149.00 | **260.81** | 0 |
| C=2 | prose | 2 | 44.17 | **86.60** | 0 |
| C=4 | code | 4 | 91.82 | **302.98** | 0 |
| C=4 | prose | 4 | 30.53 | **115.28** | 0 |

Important limit: the sparkDash native runner treats `contextLength` as prefill-only
(`runner.mjs:222`), so these decode rows use the runner's fixed short prompt, about 67
input tokens with 400 forced output tokens (`totalPrefillTokens 133` for a two-stream
job). They measure concurrency, not 200,000-token or 100,000-token prompts. The
aggregate rates must not be read as evidence that splitting a 400k window raises
throughput; that claim is withdrawn.

Prefill on the same engine: 2,814.48 at 4K, 2,698.71 at 16K, 2,621.51 at 32K.

The measurement record is `status completed`, `failures []`, `restoration_verified`
true, 13 jobs.

## Reproduction

Scripts live in the remote workspace:

- `sep-c2i8s.sh` with `sep-c2i8s_driver.py`
- `sep-c2e.sh` with `sep-c2e_driver.py`
- `sep-c2y35.sh` with `sep-c2y35_driver.py`
- `sep-c2f.sh` with `sep-c2f_driver.py`
- `sep-g.sh` with `sep-g_driver.py` (split 200k x 2 and 100k x 4)

Views are built by `build_135_views.py` run inside the engine image with
`-v $BASE:/work -v /srv/vault:/srv/vault`.

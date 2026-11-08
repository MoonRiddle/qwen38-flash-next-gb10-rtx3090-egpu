# 6. Results

All numbers come from the pinned `sparkDash` collectors described in `01-measurement-method.md`. Each
table cites the artifact files it was read from; those files live under `artifacts/` in the same shape
they were produced (one immutable envelope per job, plus the matrix execution record).

Provenance separation rule: only envelopes produced by the pinned harness are ranked here. Earlier
campaign numbers from other measurement protocols are history and are not mixed into these tables.

## Hardware

| Component | Specification |
|---|---|
| Compute | GB10, SM121, 121.7 GiB unified memory |
| Second GPU | NVIDIA RTX 3090, SM86, 24 GiB GDDR6X, BF16 unlock (Zagitta/gpu-unlock), PCIe Gen4 x4, non-P2P |
| PLE storage | 51 GiB n-gram table mmapped from USB; 10 Gbps T7 Shield or 20 Gbps NVMe, byte-identical copies |
| Engine | fork build `qwen38-flash-dgx:a5b-autoround-bottleneck-q2`, image id `sha256:b27377b7c25cc620e0b863dbbb9b81d9fe89cdf13d36a4cea6fb6b7dd8747363` |
| Checkpoint | `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`, 512 experts retained, top-5 routing, 4.8B active |
| Draft head | `draft-k10-int4`, BF16 top-10 |

## Reproduction verdict (Step 2)

The prior same-method run was `a-sparkdash-native-r2` (2026-11-04, same image, same engine flags:
`--mtp 15 --max-model-len 17408 --max-num-seqs 1 --kv-mib 4096 --block-size 160
--max-num-batched-tokens 8192`, PLE from `/mnt/nvme20/ple-table-int4`, pre-warm on). The new run is
`a-sparkdash-native-r4` with the identical schedule and controls.

| Workload | Prior median (r2) | New median (r4) | Band | Verdict |
|---|---:|---:|---|---|
| prose, C=1, 400 tok | 71.06 | 46.30 | ±20% | **FAIL** (−34.8%) |
| code, C=1, 400 tok | 207.46 | 211.77 | ±10% | **PASS** (+2.1%) |
| prefill 4K | 3,177.70 | 2,904.65 | ±10% | **PASS** (−8.6%) |
| prefill 16K | 3,051.03 | 3,099.69 | ±10% | **PASS** (+1.6%) |

Raw samples (tok/s), the full three-run curve rather than a median only:

| Run | r2 prose | r4 prose | r2 code | r4 code |
|---|---:|---:|---:|---:|
| 01 | 24.03 | 9.17 | 142.91 | 154.51 |
| 02 | 71.06 | 43.81 | 207.46 | 243.23 |
| 03 | 71.30 | 46.30 | 226.77 | 211.77 |

Prefill rows, all with `samples == samplesRequested == 3` and `cachedTokens == 0`:

| Config | Size | r2 tok/s | r3 tok/s | r4 tok/s |
|---|---:|---:|---:|---:|
| `k6-open` | 4K | 3,177.70 | 3,268.13 | 2,904.65 |
| `k6-open` | 16K | 3,051.03 | 3,057.57 | 3,099.69 |
| `k6-top2-conf50` | 4K | 3,240.37 | 3,267.62 | 3,219.45 |
| `k6-top2-conf50` | 16K | 3,052.55 | 3,049.81 | 3,044.67 |

### Why prose failed to reproduce

The engine log for the r4 run records Triton JIT compilation **during inference**, and the timestamps
line up with the prose jobs:

```
17:01:34  _qsa_mqa_paged_kernel
17:03:49  _compute_local_logits_stats_kernel
17:03:49  _rejection_kernel
17:03:49  _resample_kernel
17:03:49  _gumbel_sample_kernel
17:04:29  _qsa_sparse_paged_gqa_splitk_kernel
17:04:33  _qsa_pre_indexer_kernel
17:04:41  _expand_qsa_indices_kernel
```

The prose jobs ran at 17:01–17:03; the code jobs ran after the last warning at 17:04 and reproduced
the prior numbers almost exactly. The engine runs with `VLLM_DISABLE_COMPILE_CACHE=1`, so each new
container recompiles whatever shapes the first workload exercises. The prior run had already paid
that cost in earlier containers.

To test this directly, the same engine was re-measured on prose only after the kernels were compiled,
under a separate run prefix so the verdict above stays intact:

| Diagnostic run (`a-sparkdash-native-r4warm`) | prose C=1, 400 tok |
|---|---:|
| 01 | 95.78 |
| 02 | 87.35 |
| 03 | 82.50 |

So the prose rate recovers past the prior median once the JIT penalty is gone. The reproduction FAIL is
a cold-start artifact of the measurement order, not a regression in the recipe — and it is exactly the
Cold-2 penalty described in `03-cold-start-analysis.md`, whose fix (boot-time pre-bake) is present in
the engine but not enabled by the launcher used here.

Artifacts: `artifacts/reproduction/a-sparkdash-native-r4-execution.json`,
`artifacts/reproduction/a-sparkdash-native-r4-k6-open-prose-0{1,2,3}.json`,
`artifacts/reproduction/a-sparkdash-native-r4warm-*.json`,
`artifacts/reproduction/a-sparkdash-native-r2-*.json`.

## Recipe concurrency grid (Step 3)

Engine: `--max-model-len 17408 --max-num-seqs 4 --kv-mib 12288 --block-size 160
--max-num-batched-tokens 8192`, PLE from `/mnt/nvme20/ple-table-int4`, controls `k6-open`.
Measured admission: `GPU KV cache size: 61,740 tokens, Maximum concurrency for 17,408 tokens per
request: 3.55x`, `engine_load_seconds 296.17`.

Decode, 400 forced tokens, three independent jobs per workload, each job carrying C=1, 2 and 4 in one
envelope. `median` is the per-request median; `aggregate` is the single shared decode window across all
streams.

### Prose (temp 0.7, thinking on)

| C | run 01 | run 02 | run 03 | median of medians | aggregate median | streams ok / failed |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 80.00 | 95.57 | 75.92 | **80.00** | 80.00 | 3 / 0 |
| 2 | 65.22 | 64.01 | 72.72 | 65.22 | 126.00 | 6 / 0 |
| 4 | 47.55 | 52.31 | 52.38 | 52.31 | 191.94 | 12 / 0 |

### Code (temp 0.0, thinking off)

| C | run 01 | run 02 | run 03 | median of medians | aggregate median | streams ok / failed |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 211.98 | 206.01 | 233.77 | **211.98** | 211.98 | 3 / 0 |
| 2 | 144.01 | 152.82 | 147.73 | 147.73 | 268.52 | 6 / 0 |
| 4 | 95.69 | 95.48 | 94.78 | 95.48 | 316.01 | 12 / 0 |

Reading these two tables together:

- Per request, throughput falls as concurrency rises — that is expected, because four streams share one
  decode pipeline. C=1 prose 80.00 → C=4 prose 52.31 (−35%), C=1 code 211.98 → C=4 code 95.48 (−55%).
- Aggregate throughput rises with concurrency: prose 80.00 → 191.94 (+140%), code 211.98 → 316.01
  (+49%). So C=4 is worth using when total tokens per second matters, and C=1 is worth using when a
  single request's latency matters.
- Every C=4 row has `streamsOk == 4`, `streamsFailed == 0`, so the concurrency path is real and not a
  queued C=1 wave.

### Prefill at KV 12 GiB: not servable

The prefill job failed on all three sizes, each with `samples == 0 / 3` and an HTTP 500 carrying the
engine's own root cause:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 82.00 MiB.
GPU 0 has a total capacity of 23.61 GiB of which 56.19 MiB is free.
```

That is the RTX 3090, not the GB10. After six decode jobs at KV 12 GiB the card has no room left for a
prefill activation, so the engine core dies. Prefill therefore needs its own engine with a smaller KV
reservation; the measurement is in the next section.

Artifacts: `artifacts/recipe/a-sparkdash-native-r5f-execution.json`,
`artifacts/recipe/a-sparkdash-native-r5f-k6-open-{prose,code}-0{1,2,3}.json`,
`artifacts/recipe/a-sparkdash-native-r5f-k6-open-prefill-01.json` (the failure, kept as evidence).

## Prefill (KV 8 GiB engine)

Engine: `--max-model-len 17408 --max-num-seqs 4 --kv-mib 8192 --block-size 160
--max-num-batched-tokens 8192`, same image, same PLE copy, controls `k6-open`. Measured admission:
`GPU KV cache size: 41,082 tokens, Maximum concurrency for 17,408 tokens per request: 2.36x`,
`engine_load_seconds 287.25`.

Three independent jobs, each running all three sizes with `repeatsForSize` samples (3 each):

| Size | run 01 | run 02 | run 03 | median | median TTFT | samples | cachedTokens | lowConfidence |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 4096 | 3,242.96 | 3,076.53 | 3,133.27 | **3,133.27** | 1,441.8 ms | 3/3 | 0 | False |
| 8192 | 3,136.15 | 3,079.65 | 2,488.36 | **3,079.65** | 2,796.4 ms | 3/3 | 0 | False |
| 16384 | 3,081.53 | 3,043.94 | 3,064.22 | **3,064.22** | 5,497.2 ms | 3/3 | 0 | False |

Every row has `cachedTokens == 0` (the salted prompt defeats prefix caching) and
`samples == samplesRequested`. The 8K run-03 outlier (2,488.36) is kept in the table; the median is
reported, not the best sample.

So prefill scales almost flat with size: 3,133 → 3,064 tok/s from 4K to 16K. The shared compute path is
not the reason prose decode is slow; that is draft acceptance (see `02-why-prose-is-slower.md`).

Artifacts: `artifacts/recipe/a-sparkdash-native-r5g-execution.json`,
`artifacts/recipe/a-sparkdash-native-r5g-k6-open-prefill-0{1,2,3}.json`.

## Verification summary

Ported TensorFold checks on the KV 8 GiB engine (`docs/05-verification.md` has the reasoning):

| Check | Result |
|---|---|
| needle at 16,384 tokens, 60% depth | PASS — found the passphrase in 8.5 s |
| context boundary at 17,408 | PASS — 16 checks, streamed and non-streamed, drafts on and off |
| exactness serial vs concurrent vs draft-off | FAIL — greedy replies differ across lanes; `VLLM_MARLIN_USE_ATOMIC_ADD=1` makes reductions order-dependent |
| prompt reuse across turns | FAIL — 0.0% cached; the engine serves with prefix caching off |
| tool-call stability | could not run — engine lacks `--enable-auto-tool-choice` / `--tool-call-parser` |
| cross-build exactness | UNCHECKED — `--mtp 0` is rejected by `SpeculativeConfig` |
| vision | not applicable — text-only recipe |

Nothing here is upgraded to a pass. The two FAILs are properties of the recipe as configured, and the
two non-runs are configuration gaps, all stated rather than smoothed over.

Artifacts: `artifacts/verify/results.json` and the per-tool logs beside it.

## Why prefill is a separate engine configuration

Prefill is sequential in this harness (`PrefillBench` has no concurrency list), so every prefill row is
C=1. It still OOMs when the KV reservation is large, because prefill processes thousands of tokens per
iteration and its activation/MoE workspace sits outside the KV reservation. Decode processes roughly one
token per running sequence, so decode fits in spaces prefill does not.

The prior same-method runs and this session's runs line up into a clear pattern:

| Run | `--kv-mib` | `--max-num-seqs` | `--max-num-batched-tokens` | Prefill outcome |
|---|---:|---:|---:|---|
| r1, r2, r3, r4 | 4096 | 1 | 8192 | 4K and 16K succeeded, 14 jobs each, `samples 3/3` |
| r5g (this session) | 8192 | 4 | 8192 | 4K, 8K, 16K succeeded at ~3,100 tok/s |
| r6b (this session) | 8192 | 4 | 8192 | 4K failed, HTTP 500 |
| r5f (this session) | 12288 | 4 | 8192 | 4K failed, HTTP 500 |
| r7c (this session) | 13312 | 1 | 8192 | 32K failed, HTTP 500 |
| r7d (this session) | 8192 | 1 | 8192 | 32K failed, `samples short` |
| r7e (this session) | 8192 | 1 | 1024 | 32K skipped: prompt + template + reply exceeds a 32,768 window |
| r7f (this session) | 8192 | 1 | 1024 | 32K skipped: still exceeds a 34,816 window |
| r7g (this session) | 8192 | 1 | 1024 | 32K attempted at a 40,960 window; completed_with_failures, 2 failures |

A harness detail worth stating: `PrefillBench` decides whether a size fits from `job.config.contextLength`,
not from the server. A driver that hardcodes `contextLength` at the served window will have larger sizes
skipped with `Skipped: … plus template and reply does not fit the model context` even when the engine
itself was started with a larger `--max-model-len`. Pass the engine's actual `max_model_len` into
`contextLength` and the same size runs.

Two conclusions follow.

1. **Decode and prefill need different KV reservations.** Decode at C=4 with a 17,408-token window
   wants about 12 GiB of KV on the 3090; prefill at 4K–16K is reliable at 4–8 GiB. Reserving for decode
   leaves too little room for prefill activations.
2. **The 8 GiB result is not stable across containers.** r5g served prefill and r6b did not, at the same
   flags. That is history-dependent: whatever persistent allocation a prior decode job leaves behind
   decides whether the next prefill activation fits. So a "prefill works at 8 GiB" claim cannot be made
   from one container.

The practical recipe is therefore two engine configurations: one for decode at C=4 with a large KV, and
one for prefill with a small KV. That is what this repository measures, rather than pretending one
container serves both.

Artifacts: the prefill envelopes cited in each run's own section above, plus the failing envelopes kept
as evidence (`artifacts/recipe/a-sparkdash-native-r5f-k6-open-prefill-01.json`,
`artifacts/storage/a-sparkdash-native-r6b-k6-open-prefill-01.json`).

## Long prefill (32K)

Engine: `--max-model-len 40960 --max-num-seqs 1 --kv-mib 8192 --block-size 160
--max-num-batched-tokens 1024`, same image, same PLE copy, controls `k6-open`. Admission:
`GPU KV cache size: 81,459 tokens, Maximum concurrency for 40,960 tokens per request: 1.99x`,
`engine_load_seconds 240.25`.

| Size | tok/s | TTFT | samples | cachedTokens | lowConfidence |
|---:|---:|---:|---|---|---|
| 32,768 | **2,443.12** | 13,876.1 ms | 3/3 | 0 | False |

The 32K rate is lower than the 4K–16K rates (3,133 / 3,079 / 3,064 tok/s) because a 32,768-token prompt is
processed in 1024-token chunks, which costs some kernel efficiency but keeps the activation workspace
small enough to fit the card.

**64K and 128K are not served by this recipe on a 24 GiB card.** One 65,536-token request needs about
12.74 GiB of KV and one 131,072-token request needs about 25.48 GiB, both beyond what the 3090 can hold
alongside the weights, graphs and workspace. Those rows are recorded as dropped, not measured. A second
node, or a smaller per-request window, is the only way to serve them here.

Artifacts: `artifacts/longprefill/a-sparkdash-native-r7h-execution.json`,
`artifacts/longprefill/a-sparkdash-native-r7h-k6-open-prefill-01.json`.

## Why this recipe's window is narrower than the model author's own recipe

The same checkpoint has a full serving recipe from its author
([`azampatti/Qwen3.8-Flash-Next-Int4-FAST`](https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST)). Comparing the
two is the fastest way to see which limits here are real and which are self-imposed.

| Setting | Author's solo recipe | This recipe |
|---|---|---|
| `block_size` | **1632** | 160 |
| `num_speculative_tokens` | **5** | 15 |
| `max_model_len` | **262144** | 17408 |
| KV bytes | **20g** | 12 GiB |
| KV dtype | auto, fp8 optional | bfloat16 |
| prefix caching | **on** | off |
| tool parser | `qwen3_coder` | none |
| chat template | `medium` (+5pp acceptance) | model default |
| draft scale | 2 (+2.5pp acceptance) | 1 |

Three of these explain the window gap.

1. **`block_size` alignment.** The author's recipe sets `block_size: 1632` with the note that the QSA ring is 12 rows and
   must divide the page (`1632 = 12 × 136 = 8 × 204`). This recipe uses 160, which is not a multiple of 12, so the QSA
   ring cannot align to a page boundary. The ring depth is not fixed at 12: it follows the speculative depth. Measured
   on this machine, `--mtp 15` reports `AssertionError: QSA ring capacity 20 must divide the attention block size 3264`,
   so with 15 draft tokens the ring is 20 rows and `block_size` must be a multiple of 20 (1600, 3200). The author's
   1632 is valid only at `--mtp 5`, where the ring is 12 rows. A 1600-token page at MTP 15 admits 35,796 tokens at
   12 GiB, fewer than the 61,740 admitted by a 160-token page, so a larger page costs capacity at short windows.
2. **MTP depth.** The author's measured sweep is 3 → 71.2, 4 → 76.0, 5–7 → 79.2 tok/s, with 5 the everyday setting.
   This recipe runs 15, outside that optimum.
3. **Where KV lives.** The author's 20g reservation holds roughly 645k tokens because it sits in the GB10's 121.7 GiB
   unified pool. This recipe pins KV to the 24 GiB RTX 3090 and gets 61,740 tokens — about 6.3x fewer.

Measured on this machine: starting this recipe's own engine with `--block-size 1632 --max-model-len 262144
--kv-mib 12288 --max-num-seqs 1 --max-num-batched-tokens 1024 --mtp 5` reports
`GPU KV cache size: 404,491 tokens, Maximum concurrency for 262,144 tokens per request: 1.54x` and loads in
227 s. So the alignment change alone lifts this recipe's KV from 61,740 to 404,491 tokens at the same 12 GiB
— 6.5x, matching the arithmetic above. A 256K window is admitted, not just projected.

So the 17,408 window is not a hardware limit of the model. It is the product of KV placement and page alignment. Both are
addressable, and the measurement of the alternative is the open item on this recipe's roadmap rather than a settled
conclusion.

### Page size costs decode, measured in isolation

Holding MTP at 15 and the window at 17,408, changing only `block_size` from 160 to 1600 halves cold decode:

| Configuration | block_size | MTP | window | cold code C=1 | cold prose C=1 | admission |
|---|---|---|---|---|---|---|
| r4 | 160 | 15 | 17,408 | 154.51 | 9.17 | 61,740 tokens |
| sep-a3 | 1600 | 15 | 17,408 | 73.22 | 7.85 | 35,796 tokens |
| r9b | 1632 | 5 | 262,144 | 75.92 | 8.92 | 404,491 tokens |

The 1600 and 1632 rows agree within noise, so the regression is the page, not the speculative depth or the
window. A 400-token generation occupies three 160-token pages (480 rows) but one 1600-token page (1,600 rows);
if the kernel processes allocated rows, the last step carries 3.3x the work. That matches the measured 2.1x
cold-to-cold ratio.

Two consequences. A wide window costs roughly two thirds of decode throughput on this box, so 200K context and
200 tok/s code decode are not simultaneously available with a large page. And 160 is a multiple of 20, so it is
already aligned to the MTP-15 ring; the alignment rule does not force a large page at MTP 15.

## 256K window (block 1632)

Engine: `--max-model-len 262144 --block-size 1632 --kv-mib 12288 --max-num-seqs 1
--max-num-batched-tokens 1024 --mtp 5`, same image, same PLE copy. Admission:
`GPU KV cache size: 404,491 tokens, Maximum concurrency for 262,144 tokens per request: 1.54x`,
`engine_load_seconds 227.14`.

| Workload | C | median tok/s | aggregate tok/s |
|---|---:|---:|---:|
| code | 1 | 75.92 | 75.92 |
| code | 2 | 87.93 | 84.25 |
| prose | 1 | 8.92 | 8.92 |
| prose | 2 | 22.99 | 20.80 |

These are cold-engine numbers from a fresh container. The JIT penalty documented in the reproduction
section applies in full, and the prose cells are the first prose work this engine ever did. They are not
comparable to the warm tables above, and they are recorded because they are what a first request actually
returns.

Prefill at 262,144 was skipped with `Skipped: 256k plus template and reply does not fit` because
`max_model_len` is exactly 262,144, so the prompt plus template plus reply exceeds it. A 200K prompt fits
inside that window; that measurement is recorded below.

### 200K prefill

Engine: same 256K configuration (`--max-model-len 262144 --block-size 1632 --kv-mib 12288
--max-num-seqs 1 --max-num-batched-tokens 1024 --mtp 5`).

| Size | tok/s | TTFT | samples | cachedTokens | lowConfidence |
|---:|---:|---:|---|---|---|
| 204,800 | **2,035.64** | 100,754.6 ms | 1/1 | 0 | False |

A 204,800-token prompt was processed in 1024-token chunks and answered. So the earlier "32K is the
prefill limit" reading was wrong in its cause: the limit was KV capacity, not the ability to process
long input. With 404,491 tokens of KV the same chunked prefill path handles 200K, at about 2,036 tok/s
and a 101-second time to first token.

Artifacts: `artifacts/recipe/a-sparkdash-native-r9c-execution.json`,
`artifacts/recipe/a-sparkdash-native-r9c-k6-open-prefill-01.json`.

## 204,800 window at block 160 (sep-c)

Engine: `--max-model-len 204800 --max-num-seqs 1 --kv-mib 12288 --block-size 160 --max-num-batched-tokens 2048
--mtp 15`, CUDA graphs `NONE`.

Admission: `GPU KV cache size: 299,323 tokens, Maximum concurrency for 204,800 tokens per request: 1.46x`,
`engine_load_seconds 226.27`, and no OOM during load.

Decode (sparkDash k6-open, C=1):

| Workload | Run 1 | Run 2 | Notes |
|---|---|---|---|
| code | 192.28 | — | cold, first code job |
| prose | 24.31 | 76.12 / 68.92 | first prose is cold; warm runs recover |

Prefill (same engine, batch 2048):

| Size | Result |
|---|---|
| 4K | 3,073.23 tok/s |
| 16K | TTFT unmeasured |
| 32K | HTTP 500, engine core OOM |
| 200K | fetch failed |

So a 204,800 window at a 160-token page keeps code decode above 190 tok/s and warm prose above 68 tok/s, which
is about 2.5x the author's 70-75 tok/s.

Prefill at `--max-num-batched-tokens 2048` fits the same engine and clears 2,000 tok/s at every size:

| Size | promptTokens | samples | prefillTps | TTFT |
|---|---:|---:|---:|---:|
| 4K | 4,136 | 3 | 2,844.53 | 1.56 s |
| 16K | 16,422 | 3 | 2,727.28 | 6.13 s |
| 32K | 32,806 | 3 | 2,657.05 | 12.45 s |
| 200K | 200,039 | 1 | 2,345.07 | 85.41 s |

At 8192 batched tokens the prefill workspace does not fit beside the KV cache and the engine core dies at 32K;
at 2048 it fits and every size completes. The trade is fewer tokens per chunk, which costs a few percent of
prefill rate but keeps the engine alive.

Artifacts: `artifacts/recipe/a-sep-c-execution.json`, `artifacts/recipe/a-sep-c-k6-open-{prose,code}-01.json`,
`artifacts/recipe/a-sep-c-warm-execution.json`, `artifacts/recipe/a-sep-c-prefill-execution.json`.

## 262,144 window (model maximum, sep-c)

The checkpoint's `max_position_embeddings` is 262,144, so that is the hard ceiling for any window; a larger
`--max-model-len` is rejected at startup:

```
Value error, User-specified max_model_len (348800) is greater than the derived max_model_len
(max_position_embeddings=262144.0)
```

Engine: `--max-model-len 262144 --max-num-seqs 1 --kv-mib 14336 --block-size 160 --max-num-batched-tokens 2048
--mtp 15`, CUDA graphs `NONE`.

Admission: `GPU KV cache size: 380,966 tokens, Maximum concurrency for 262,144 tokens per request: 1.45x`,
`engine_load_seconds 218.27`. The 3090 sits at 22,913 of 24,576 MiB (93%) and host RAM at 69.7 of 121.7 GiB,
so both stay inside their limits without a guard trip.

Decode (C=1): code 194.70 cold, prose 33.18 cold then 76.77 / 73.61 warm.

Prefill at batch 2048:

| Size | promptTokens | samples | prefillTps | TTFT |
|---|---:|---:|---:|---:|
| 4K | 4,136 | 3 | 2,814.22 | 1.58 s |
| 16K | 16,423 | 3 | 2,723.52 | 6.14 s |
| 32K | 32,805 | 3 | 2,649.84 | 12.49 s |
| 200K | 200,041 | 1 | 2,331.81 | 85.90 s |
| 261K | 261,036 | 1 | 2,240.99 | 116.59 s |

Going from 204,800 to 262,144 costs nothing measurable: decode and prefill sit within run-to-run noise of the
204,800 figures. The extra KV comes from the 2.5 GiB the 12 GiB configuration left free plus the 1 GiB the
smaller prefill batch released.

### 3090 memory split

| Component | Bytes | GiB |
|---|---|---:|
| target weights on the 3090 | 6,646,993,920 | 6.19 |
| draft weights on the 3090 | 9,757,622,784 | 9.09 |
| KV reservation at 14 GiB | 15,032,385,536 | 14.00 |
| measured 3090 total | 24,034,312,192 | 22.38 |

The arithmetic does not add up to the measured total because vLLM releases part of the weight staging buffer
after load, so the reservation figure is an upper bound rather than a sum. The usable headroom on the card at
262,144 is 1.6 GiB.

### Forcing 409,600: admitted but not servable

With `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` the engine accepts `--max-model-len 409600` and reports
`GPU KV cache size: 481,483 tokens, Maximum concurrency for 409,600 tokens per request: 1.18x` at `--kv-mib 16384`.
It then dies during start-up with `memory allocation failed with OOM on device 0` (1.28 GB request, 44 MB free),
because the three residents do not fit together:

| Component | GiB |
|---|---:|
| draft weights on the 3090 | 9.09 |
| target weights on the 3090 | 6.19 |
| KV at 16 GiB | 16.00 |
| sum | 31.28 |

The card holds 24 GiB. Two escape routes were tried and both fail:

1. **Move the draft to the GB10.** `peer_worker.py` was changed so the draft loads on the helper device, but the
   measured placement still reports `primary_bytes 9,757,622,784` on device 0: vLLM's worker allocates KV and
   weights against device 0 regardless of the load device, so the change has no effect.
2. **Enable prefix caching.** Without it, `--mamba-block-size` is rejected when the window exceeds the model
   maximum; with `--enable-prefix-caching` the engine gets past that check but still hits the same OOM.

So 409,600 is admitted on paper and refused in practice on a 24 GiB card. The window stays at 262,144, where the
card keeps 1.6 GiB of headroom and every measurement above was taken.

### Swapping device 0 to the GB10

The third escape route is to put the GB10 first in `CUDA_VISIBLE_DEVICES`, so vLLM's device 0 becomes the
unified-memory device and KV lands there. `QWEN38_VISIBLE_DEVICES="<GB10>,<3090>"`, `QWEN38_OWNER=1`,
`QWEN38_HELPER=0` and `QWEN38_SKIP_DEVICE_CHECK=1` get past the role assertions and the engine reaches model load.
It then dies in two stages:

1. `Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on
   cuda:1`, raised from `vocab_parallel_embedding.py`'s `embedding`. The trunk move puts the embedding table on
   the owner, while the input ids are created on device 0 (the GB10), so the two disagree.
2. Moving `VocabParallelEmbedding` to the helper clears that, but Triton then rejects the table with
   `Pointer argument cannot be accessed from Triton (cpu tensor?)`. The GB10's unified memory is not reachable as
   a device pointer from the Triton kernels this build uses.
3. Wrapping the embedding forward in the input tensor's own device context (`torch.cuda.device(input_.device)`) and
   leaving the table on the 3090 gets past Triton, but the very first `index_select` then fails with
   `Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on
   cuda:1`. The ids are produced on device 0 before the embedding sees them, so the device split has to be fixed
   upstream rather than in this call site.

So the GB10 cannot host the embedding table, and without it the device-0 swap cannot complete. Restoring the
default device order and the 14 GiB KV keeps the working 262,144 configuration.

## 409,600 window (config extension, sep-e and sep-f)

The 409,600 window was reached without moving KV off the card. The checkpoint's `config.json` had no top-level
`max_position_embeddings`, and `text_config` carried 262144 with `rope_type: default`. A wrapper view
(`target-409k`, `draft-409k`) was built with symlinks to the read-only checkpoint and a patched `config.json`:

```
max_position_embeddings: 409600            (top level and text_config)
rope_type: yarn, factor: 1.5625            (= 409600 / 262144)
original_max_position_embeddings: 262144
mrope_interleaved: true, mrope_section: [11, 11, 10]
partial_rotary_factor: 0.25, rope_theta: 10000000
```

The fork also needed one code fix. `standardize_rope_params()` reads `self.max_position_embeddings` before
`PretrainedConfig.__init__` runs, so `Qwen3_8FlashNextConfig.__init__` now sets it from `kwargs` first. That module
is bind-mounted read-only into the engine.

Engine: `--max-model-len 409600 --max-num-seqs 1 --kv-mib 14336 --block-size 160 --max-num-batched-tokens 2048
--mtp 15 --graphs NONE`.

| Attempt | KV | Admission | Result |
|---|---:|---|---|
| first force (VLLM_ALLOW_LONG_MAX_MODEL_LEN) | 16 GiB | `481,483` / 1.18x | OOM in start-up peak |
| config extension | 14 GiB | `427,468` / 1.04x | works, load 297.25 s |
| MTP experts int4 | 17 GiB | `518,185` / 1.27x | OOM (1.27 GiB request, 116 MiB free) |
| MTP experts int4 | 16 GiB | `487,946` / 1.19x | OOM (1.19 GiB request, 1,004 MiB free) |
| MTP experts int4 | 14 GiB | `427,468` / 1.04x | works, load 305.18 s |

Decode at 409,600:

| Workload | config extension (14 GiB) | MTP int4 (14 GiB) |
|---|---:|---:|
| code, median | 181.78 | 185.89 |
| prose, median | 67.28 | 70.93 |

Prefill at 409,600:

| Size | config extension | MTP int4 |
|---|---:|---:|
| 4K | 2,753.36 | 2,813.41 |
| 16K | 2,688.15 | 2,699.81 |
| 32K | 2,623.35 | 2,629.04 |

Against 262,144 the window grows 56% for about 6.6% decode and 1-2% prefill. A prompt of 262,209 tokens crossed
the native boundary without the MTP RoPE crash (`ok: true`, 16 tokens, `finish_reason: length`).

An 8-bit KV cache was also tried. `fp8_e4m3` reached `677,415` tokens at 1.65x and was then rejected by Triton
(`type fp8e4nv not supported in this architecture`). The `fp8_e5m2` path was initially rejected by FlashAttention
(`does not support fp8_e5m2 kv-cache on this device`), but that guard was reached because the QSA fork never
neutralised `fp8_e5m2` for the parent `FlashAttentionImpl.__init__`. After patching the dtype tuple, the uint8
view branch, and the storage checks, `fp8_e5m2` runs: `--kv-mib 14336` gives 677,415 tokens at 1.65x, and
`--kv-mib 15616` gives 738,180 tokens at 1.80x with `--max-num-seqs 2`. The 17 GiB pool admits 823,701 tokens
at 2.01x but fails the KV tensor allocation peak on a single 1.23 GiB request.

### Concurrency 2 at 340,000

Two full 409,600 requests do not fit. The 14 GiB pool admits `427,468 tokens, 1.04x`,
while two requests need 819,200 tokens of raw context. Lowering `--max-model-len` to
340,000 with `--kv-mib 15360` does reach ready (`677,901` tokens, `1.99x`, load 311.25 s)
and serves two streams:

| Case | C=1 | C=2 per request | C=2 aggregate | Failed streams |
|---|---:|---:|---:|---:|
| code 1 | 114.11 | 86.79 | 163.46 | 0 |
| code 2 | 108.13 | 83.83 | 156.11 | 0 |
| code 3 | 115.32 | 90.60 | 177.42 | 0 |
| prose 1 | 34.74 | 37.70 | 73.25 | 0 |
| prose 2 | 45.85 | 41.87 | 78.35 | 0 |
| prose 3 | 59.28 | 40.16 | 79.66 | 0 |

Prefill needs `--max-num-batched-tokens 1024` and `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32` to survive the peak,
but on this pair the prefill job failed at every size: the `a-sep-c2` record returned
`prefillTps 0` and `samples 0` (`fetch failed`), and `a-sep-c2s48` returned HTTP 500. No
prefill rate is reported for the 340,000 window.

Deeper prefill (131,072 and above) still fails: the indexer workspace asks for 128 MB with about 92 MiB free.
Concurrency 2 and deep prefill pull in opposite directions on this card.

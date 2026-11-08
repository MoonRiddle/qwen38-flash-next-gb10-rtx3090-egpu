# 12. Decoding regression across the context-window stages

This document exists so an outside reader can reconstruct the whole experiment without the
session logs. It records every stage of the context extension, the exact engine flags, the
measured decode and prefill rates, and the candidate causes of the decode drop. The target is
code decode at C=1 above 180 tok/s and prose decode at C=1 above 70 tok/s.

## 1. Hardware and model

| Item | Value |
|---|---|
| Model | `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` |
| Active parameters | 6.0B of 125B (top-10 experts unpruned) |
| Device 0 | NVIDIA RTX 3090, SM86, 24,176 MiB, 833 GB/s, eGPU over PCIe Gen4 x4, non-P2P |
| Device 1 | GB10, SM121, 121.7 GiB unified memory, 246 GB/s |
| Link | 33.14 us round trip, 6.2/6.6 GB/s |
| Storage | Samsung T7 Shield PSSD, 10 Gbps USB |
| Runtime | vLLM fork `0.1.dev20073`, engine image `qwen38-flash-dgx:a5b-autoround-bottleneck-q2` |
| Speculation | MTP depth 15, INT4 draft head, probabilistic draft sampling |
| KV placement | device 0 (the 3090) only |
| Measurement | MiaAI-Lab sparkDash native runner, commit `7c2b9638b025cd39fb43f95c225de62980f130ee` |

Every decode number below is the sparkDash native `code` or `prose` case with 400 forced output
tokens. Every prefill number is the sparkDash native `prefill` case. No local harness substitutes
for these.

## 2. What each stage changed

| Stage | Window | KV dtype | KV pool | Block | Batch | Seqs | Draft embed | Other |
|---|---|---:|---:|---:|---:|---:|---|---|
| Pre-extension | 204,800 | bf16 | 12 GiB (12,288) | 160 | 2048 | 1 | GPU | baseline |
| Stage 1 | 262,144 | bf16 | 14 GiB (14,336) | 160 | 2048 | 1 | GPU | native model maximum |
| Stage 2 path A | 409,600 | bf16 | 14 GiB (14,336) | 160 | 2048 | 1 | GPU | YaRN factor 1.5625, `config.json` view |
| Stage 2 path B | 409,600 | bf16 | 14 GiB (14,336) | 160 | 2048 | 1 | GPU | YaRN plus int4 draft experts |
| Stage 3 | 340,000 | fp8_e5m2 | 15 GiB (15,360) | 160 | 1024 | 2 | CPU | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32`, QSA fork patched for `fp8_e5m2` |

Stage 3 changed six things at once relative to stage 1: window, KV dtype, KV size, batch,
sequence slots, and the draft embedding device. That is why the drop cannot be attributed from
the numbers alone.

## 3. Measured results per stage

Decode, C=1, tok/s:

| Stage | Code | Prose warm (two runs) |
|---|---:|---:|
| Pre-extension (204,800) | 192.28 | 76.12 / 68.92 |
| Stage 1 (262,144) | 194.70 | 76.77 / 73.61 |
| Stage 2 path A (409,600) | 181.78 | 72.84 |
| Stage 2 path B (409,600) | 185.89 | 71.90 |
| Stage 3 (340,000, C=2 config) | 139.91 | 38.50 / 52.89 / 52.23 |

Prefill, tok/s at 4,096 tokens:

| Stage | Prefill 4K | TTFT ms |
|---|---:|---:|
| Pre-extension | 2,844.53 | 1,560 |
| Stage 1 | 2,814.22 | 1,580 |
| Stage 2 path A | 2,753.36 | not recorded |
| Stage 2 path B | 2,813.41 | not recorded |
| Stage 3 | 2,289.39 | 1,921.87 |

Stage 3 loses about 28% of code decode and about 29% of warm prose against stage 1, and about 19%
of prefill. Stage 2 cost only 6.6% of code decode and 1% of prose, so the window stretch by itself
is not the cause.

Engine load time per arm, which is also a health signal:

| Arm | Load seconds |
|---|---:|
| Pre-extension | 226.27 |
| Stage 1 | 218.27 (repeat 220.14) |
| Stage 3 | 311.25 (repeats 322.57, 353.46) |
| Embed GPU, seqs 1 | 276.17 |
| Embed CPU, seqs 1 (rejected) | 854.87 |

## 4. Candidate causes, in order of suspicion

Status as of the A/B run: cause 1 is refuted for code, cause 3 is now the leading candidate, and
causes 2, 4 and 5 are untested.

| # | Change | Why it could cost this much | How to separate it |
|---|---|---|---|
| 1 | Draft embedding moved to CPU (`peer_worker.py:762`) | MTP drafts 15 tokens per step; each step gathers the embedding on the CPU and copies the result to the 3090 over the non-P2P link. That is a per-step host-to-device transfer of a small but latency-bound tensor | **Refuted**: moving it back to the 3090 gave 105.85 code, below the 139.91 CPU arm |
| 3 | `--max-num-seqs` 1 to 2 | Reserves a second sequence slot in every batch, which can change graph capture shapes and sampler row counts | **Leading**: the seqs 1 arm collapsed to 57 code and a 32.8 s TTFT, so a single slot is not a usable configuration here |
| 2 | `--max-num-batched-tokens` 2048 to 1024 | Halves the scheduler's token budget. Decode batches are small, so this matters more for prefill than decode | Same engine, batch 2048, one variable only |
| 4 | KV dtype bf16 to fp8_e5m2 | Every attention read now dequantizes. The QSA path keeps KV as int8 storage with a float8 view, so the read path changes shape | Same engine, `--kv-cache-dtype auto` |
| 5 | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32` | Caps the indexer workspace. It is a memory guard, not a compute change, but it may force extra passes | Same engine, unset |
| 6 | Window 262,144 to 340,000 | Adds KV pages and a longer block table. Stage 2 already showed this is a 1-7% effect | Same engine, window 262,144 |

Cause 3 is the strongest candidate now because the only arm that ran slower had one sequence slot.
The other untested changes are configuration shifts that mostly affect scheduling and memory, and
stage 2 proves that a window stretch alone costs single digits.

## 5. The A/B experiment

The engine accepts `QWEN38_DRAFT_EMBED_DEVICE=gpu` (default `cpu`). Everything else is identical
to stage 3 except `--max-num-seqs 1` and `--kv-mib 14176`, because the embedding table returns to
the 3090 and the pool must give back 1.19 GiB.

```
export QWEN38_DRAFT_EMBED_DEVICE=gpu
--max-model-len 340000 --max-num-seqs 1 --kv-mib 14176 --kv-cache-dtype fp8_e5m2 \
--block-size 160 --max-num-batched-tokens 1024
```

Result: code fell to a 105.85 median (runs 105.85 / 101.59 / 114.13) and prefill fell to
2,162.97 tok/s, so moving the embedding back to the card did not recover decode. The hypothesis
is refuted for code. The A/B arm also changed `--max-num-seqs` from 2 to 1, so the sequence-slot
effect is not yet separated; `docs/13-draft-embed-ab.md` carries the full table.

Cause 3 is now the leading candidate: a single sequence slot appears to be slower than two on
this engine, probably through smaller CUDA-graph capture shapes and a smaller sampler row count.

The second arm keeps the scheduler shape at two sequence slots and moves only the embedding device,
using a 262,144 window where 14,176 MiB still admits two full requests:

```
export QWEN38_DRAFT_EMBED_DEVICE=gpu
--max-model-len 262144 --max-num-seqs 2 --kv-mib 14176 --kv-cache-dtype fp8_e5m2 \
--block-size 160 --max-num-batched-tokens 1024
```

That arm did not reach ready: the engine reported `starting` for more than 32 minutes with the
weights already resident (22,533 MiB on the 3090), far outside the 276-353 s window every other arm
needed. It is recorded as an unfinished attempt rather than a decode result, and the embedding
device stays marked neutral-but-unproven.

### Run-to-run variance

A third arm measured code 57.06 / 24.28 / 25.11 and prose 2.28 / 27.22 / 2.83 with the same
CPU embedding and `--max-num-seqs 1`. Those numbers are not comparable to the other arms: the
engine load in that run was 854.87 s against 311.25 s and 276.17 s, and the host was carrying a
large page cache from the previous engine. A run whose load time triples has not reached a steady
state, so its decode figures are treated as invalid rather than as a fourth data point. Every arm
in this document needs a fresh engine and a load time near 300 s before its numbers are read.

## 6. Paths to the targets

| Path | Mechanism | Status |
|---|---|---|
| A. Draft embedding back on the 3090 | Removes per-step host-to-device copies | **Refuted**: 105.85 code, below the 139.91 CPU arm |
| B. Keep the embedding on CPU but gather on the 3090 | Copy the table once per step instead of per token, or use a pinned buffer and a single async copy | Untested; low expected gain now that A is refuted |
| C. Raise `--max-num-batched-tokens` to 2048 | Restores the prefill budget | Untested; prefill-only lever |
| D. Quantize the draft embedding to int8/fp8 | 0.6 GiB instead of 1.19 GiB | Untested; memory-only lever |
| E. `--max-num-seqs 2` with the embedding on the 3090 at a 262,144 window | Separates the embedding device without breaking the scheduler shape; 14,176 MiB still admits 2 x 262,144 | Running |
| G. `--max-num-seqs 1` | Was measured and rejected: the engine load tripled and decode collapsed, so it is not a usable arm | **Rejected** |
| F. Return to the stage 1 recipe and accept a 262,144 window | No concurrency 2, but code back to 194.70 | Known-good fallback |

The 180 / 70 targets need the scheduler shape and the KV dtype read path, not the embedding
placement. Path E is the measurement that decides whether the stage 3 recipe can be repaired
without giving up concurrency 2.

## 7. Blocked points

- Two full 409,600 requests need 592 KV blocks; even a 17 GiB pool gives 311. Concurrency 2 at
  409,600 is arithmetically impossible on this card.
- Prefill beyond 32,768 needs an indexer workspace of 128 MB with about 92 MiB free. Even at
  `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32` the request is 24 MB against 24.2 MiB.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` pushed host RAM to 121 GiB and stalled SSH;
  it is not usable on this workload.
- Engine load time drifts from 276 s to 854 s across arms. The cause is the host page cache and the
  PLE table: a run started right after another engine inherits a cold cache for the 51.2 GiB table
  and pays the difference in load time. Load time is therefore a health signal, and an arm whose
  load time triples is discarded rather than compared.

## 8. Reproduce

Engine flags per stage:

```
# Pre-extension and stage 1
--max-model-len 204800 --max-num-seqs 1 --kv-mib 12288 --block-size 160 --max-num-batched-tokens 2048
--max-model-len 262144 --max-num-seqs 1 --kv-mib 14336 --block-size 160 --max-num-batched-tokens 2048

# Stage 2 path A and B
--max-model-len 409600 --max-num-seqs 1 --kv-mib 14336 --block-size 160 --max-num-batched-tokens 2048

# Stage 3
--max-model-len 340000 --max-num-seqs 2 --kv-mib 15360 --kv-cache-dtype fp8_e5m2 \
--block-size 160 --max-num-batched-tokens 1024
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32
```

Stage 2 and later need `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` plus a `config.json` view whose
`text_config.max_position_embeddings` matches the window. Stage 3 needs the QSA fork patches for
`fp8_e5m2` (`qwen38_qsa_v2.py`, `qwen38_ops_qsa_v2.py`) mounted over the image files.

The measurement itself is the pinned sparkDash runner:

```
node sparkdash-exact/runner.mjs --options <options.json>
```

where `options.json` carries `concurrencies`, `maxTokens: 400`, and for prefill
`contextSizes`. The driver scripts in this repository (`sep-c2_driver.py`, `sep-c2gpu_driver.py`,
`sep-c2s1_driver.py`) build that options file and record the effective config from the artifact.

## 9. Related documents

- `docs/06-results.md` carries the same stage numbers with surrounding context.
- `docs/09-context-extension-routes.ko.md` carries the route design and the YaRN boundary pass.
- `docs/10-409k-advisory-report.ko.md` carries the 409,600 advisory answers.
- `docs/11-concurrency-two.md` carries the 8-bit KV patch and the concurrency ladder.
- `docs/13-draft-embed-ab.md` carries the embedding-device A/B.
- `docs/14-pdf-recovery-ladder.md` carries the six single-variable recovery steps and their decisions.
- `docs/16-recovery-paths-results.md` carries the four remaining advisory paths and the adopted combination.
- `docs/12-decoding-regression.ko.md` is the Korean summary of this page.

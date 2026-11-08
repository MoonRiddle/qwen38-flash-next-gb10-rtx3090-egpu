# Advisory review brief: long-window measurements on GB10 + RTX 3090 eGPU

This brief asks for an independent check of the measurement set recorded in
`docs/14-pdf-recovery-ladder.md` and `docs/16-recovery-paths-results.md`. The question is
whether the numbers are trustworthy and whether the reporting distinguishes what was
measured from what was only attempted.

## 0. Environment

- Helper: NVIDIA GB10 (SM121), 121.7 GiB unified memory.
- Owner: RTX 3090 eGPU (SM86), 24 GiB GDDR6X, PCIe Gen4 x4, non-P2P, unlocked BF16.
- Model: `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` with MTP draft
  (`draft-k10-int4`), GPTQ 4-bit, `group_size 128`, `desc_act false`.
- Engine: fork vLLM `0.1.dev20073+g8e685d198`, image
  `qwen38-flash-dgx:a5b-autoround-bottleneck-q2`.
- Driver, unlock and power settings are untouched. No sudo was used.

## 1. Measurement contract

- Runner: MiaAI-Lab sparkDash native runner pinned at commit
  `7c2b9638b025cd39fb43f95c225de62980f130ee`. No local collector substitutes.
- Decode: `maxTokens 400`, C=1 and C=2 in one job per repetition, three repetitions.
- Prefill: one job per configuration at 4K, 16K, 32K.
- Reported decode figure: median of the three per-request `medianDecodeTps` values.
- Reported aggregate figure: median of the three `aggregateDecodeTps` values.
- A run counts as a success only when the execution record says `status completed`,
  `failures` is empty, and `restoration_verified` is true.
- Instantaneous peaks and summed concurrent rates are never used as evidence for a
  single-stream claim.

## 2. Run record status

| Record | status | failures | restoration_verified | Note |
|---|---|---|---:|---|
| `sep-c2` | failed | 3 | false | 340,000 decode finished, prefill short |
| `sep-c2s48` | failed | 3 | false | baseline decode finished, prefill short |
| `sep-c2i8s` | failed | 1 | false | decode finished, prefill engine death |
| `sep-c2e` | failed | 3 | false | RTX 3090 OOM in prefill |
| `sep-c2y35` | failed | 3 | false | engine refused connections in prefill |
| `sep-c2f` | failed | 3 | false | RTX 3090 OOM in prefill |
| `sep-g` | completed | 0 | true | 13 jobs |

The six `failed` records still contain three decode repetitions each. The decode
medians below come from those repetitions. The failure column is kept visible rather
than dropped, so no failed run is presented as a completed one.

Also visible: `a-sep-c2-k6-open-prefill-01.json` returned `prefillTps 0`, `samples 0`
and `error "fetch failed"` for all three prefill sizes, so the 340,000 prefill numbers
in `docs/11-concurrency-two.md` are not traceable to that record.

## 3. Recovery ladder results (decode, C=1)

Baseline is `sep-c2s48.sh`: skip-layers 48, index share 0, batch 1024, `fp8_e5m2`,
409K views, draft depth 15.

| Configuration | code C=1 | prose C=1 | admission | load s | Decision |
|---|---:|---:|---|---:|---|
| baseline `sep-c2s48` | **122.96** | **50.58** | 677,901, 1.99x | 353.4 | kept |
| int8 side-scale `sep-c2i8s` | 18.47 | 16.90 | 677,901, 1.99x | 301.2 | reverted |
| draft embed `sep-c2e` | 115.84 | 48.13 | — | 361.5 | reverted |
| YaRN 1.35 `sep-c2y35` | 96.18 | 49.70 | 677,901, 1.99x | 340.3 | reverted |
| single stream `sep-c2f` | 106.93 | 49.62 | 677,901, 1.99x | 341.4 | reverted |

Layer 48 routed experts are already INT4 in the published checkpoint, so that path was
dropped without a run.

## 4. Long-window split results (decode)

The 409,600 window admits `427,468 tokens, 1.04x`, so two full requests do not fit.
Engine `sep-g` is `sep-f` with `--max-num-seqs 4`, KV 14 GiB, run at C=2 and C=4 with
three repetitions.

Limit: the sparkDash native runner treats `contextLength` as prefill-only, so these
decode rows use the fixed short prompt (about 67 input tokens, 400 forced output
tokens). They measure concurrency, not 200k or 100k prompts.

A separate long-prompt probe with `--kv-mib 14336` (`631,728 tokens, 1.86x`) sent two
requests at 4K, 16K, 32K and 315,000: prompt tokens were 4,244, 16,975, 33,940 and 326,260
per stream and every request completed. The probe was non-streaming and recorded only wall
time, so TTFT overlap and simultaneous residency are unverified.

| Case | Workload | C | per request | aggregate | streams failed |
|---|---|---:|---:|---:|---:|
| C=2 | code | 2 | 149.00 | 260.81 | 0 |
| C=2 | prose | 2 | 44.17 | 86.60 | 0 |
| C=4 | code | 4 | 91.82 | 302.98 | 0 |
| C=4 | prose | 4 | 30.53 | 115.28 | 0 |

Prefill on the same engine: 2,814.48 at 4K, 2,698.71 at 16K, 2,621.51 at 32K.

## 5. Headline numbers used in the README

| Window | Workload | C=1 | C=2 per request | C=2 aggregate | Config |
|---|---|---:|---:|---:|---|
| 409,600 whole | code | 185.89 | — | — | `sep-f`, MTP int4, KV 14 GiB |
| 409,600 whole | prose | 67.28 | — | — | `sep-e`, config extension, KV 14 GiB |
| 340,000 | code | 114.11 | 86.79 | 163.46 | `sep-c2`, KV 15 GiB |
| 340,000 | prose | 45.85 | 40.16 | 78.35 | `sep-c2`, KV 15 GiB |

The C=1 values are medians of three repetitions. The `sep-e` prose repetitions were
27.34, 67.28 and 72.84, so 72.84 is the observed maximum rather than the median; the
README now reports 67.28. The `sep-c2` code repetitions were 114.11, 108.13 and 115.32,
so 139.91 is a single repetition rather than the median; the README now reports 114.11.
The two 409,600 rows come from different configurations. Code peaks with MTP int4 (185.89),
and prose is also higher with MTP int4 (70.93) than with the plain config extension (67.28).

## 6. Review requests

1. Is the three-repetition median with the pinned runner sufficient to call a decode
   difference real at this noise level? The prose C=1 spread in `sep-g` was 30.80 to
   51.97, which is wide.
2. Is it defensible to report decode medians from records whose `status` is `failed`
   because prefill died afterwards, as long as the failure is shown next to the number?
3. Does splitting a 400,000-token window into 200k x 2 and 100k x 4 measure the same
   work as one 409,600 request, or does it change the workload enough that the
   aggregate comparison is misleading?
4. Are the admission ratios quoted from the engine logs (`677,901 tokens, 1.99x` and
   `427,468 tokens, 1.04x`) the right basis for the "two requests do not fit" claim?
5. Is anything in the reporting that overstates what was observed, or that omits a
   failed attempt that should be visible?

## 7. Reproduction

Remote workspace `/home/user/OMP_Workspace/qwen38-gb10-3090-20261029`:

- `sep-c2i8s.sh` + `sep-c2i8s_driver.py`
- `sep-c2e.sh` + `sep-c2e_driver.py`
- `sep-c2y35.sh` + `sep-c2y35_driver.py`
- `sep-c2f.sh` + `sep-c2f_driver.py`
- `sep-g.sh` + `sep-g_driver.py`
- `build_135_views.py` (run inside the engine image with `-v $BASE:/work -v /srv/vault:/srv/vault`)

Execution records are `a-<prefix>-execution.json`; per-job artifacts are
`a-<prefix>-k6-open-*.json`.

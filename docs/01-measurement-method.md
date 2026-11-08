# 1. Measurement method

Every number in this repository comes from MiaAI-Lab's [`sparkDash`](https://github.com/MiaAI-Lab/sparkDash)
collectors, run through a pinned, byte-verified copy in `bench/sparkdash-exact/`. Nothing here is a
hand-rolled benchmark, and no result is a best-of-N retry.

## Pinned upstream commit

| Field | Value |
|---|---|
| Repository | `https://github.com/MiaAI-Lab/sparkDash` |
| Pinned commit | `7c2b9638b025cd39fb43f95c225de62980f130ee` (2026-11-02) |
| Upstream HEAD checked | `da20ac5b` (v2.77.0, 2026-11-04) |
| Verdict | All nine files of the measurement closure are byte-identical between the pinned commit and HEAD |

The nine files are `server/collectors/DecodeBench.js`, `server/collectors/PrefillBench.js`,
`server/collectors/LlmStreaming.js`, `server/collectors/contextFill.js`, `server/util/atomicWrite.js`,
`src/shared/llmPrompts.js`, `src/shared/llmTarget.js`, `src/shared/prefillBench.js`,
`src/shared/contextFit.js`. Their sizes and SHA-256 digests are recorded in
`bench/sparkdash-exact/source-lock.json`, and `runner.mjs` refuses to start a job if any digest
mismatches.

This means the harness is not a fork of the method: it is the method. The only file that differs from
upstream is `runner.mjs`, which is our invocation wrapper and is described below.

## Decode throughput

`server/collectors/LlmStreaming.js` computes the rate from the streamed token timestamps:

```
decodeTps = (completionTokens - 1) / (tLast - tFirst) * 1000
```

`DecodeBench.js` fixes the request so the measurement cannot drift:

- `min_tokens = max_tokens` — the server is forced to emit exactly the requested count.
- `ignore_eos` — a natural end of turn cannot truncate the sample.
- `stop: []` — no stop sequence can end the stream early.
- Defaults: `DEFAULT_TEMPERATURE = 0.7`, `DEFAULT_THINKING = true`, `DEFAULT_MAX_TOKENS = 400`,
  `WARMUP_MAX_TOKENS = 32`.
- Allowed concurrency set: `{1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32}`.
- One 32-token warmup request per job, before the measured waves.

`completionTokens - 1` is deliberate: the first token is the prefill's output, so it is excluded from
the decode window.

## The two decode workloads

| Workload | Temperature | Thinking | Prompt source |
|---|---:|---|---|
| `prose` | 0.7 | on | `DECODE_PROSE_PROMPT` in `src/shared/llmPrompts.js` — a hash-map explanation task |
| `code` | 0.0 | off | `DECODE_CODE_TASKS` in `src/shared/llmPrompts.js` |

`runner.mjs` derives these from the case name, not from flags, so the two workloads cannot be mixed
up. Code uses greedy sampling and no reasoning tokens; prose uses the model's sampling temperature and
its reasoning path.

## Prefill throughput

`server/collectors/PrefillBench.js` and `src/shared/contextFill.js` build a prompt of a requested size:

- A unique salt header per attempt, so no two attempts share a prefix (prefix caching cannot help).
- LCG-generated filler words after the header.
- The reply is a fixed "OK".

The rate is TTFT-based, with a calibrated overhead subtracted:

```
rate = promptTokens / (ttftMs - overheadMs)
```

- `MAX_OVERHEAD_CORRECTION = 0.5` — the overhead correction is capped at half the TTFT.
- `LOW_CONFIDENCE_OVERHEAD_SHARE = 0.3` — above a 30% overhead share the row is flagged
  `lowConfidence`.
- `CALIBRATION_TOKENS = 16`, `CALIBRATION_SAMPLES = 3` — the overhead is measured, not assumed.
- `repeatsForSize`: 3 samples at ≤32,768 tokens, 2 samples at ≤131,072, 1 sample above that.
- Default size list: `[4096, 8192, 16384, 32768]`.

Prefill is sequential. `PrefillBench` does not take a concurrency list, so a "C=4 prefill" number
does not exist in this method.

## The invocation wrapper

`bench/sparkdash-exact/runner.mjs` calls the untouched upstream `DecodeBenchManager` and
`PrefillBenchManager` public API and writes one immutable envelope per job. Upstream `runner.mjs`
hardcodes `concurrencies: [1]`, `maxTokens: 400` and prefill `contextSizes: [4096, 16384]`; this copy
accepts them as options so the same upstream collectors can be asked for other concurrency and size
grids:

```jsonc
{
  "case": "prose",              // prose | code | prefill
  "concurrencies": [1, 2, 4],   // decode only; validated against the upstream allowed set
  "maxTokens": 400,             // decode only; validated 64..2048
  "contextSizes": [4096, 8192, 16384],  // prefill only; validated 256..300000
  "configurationId": "k6-open",
  "configuration": { /* the live server controls, freshly read before the job */ },
  "freshSnapshot": { /* engine identity, peer state, PLE diagnostics, observation only */ },
  "engineInstanceId": "<uuid of the engine process that served this job>"
}
```

The wrapper rejects an existing envelope path rather than overwriting it, so a job can never be
re-run "to get a better number" under the same name. A failed job keeps its failure.

## What a row means

Decode rows carry `medianDecodeTps` (per-request median across the streams), `aggregateDecodeTps`
(one shared decode window across all streams), `streamsOk`, `streamsFailed`, `medianTtftMs` and
`totalCompletionTokens`. At C=1 the median and the aggregate are the same number; above C=1 they are
different metrics and both are reported.

Prefill rows carry `prefillTps`, `ttftMs`, `overheadMs`, `promptTokens`, `targetTokens`, `samples`,
`samplesRequested`, `cachedTokens`, `lowConfidence` and `method`.

## Engine controls applied before every job

The server exposes its speculative-decoding controls over an RPC. Before each job the matrix runner
pushes a named configuration and then re-reads the server state to prove the push took effect; the
configuration actually used is recorded in the envelope. Two configurations are used in this
repository:

| Configuration | prose draft depth | prose top-k | confidence gate |
|---|---:|---:|---:|
| `k6-open` | 6 | none | none |
| `k6-top2-conf50` | 6 | 2 | 0.5 |

Code draft depth is fixed at the server's `--mtp 15`; the engine does not expose a runtime setter for
it, which the envelope records as `code_depth_runtime_setter_supported: false`.

## Reproducing a measurement

```bash
# 1. start the engine (see engine/ for the build and the launcher)
# 2. capture provenance without issuing inference
python3 bottleneck_provenance.py --container qwen38-<name> --state <name>.json --out <name>-provenance.json
# 3. run the grid
python3 run_sparkdash_original_matrix.py \
    --state-file <name>.json --provenance <name>-provenance.json \
    --engine-id <uuid from the state file> \
    --node bench/sparkdash-runtime/node-v24.22.2-linux-arm64/bin/node \
    --run-prefix <prefix> --matrix-id <matrix> \
    --cases prose code prefill --configs k6-open \
    --concurrencies 1 2 4 --context-sizes 4096 8192 16384 \
    --decode-jobs 3 --prefill-jobs 3
```

Every job writes `<prefix>-<config>-<case>-<NN>.json` plus a `-options.json`. The matrix runner also
writes `<prefix>-execution.json`, which records the schedule, the control snapshots, the restore step
and whether the restore was verified.

### A concurrency grid needs a different driver

`run_sparkdash_original_matrix.py` registers each envelope through the campaign validator in
`progress-web/sparkdash_exact.py`, and that validator's decode contract is

```python
config["concurrencies"] == [1] and config["maxTokens"] == 400
```

so it accepts a C1-only matrix and rejects any envelope with a wider concurrency list. That is a
campaign bookkeeping rule, not a measurement rule. For a C=2/C=4 grid, use `bench/run_matrix.py`,
which pushes the same server controls and calls the same pinned `runner.mjs`, then asserts the
effective `concurrencies` itself:

```bash
python3 bench/run_matrix.py --state-file <name>.json \
    --node bench/sparkdash-runtime/node-v24.22.2-linux-arm64/bin/node \
    --config k6-open --case prose --concurrencies 1 2 4 --max-tokens 400 --repeats 3
```

The envelopes it produces are identical in shape; only the registration step differs. Both drivers
refuse to overwrite an existing envelope.

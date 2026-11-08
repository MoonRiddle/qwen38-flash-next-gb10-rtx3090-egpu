# 5. Verification

This repository ports the checks from
[`MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
at commit `20e15e41f2f514a0d287710f4665ee1b50099ba41` (MIT), adapted only where this recipe's 17,408-token
window and engine require it. The tools live in `verify/`.

## Exit codes

Every tool uses the same four codes, so a summary is unambiguous:

| Code | Meaning |
|---:|---|
| 0 | checked and passed |
| 1 | checked and failed |
| 2 | could not run (server unreachable, HTTP error, bad reply) |
| 3 | ran, but the check could not be verified — reported as UNCHECKED, never as a pass |

## What each check proves

| Tool | What it proves | Engine feature it needs |
|---|---|---|
| `exact.py` | A greedy reply is identical whether the request is served alone, concurrently, or with the draft path disabled. Also reports whether the server says drafts actually ran. | `"draft": false` request field; optional token-level sha |
| `exactness_cross.py` | Greedy replies are identical across two engine builds (drafts on vs drafts off), which a single server cannot show. | two `exact.py --dump` files |
| `toolcheck.py` | Tool-call emission survives the speculative path: the parsed call is stable across serial and concurrent requests. | tool-calling chat template |
| `needle.py` | A passphrase buried at 60% depth of a 16,384-token prompt is still retrieved greedily with thinking off. | prompt length inside the served window |
| `prompt_reuse.py` | A follow-up turn reuses the previous prompt prefix (cached tokens), including with an interleaved conversation. | `cachedTokens` reported in usage |
| `context_boundary.py` | The advertised window is real: a request that fills `--context` minus a 16-token reply margin is accepted and answered, streamed and non-streamed, drafts on and off. | `--max-model-len` matching the advertised window |

## What was skipped, and why

`visioncheck.py` is **not** run. This recipe serves text only; the engine image has no vision tower and
the model views contain no image encoder. Running it would fail for a reason unrelated to the recipe, so
it is recorded as not applicable rather than as a pass.

`bench.py` and `bench_streams.py` from TensorFold are also not ported. They duplicate what `sparkDash`
already measures with a stricter contract, and mixing two throughput harnesses in one repository would
break the measurement provenance rule stated in `01-measurement-method.md`.

## Adaptations made

Only three, all of them about the window rather than the method:

| File | Change |
|---|---|
| `needle.py` | default size 16,384 instead of 200,000; the hard 1,048,576-token guard became `RECIPE_MAX_CONTEXT` (default 17,408) |
| `context_boundary.py` | `--context` defaults to `RECIPE_MAX_CONTEXT` (17,408) instead of being required at 262,144 |
| `prompt_reuse.py` | default first-turn size 16,000 instead of 30,000 |
| `exact.py` | added `--dump PATH`, which writes per-prompt reply hashes so `exactness_cross.py` can compare two engine builds |

`client.py` and `toolcheck.py` are unmodified.

## Environment

The tools read the same variables as upstream:

```bash
export API_URL=http://127.0.0.1:18800    # or PORT
export MODEL=qwen38-mia-a5b
export RECIPE_MAX_CONTEXT=17408
```

## Results

Recorded in `artifacts/verify/results.json`, one entry per tool:

```jsonc
{
  "tool": "exact.py",
  "argv": ["--pad", "3000", "--dump", "exact-drafton.json"],
  "exit_code": 0,
  "summary": "PASS draft ... PASS conc ...",
  "unchecked_reason": null
}
```

A tool whose exit code is 2 or 3 carries the exact HTTP status, response body or missing-field reason in
`unchecked_reason`. No such row is ever rewritten as a pass.

## Measured outcomes

Engine: KV 8 GiB, `--max-model-len 17408`, `--max-num-seqs 4`, image `b27377b7`, controls `k6-open`,
PLE from `/mnt/nvme20/ple-table-int4`.

| Tool | Exit | Outcome | What it showed |
|---|---:|---|---|
| `needle.py 16384` | 0 | **PASS** | prompt 16,379 tokens, needle at 60% depth, found `warm-library-4557` in 8.5 s |
| `context_boundary.py --context 17408` | 0 | **PASS** | 16 checks across streamed/non-streamed and drafts on/off at the advertised window |
| `exact.py --pad 3000` | 1 | **FAIL** | greedy replies differ between the draft-on path and `"draft": false`, and between serial and concurrent |
| `prompt_reuse.py 16000` | 2 | could not run | HTTP 400: prompt 17,361 + 48 reply tokens exceeds the 17,408 window |
| `prompt_reuse.py 12000` | 1 | **FAIL** | follow-up turns resumed 0.0% of their prompt; the engine serves with prefix caching off |
| `toolcheck.py` | 2 | could not run | HTTP 400: `"auto"` tool choice requires `--enable-auto-tool-choice` and `--tool-call-parser` |

### Why `exact.py` fails

Six prompts, `temperature 0`, `seed 4321`, 3,000 tokens of padding. Two of six replies matched across
all three lanes; four differed. Examples from the dump:

- prompt #2 serial begins `The first twelve prime numbers are:` while the draft-off lane begins
  `Here are the first twelve prime numbers:` — a different opening, not a truncated tail.
- prompt #4 serial emits a fenced code block while the draft-off lane emits bare JSON, and the subject
  tags differ.

The envelope also shows the server does not report `tensorfold.drafts` (`None`) or `speculative.accepted`
(`None`), and reports no token-level sha. So `exact.py` cannot even tell whether drafts ran; it compares
text only.

The engine runs with `VLLM_MARLIN_USE_ATOMIC_ADD=1`. An atomic-add reduction is order-dependent, so the
same logits are not guaranteed bit-identical between a batched draft-verify path and a plain path. That
is the mechanism behind a temperature-0 difference: the recipe is *numerically* non-deterministic across
execution paths.

This is recorded as FAIL, not downgraded. It means:

1. The recipe does not claim bit-exact reproducibility for greedy output across draft settings.
2. Cross-build exactness (`exactness_cross.py`) can only report what it observes, and a difference there
   has the same root cause rather than indicating a broken draft head.

If bit-exact greedy output is a requirement, the fix is `VLLM_MARLIN_USE_ATOMIC_ADD=0`, which costs
decode throughput on this card; that trade was not measured here.

### Why `prompt_reuse.py` fails

`cachedTokens` is 0 on every turn, including the cold first turn, so the server is not reusing a prefix.
The engine is launched without prefix caching enabled. For this recipe that is deliberate: every
`sparkDash` prefill prompt carries a unique salt precisely so that prefix caching cannot help, and
`cachedTokens == 0` is the evidence that the measurement is honest. The consequence is that multi-turn
chat pays full prefill each turn, which matters for interactive use and does not matter for the
benchmark.

### Why `toolcheck.py` cannot run

The engine image is built for text generation and does not pass `--enable-auto-tool-choice` or
`--tool-call-parser`. A tool-call check therefore cannot be exercised on this recipe as configured. This
is a configuration gap, not a defect in the model: the checkpoint's own card reports tool use preserved
at 91–92. Adding the flags would enable the check; that was not done here because it changes the served
contract.

### Not applicable

`visioncheck.py` is skipped: text-only recipe, no vision tower in the engine image.

### Cross-build exactness: UNCHECKED

`exactness_cross.py` needs two `exact.py --dump` files from two engine builds, one with the draft path
and one without. The recipe cannot produce the second dump:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for SpeculativeConfig
  num_speculative_tokens: Input should be greater than 0
```

Starting the engine with `--mtp 0` is rejected by vLLM's `SpeculativeConfig`, so there is no
draft-free build to compare against. The `"draft": false` request field exists but, as shown above, the
server neither honours it bit-exactly nor reports whether drafts ran.

Recorded as UNCHECKED with that reason. It is not reported as a pass, and the comparison script itself
was validated independently: given two identical dumps it exits 0 with `PASS cross`, and given one
differing row it exits 1 with `FAIL cross` naming the index.

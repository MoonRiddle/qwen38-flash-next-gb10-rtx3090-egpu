# 2. Why this checkpoint is fast at code and slow at prose

The checkpoint is `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`. Its code decode rate and its
prose decode rate differ by roughly 3x on the same engine, same session, same measurement. This
document states what the model author actually changed, and separates that from a framing that is
often repeated but is not what happened.

## The claim that is wrong

> "azampatti pruned the prose experts, so the model lost prose capability and kept code."

This is inaccurate. Nothing was removed from the expert pool. The model card and the author's own
forum posts both state that all **512 routed experts are retained**; only the routing depth changed.
There is no per-expert deletion, no expert-level surgery on the prose experts, and no "coding-only"
expert set.

## What actually changed

| Change | Before | After | Source |
|---|---|---|---|
| Routed experts kept | 512 | 512 (unchanged) | Model card; DGX Spark forum topic 382733, post 1 |
| Routed experts per token | 10 | 5 | Model card; forum post 1 |
| Active parameters | 6.0 B | 4.8 B | Model card; forum post 1 |
| Healing scope | — | one always-on component, 37.75 M parameters | Forum post 1 |
| Healing data | — | 14,667 examples of code, maths, general text, tool calls and long context | Model card |
| lm_head | BF16 | int4 GPTQ-Marlin g128 (97.2% top-1 agreement vs BF16); int8 variant 99.6% | Model card |
| Draft head | BF16 top-10 | BF16 top-10 (unchanged) | Forum post 1 |

The healing is a single shared component trained by KL self-distillation against the original BF16
teacher. Only that component learns; the 512 experts keep their original weights. So the prose
capability was not deleted — it was left un-healed while the routing depth changed under it.

## The measured acceptance gap

Speculative decoding only speeds up decode when the draft head's tokens are accepted. The author
publishes acceptance per workload, and the numbers explain the speed gap directly.

Author's own statement (`azampatti/Qwen3.8-Flash-Next-Int4-FAST`, README):

> At temperature 0.5 it gets 75% of its drafted tokens accepted overall (85% code, 63% prose, 89%
> long context).

DGX Spark forum topic 382733, post 6, per-category acceptance:

| Category | Acceptance |
|---|---:|
| CODE | 82.1% |
| LONGCTX | 85.5% |
| MATH | 76.8% |
| TOOL | 68.9% |
| GENERAL | 55.7% |

Same forum thread, post 5, is explicit about the consequence:

> "prose" text is SLOOOWW to draft from the MTP.

So the prose penalty is a draft-acceptance penalty, not a capability removal. At 55.7–63% acceptance,
roughly two of every five drafted tokens are rejected and must be re-verified, and the reasoning
(thinking) path spends most of its tokens on exactly the prose-like distribution where acceptance is
lowest. That is why the measured prose rate lands near a third of the code rate.

## The head dtype mismatch

The target's `lm_head` is int4 while the draft head stays BF16 top-10. The model card measures the
consequence: the int4 head agrees with the BF16 head on 97.2% of top-1 tokens, and the int8 variant
reaches 99.6%. A 2.8% top-1 disagreement is small per token but it feeds the rejection sampler, which
is the mechanism that turns acceptance into throughput.

## The unverified part

PDF 9 (`analysis/GB10-and-RTX3090-decoding-optimization-9.pdf`) states a healing-data mix of 30%
maths, 22% tool, 22% code and 20% general prose. That breakdown is **not** present in the Hugging Face
model card, which describes the 14,667 examples only by category without percentages. Treat the
percentages as a PDF-9 claim that has not been confirmed against a primary source.

## What this means for the recipe

- Code decode is the fast path because acceptance is 82–85%.
- Prose decode is bounded by acceptance, not by memory bandwidth, expert dispatch or PLE latency:
  prefill rates are identical across workloads (see `06-results.md`), so the shared compute path is
  not the differentiator.
- Raising prose throughput by engine tuning alone has a ceiling. The two levers that actually move it
  are draft depth for the prose path (`k6-open` vs `k6-top2-conf50`) and warmup, which is why the
  cold-start analysis in `03-cold-start-analysis.md` matters.
- A checkpoint with a prose-healed component, or an int8/BF16 head, would change this. This recipe
  does not claim otherwise.

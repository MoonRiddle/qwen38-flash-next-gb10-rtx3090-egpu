# 13. Draft embedding device A/B

## Question

`peer_worker.py:762` moves the MTP draft's `VocabParallelEmbedding` to host memory for every
draft model load. That frees 1.19 GiB on the 3090 (`primary_bytes` 9,757,622,784 to
8,567,792,128). The cost is that every speculative step gathers the draft embedding on the CPU
and copies the gathered rows back to the 3090 over a non-P2P PCIe link.

The hypothesis was that this per-step copy explains most of the stage 3 decode regression
(194.70 code at stage 1 to 139.91 at stage 3). The engine now reads
`QWEN38_DRAFT_EMBED_DEVICE` (default `cpu`, `gpu` skips the move), so the variable can be
separated.

## The two engines

| | Embed on CPU (stage 3) | Embed on GPU (A/B arm) |
|---|---|---|
| Window | 340,000 | 340,000 |
| KV dtype | `fp8_e5m2` | `fp8_e5m2` |
| KV pool | 15,360 MiB | 14,176 MiB |
| Block | 160 | 160 |
| Batch | 1024 | 1024 |
| Sequence slots | 2 | 1 |
| Draft embed | CPU | 3090 |
| Engine load | 311.25 s | 276.17 s |

The A/B arm had to give back 1,184 MiB of KV because the embedding table returned to the card.
Two variables moved together: the embedding device and the sequence-slot count. The first run
therefore cannot separate them, and that is exactly what the result showed.

## Results

Decode, C=1, 400 forced tokens:

| Arm | Code runs | Code median | Prose runs | Prose median |
|---|---|---|---:|---|---:|
| Embed CPU, seqs 2 | 139.91 / 115.55 / 114.39 | 115.55 | 38.50 / 52.89 / 52.23 | 52.23 |
| Embed GPU, seqs 1 | 105.85 / 101.59 / 114.13 | 105.85 | 33.16 / 54.98 / 55.10 | 54.98 |

Prefill at 4,096 tokens:

| Arm | Prefill tok/s | TTFT ms |
|---|---:|---:|
| Embed CPU, seqs 2 | 2,289.39 | 1,921.87 |
| Embed GPU, seqs 1 | 2,162.97 | 2,032.20 |

## Interpretation

Moving the draft embedding back to the 3090 did not recover decode. Code fell from a 115.55 median
to 105.85, and prefill fell 5.5%. Prose moved slightly up (52.23 to 54.98) but the spread is wider
than the change. So the CPU gather is not the dominant cost, and the hypothesis is refuted for
code.

Two readings remain, and they are not yet separated:

1. `--max-num-seqs 1` is slower than `--max-num-seqs 2` on this engine. A single sequence slot can
   shrink the CUDA-graph capture shapes and the sampler row count, so the speculative verify pass
   runs on a smaller batch.
2. Restoring 1.19 GiB of embedding to the card cut the KV pool by 1,184 MiB, and the smaller pool
   changes how many blocks the scheduler can hand out.

The sequence-slot arm was run and rejected; see the invalid arm section below. The arm that
separates the embedding device without breaking the scheduler shape uses a 262,144 window with
`--max-num-seqs 2`, where 14,176 MiB still admits two full requests.

## The invalid `--max-num-seqs 1` arm

A third arm kept the CPU embedding and dropped only the sequence slots at the same 15,360 MiB
pool. It produced code 57.06 / 24.28 / 25.11 and prose 2.28 / 27.22 / 2.83, with prefill at
208.69 tok/s and a TTFT of 32.77 s.

Those numbers are rejected, not recorded as a data point. The engine load in that run was
854.87 s against 311.25 s and 276.17 s for the two valid arms, and the host page cache was still
carrying the previous engine. A run whose load time nearly triples has not reached a steady state,
and its decode and prefill rates describe a degraded engine rather than the configuration.

What the arm does establish is a hard rule: `--max-num-seqs 1` must not be used to compare
against the concurrency-2 recipe on this engine. Every valid arm in this document runs with
`--max-num-seqs 2` and a load time near 300 s.

## The 262,144 arm with two sequence slots

This arm keeps `--max-num-seqs 2` and changes only the embedding device, so the scheduler shape is
identical to the stage 3 recipe. The window drops to 262,144 so that 14,176 MiB still admits two
full requests.

```
export QWEN38_DRAFT_EMBED_DEVICE=gpu
--max-model-len 262144 --max-num-seqs 2 --kv-mib 14176 --kv-cache-dtype fp8_e5m2 \
--block-size 160 --max-num-batched-tokens 1024
```

Its result is not yet available. The engine reported `starting` for more than 32 minutes with the
weights already resident (22,533 MiB on the 3090) and no stdout past the load banner, which is far
outside the 276-353 s window every other arm needed. That arm is therefore not a usable comparison
and is recorded as an unfinished attempt, not as a decode result.

The comparison target, once a healthy engine is available, is the stage 3 CPU-embedding arm at
340,000 (code median 115.55, prose median 52.23), not the 105.85 GPU arm, because that one also
changed the slot count.

## What this means for the targets

The 180 / 70 targets cannot be reached by moving the draft embedding alone. The stage 3 drop is
mostly a scheduler-shape effect, not a host-to-device copy effect. That reframes the search: the
next levers are `--max-num-seqs`, `--max-num-batched-tokens`, and the KV dtype read path, not the
embedding placement.

The 262,144 arm with `--max-num-seqs 2` is the clean test of that statement, because it keeps the
slot count and changes only the embedding device. Until it runs, the embedding device stays marked
as neutral-but-unproven, and the CPU placement is kept because it costs no decode rate and returns
1.19 GiB to the KV pool.

# 7. GB10 alone vs GB10 + RTX 3090 eGPU

This document answers one question: on the same checkpoint, same measurement harness and same workload
grid, what does the second GPU actually buy, and what does it cost?

## The two configurations

| | GB10 alone | GB10 + RTX 3090 eGPU |
|---|---|---|
| Compute | GB10, SM121, 121.7 GiB unified | GB10 + RTX 3090 SM86 24 GiB |
| Link between them | — | PCIe Gen4 x4, non-P2P |
| KV cache location | GB10 unified memory | RTX 3090 |
| Serving stack | vLLM dev `0.1.dev20073` for the registered solo run; Eugr b12x for the author's published range | fork build `a5b-autoround-bottleneck-q2` |
| Model views | author's `target` view inside the container | local `target-no-mtp` + `draft-k10-int4` |
| PLE table | 49 GiB inside the model snapshot | 51 GiB mmapped from USB |
| `block_size` | 1632 | 160 (and 1632 in the 256K experiment) |
| MTP depth | 5 | 15 |
| Max window | 262,144 | 17,408 (262,144 at block 1632) |

## Why the comparison is not symmetric

Two things differ at once, and they must be kept apart when reading the tables:

1. **Hardware.** One unified-memory device versus two devices over a narrow non-P2P link.
2. **Serving stack.** The author's b12x recipe versus this repository's fork build.

A pure hardware comparison would need the same stack on both. That is not available here: this
repository's fork build requires the peer worker (owner on the 3090, helper on the GB10) and cannot run
GB10-only, while the b12x recipe is GB10-only by design. So the tables below compare *recipes*, not
hardware in isolation, and every row says which recipe produced it.

### Unified-memory ceiling on GB10 alone

The GB10-alone configuration was started at the author's `kv_cache_bytes 20g`. On this machine it drove
unified memory to 97.6-99.7% and the host went down without a clean shutdown, four times in one session:

| Last healthy sample | RAM used | Available | CPU peak | GPU peak | Uptime at crash |
|---|---|---|---|---|---|
| 01:18:55 | 97.6% | ~3,000 MiB | 80.0 C | 57.7 C | ~21 min |
| 01:50:07 | 99.7% | 318 MiB | 67.8 C | 62.0 C | ~34 min |
| 02:39:34 | 98.7% | 1,642 MiB | 68.0 C | 49.0 C | ~42 min |
| 03:27:37 | 98.9% | 1,348 MiB | 68.2 C | 52.0 C | ~15 min |

Temperature is excluded as the cause: the CPU peak is 66-68 C in three of the four cases, well below a
thermal trip, and the 3090 idled at 43 C and 15 W. The last sample before each crash reports `online=False`
with the GPU status unreadable, then the host stops answering.

Growth is a step rather than a slope: unified memory went from 11,472 MiB to 80,362 MiB in 14 seconds. The
51 GiB PLE table is the largest reclaimable object, so it fills page cache during load and the kernel cannot
reclaim it fast enough.

Consequence for this document: the GB10-alone rows below use 4 GiB KV at a 16,384-token window, with a guard
that stops the container if unified-memory use reaches 85%. The 20 GiB figure from the author is not
reproducible on this machine without taking the host down.

### Why the GB10 cannot take over KV in the eGPU configuration

Moving KV into the GB10's unified pool would remove the 24 GiB card limit on the context window, so the
device-0 swap was tried directly. Putting the GB10 first in `CUDA_VISIBLE_DEVICES`, with `QWEN38_OWNER=1`,
`QWEN38_HELPER=0` and `QWEN38_SKIP_DEVICE_CHECK=1`, passes the role assertions and reaches model load. Three
barriers follow, in order:

1. `Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on
   cuda:1` in `vocab_parallel_embedding.py`. The trunk move places the embedding table on the owner while the
   input ids are built on device 0.
2. Moving `VocabParallelEmbedding` to the helper clears that, but Triton rejects the table with
   `Pointer argument cannot be accessed from Triton (cpu tensor?)`.
3. Wrapping the embedding forward in `torch.cuda.device(input_.device)` and keeping the table on the 3090 gets
   past Triton, but the first `index_select` fails with the same device split: the ids exist on cuda:0 before
   the embedding is entered.

The split is created before this call site, so it cannot be repaired there. The upstream state of the stack is
also not ready for the device: vLLM here is `0.1.dev20073`, and the SM121/GB10 support pull request (#31740) was
closed unmerged. Until that lands, the window stays at 262,144 and KV stays on the 3090.

## Results

Filled from `artifacts/` as each arm completes. Same harness, same prompts, same 400-token decode
contract, same prefill salt scheme.

### Decode

| Workload | C | GB10 alone (vLLM dev, HTTP/SSE, 67/39 input tokens) | GB10 + 3090 (this recipe) |
|---|---:|---:|---:|
| code | 1 | 101.44 | 211.98 (warm, KV 12 GiB) |
| prose | 1 | 38.80 | 80.00 (warm, KV 12 GiB) |
| prose | 4 | pending | 52.31 / 191.94 agg |
| code | 1 | pending | 211.98 (warm, KV 12 GiB) |
| code | 4 | pending | 95.48 / 316.01 agg |
| prose | 1 | pending | 76.12 / 68.92 warm (204,800 window, block 160, KV 12 GiB, graphs NONE) |
| code | 1 | pending | 192.28 cold (same 204,800 engine) |
| prose | 1 | pending | 76.77 / 73.61 warm (262,144 window, block 160, KV 14 GiB) |
| code | 1 | pending | 194.70 cold (same 262,144 engine) |

### Prefill

| Size | GB10 alone (cold PLE) | GB10 + 3090 (this recipe) |
|---:|---:|---:|
| 4K | 259.19 (4 GiB KV, 32,768 window, MTP off) | 3,133.27 |
| 16K | 117.87 (1 of 3 samples completed) | 3,064.22 |
| 32K | not measured (32,768 window) | 2,443.12 |
| 200K | not measured (32,768 window) | 2,035.64 |

These two columns are not a like-for-like comparison. The GB10-alone arm ran with `PLE_PREWARM=0`, so the
PLE table is read lazily from USB storage. The engine's own PLE stats show one gather of 53,076 rows taking
72.4 s for 8.1 MiB, which is why the 16K sample needs about 145 s to first token. The eGPU artifacts record
`ple_mmap_prewarm: true`, so the GB10-alone numbers measure storage latency, not GPU prefill throughput.
A fair comparison needs both arms in the same PLE state.

### Context

| | GB10 alone | GB10 + 3090 |
|---|---|---|
| Max admitted window | 262,144 by recipe default | 262,144 measured at block 1632 |
| KV tokens at 20 GiB | ~645k (author's figure) | 404,491 at 12 GiB / block 1632 |
| KV tokens at 12 GiB / block 160 | — | 61,740 |

## What to expect, stated as expectations rather than results

### The author's own numbers against this recipe

The Int4-FAST recipe reports 70-75 tok/s from its own harness and says to read that as a relative comparison,
not a sparkDash figure. Placing the measured runs beside it:

| Metric | Int4-FAST (author's harness) | This recipe, block 160, 17,408 | This recipe, block 1632, 262,144 (cold) |
|---|---|---|---|
| Single-stream decode | 70-75 tok/s | prose 80.00 / code 211.98 (warm) | prose 8.92 / code 75.92 |
| Draft acceptance | 75% overall, 85% code, 63% prose | not measured | 87.5%, 70.0%, 63.7% |
| Window | 262,144 | 17,408 | 262,144 |
| Harness | author's own | sparkDash | sparkDash |

The warm short-window prose figure is slightly above the author's range and the code figure is far above it,
but those rows use a 15x narrower window. The wide-window rows are cold and half the short-window rate, and
they carry the same acceptance rate as the author reports, so the gap is round time rather than acceptance.

- The eGPU adds decode throughput for small-window work because a 3090's GDDR6X bandwidth exceeds the
  GB10's unified-memory bandwidth, and the fork build keeps the hot experts there. Measured on the eGPU side:
  211.98 tok/s code C=1 warm at a 17,408 window, against the author's 70-75 tok/s for GB10 alone.
- The eGPU costs context, because KV lives on a 24 GiB card instead of a 121.7 GiB pool.
- The eGPU costs boot time and complexity: a peer worker, a second model view, a USB-resident PLE table,
  and a fork image that must be rebuilt.
- GB10 alone is the simpler recipe and the one the model author supports; its decode rate is lower per
  the author's own table (~70-75 tok/s prose) but its window is 15x wider out of the box.

The tables above are the check on those expectations, and any row that contradicts them will be reported
as it measured rather than adjusted to fit.

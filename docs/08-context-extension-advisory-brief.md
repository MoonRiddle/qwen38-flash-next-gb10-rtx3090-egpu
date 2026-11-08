# 8. Context extension: advisory brief (full context for a first-time reader)

This file is written for an outside AI or engineer who has no prior context. Reading only this document should
be enough to understand what is being measured, on which hardware, with which model, and where the work is
currently blocked.

- Section 0 is background: the model, the quantization, the hardware, the upstream repositories, the measurement
  contract, the vocabulary, and the history so far.
- Sections 1-9 are the measured record: the confirmed configuration, the numbers, the failed attempts with their
  exact errors, the upstream state, the open questions, and the artifacts.

Every number is copied from the artifacts in section 9 and from `docs/06-results.md`. Nothing is modelled. Where
a claim could not be confirmed against a primary source, it is marked unverified at that point. The Korean
edition is `08-context-extension-advisory-brief.ko.md`; the two carry the same numbers.

---

## 0. Background

### 0.1 One-sentence summary

This is an experiment to serve a 125B-parameter MoE model quantized to INT4 on two consumer-class devices — one
GB10 unified-memory box plus one RTX 3090 attached over PCIe — while pushing the context window as wide as the
hardware allows. The window is confirmed at 262,144 tokens. A 409,600-token attempt was made and failed on the
memory arithmetic of a 24 GiB card.

### 0.2 The questions this document answers

1. How far can this model be pushed on this hardware, specifically in **context window**?
2. What is lost when the window grows — decode rate, prefill rate, memory headroom?
3. To grow it further, which of the three resident objects must shrink or move, and is that move possible in
   this fork engine and the current vLLM build?

### 0.3 Why the hardware is unusual

This is not a normal multi-GPU server. The two devices have fundamentally different characters.

| Device | Character | Role in this experiment |
|---|---|---|
| GB10 (SM121) | CPU and GPU share one 121.7 GiB unified pool, 246 GB/s | Bulk of the model weights and the PLE table's page cache |
| RTX 3090 (SM86) | Separate 24 GiB GDDR6X, 833 GB/s, attached as an eGPU | Hot-expert cache and the KV cache. The real bottleneck |

The link is PCIe Gen4 x4 and **Non-P2P**. The two cards cannot read each other's memory directly; transfers go
through host RAM. Measured round trip is 33.14 µs at 20 KB, and `nvbandwidth` copy bandwidth is 6.20 GB/s
host→device and 6.59 GB/s device→host. Those figures are in `artifacts/recipe/sep-d.egpu.log`.

The 3090 carries a software unlock that makes BF16 compute 70-80% faster than stock (Zagitta/gpu-unlock,
142-157 TFLOPS). This recipe did not create that state and does not change it.

The storage is also unusual. The model's 51 GiB PLE table lives on USB and is read through mmap. There are two
byte-identical copies: a 10 Gbps Samsung T7 Shield and a 20 Gbps USB NVMe (52,259,870,014 bytes total each).

### 0.4 What the model is

The checkpoint is Hugging Face's `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`. It is a derivative of
the Qwen3.8-Flash-Next family, requantized to INT4 and partly retrained by one user.

| Item | Value |
|---|---|
| Total parameters | 125B |
| Active parameters per token | 4.8B (`A5B` names "about 5B active") |
| Structure | MoE. All 512 routed experts retained, 5 selected per token |
| Original routing | 10 selected per token, 6.0B active |
| Quantization | INT4 AutoRound (Intel AutoRound lineage, W4A16) |
| Healing | One always-on shared component, 37.75M parameters |
| Healing data | 14,667 examples of code, maths, general text, tool calls and long context |
| lm_head | int4 GPTQ-Marlin g128 (97.2% top-1 agreement with BF16); int8 variant 99.6% |
| Draft head | BF16 top-10, unchanged |
| Maximum position | `max_position_embeddings = 262,144` |

The model has several special components that recur throughout this document, so they are defined first.

- **MTP (Multi-Token Prediction) draft head.** A small head that proposes several tokens ahead for speculative
  decoding; the target model then verifies them, so one target forward pass can yield several tokens. This
  recipe uses depth 15 on the code path and a runtime-selectable depth on the prose path.
- **QSA ring.** One of the attention implementations. Its ring capacity follows the MTP depth: 20 rows at MTP 15,
  12 rows at MTP 5. The KV page size (`block_size`) must divide that capacity.
- **GDN layers.** State-carrying linear-attention layers, 36 of them.
- **PLE table.** A 51 GiB int4 n-gram lookup table that exists separately from the weights. It is used in Layer
  1 only and its input is the input ids. It does not fit beside the weights on one device, so it lives on USB and
  is mmapped.

### 0.5 What INT4 AutoRound is

AutoRound is a quantization method published by Intel that rounds weights to 4-bit integers while learning a
correction for the rounding error (W4A16: 4-bit weights, 16-bit activations). This checkpoint follows that
lineage and additionally reduces routing depth from 10 to 5 and attaches a healing component.

Critically, **no expert was deleted**. All 512 routed experts remain; only the per-token selection count dropped.
The common claim "the author pruned the prose experts" is not true of this checkpoint, and
`docs/02-why-prose-is-slower.md` corrects it with sources.

### 0.6 Upstream repositories

This work stands on several external repositories. Roles and pinned versions are listed so a first-time reader
can trace every number.

| Repository | Kind | Role here | Pinned |
|---|---|---|---|
| `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` | Hugging Face model | The served checkpoint | The checkpoint itself |
| `azampatti/Qwen3.8-Flash-Next-Int4-FAST` | Hugging Face model/recipe | The author's own recipe. Source of the acceptance figures | Documentation |
| `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ` | Hugging Face model | Alternative checkpoint for comparison | The checkpoint itself |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | Hugging Face model | Lineage original. Not loaded by this recipe | — |
| `MiaAI-Lab/sparkDash` | GitHub | Measurement harness. Source of every throughput figure | `7c2b9638b025cd39fb43f95c225de62980f130ee` |
| `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` | GitHub | Quality and correctness checks (MIT) | `20e15e41f2f514a0d287710f4665ee1b50099ba41` |
| `Saren-Arterius/qwen3.8-Flash-DGX-AutoRound` | GitHub | Base build for the engine image | Image tag `a5b-autoround-bottleneck-q2` |
| `Zagitta/gpu-unlock` | GitHub | RTX 3090 BF16 unlock. This recipe does not change it | — |

The engine is not stock vLLM; it is a fork build. `vLLM 0.1.dev20073+g8e685d198` is bind-mounted over with
`peer_worker.py` and `benchmark_worker.py`. Those two files decide the owner/helper device split, the trunk move,
draft placement, and the PLE lookup path.

The measurement harness `sparkDash` is used as **unmodified upstream bytes**, not as a fork. All nine files of
the measurement closure are byte-identical between the pinned commit and upstream HEAD (`da20ac5b`, v2.77.0), and
their SHA-256 digests are recorded in `bench/sparkdash-exact/source-lock.json`. The only file in this repository
that differs from upstream is the invocation wrapper `runner.mjs`.

### 0.7 What the recipe actually does

1. **Serves the checkpoint as-is.** No expert pruning, top-5 routing kept, 4.8B active.
2. **Keeps PLE outside the weights.** The 51 GiB table is mmapped from USB with `PREWARM=1`, `MADV_RANDOM=0`,
   32 mmap workers and 512 fast rows, so the model fits on one GB10.
3. **Runs a fork engine with resident speculative decoding.** Code depth 15 is fixed at the server; the prose
   depth is set over RPC (`k6-open` = depth 6, no gate).
4. **Measures with sparkDash.** Every envelope records the engine controls, so a number is always attributable to
   a configuration.

### 0.8 The measurement contract

Throughput figures come from MiaAI-Lab's `sparkDash` collectors. Decode is computed from streamed token
timestamps.

```
decodeTps = (completionTokens - 1) / (tLast - tFirst) * 1000
```

The first token is the prefill's output, so it is excluded from the decode window. The request is pinned so the
measurement cannot drift: `min_tokens = max_tokens`, `ignore_eos`, `stop: []`, 400 tokens by default, and one
32-token warmup per job.

| Workload | temperature | thinking | Prompt source |
|---|---:|---|---|
| `prose` | 0.7 | on | `DECODE_PROSE_PROMPT` — a hash-map explanation task |
| `code` | 0.0 | off | `DECODE_CODE_TASKS` |

Prefill attaches a unique salt header per attempt so prefix caching cannot help, then computes the rate from TTFT
minus a calibrated overhead.

```
rate = promptTokens / (ttftMs - overheadMs)
```

The overhead correction is capped at half the TTFT; above a 30% overhead share the row is flagged
`lowConfidence`. The overhead is measured, not assumed: 16 calibration tokens, 3 samples. Repeats are 3 samples
at ≤32K, 2 at ≤131K, 1 above that. Prefill takes no concurrency list, so a "C=4 prefill" number does not exist in
this method.

### 0.9 Vocabulary

| Term | Meaning |
|---|---|
| A5B | Name for "about 5B active parameters per token". Actual 4.8B |
| MoE | Mixture of Experts; only a subset of experts activates per token |
| routed expert | Expert chosen by the router per token. 512 exist, 5 are used |
| MTP | Multi-Token Prediction; the draft head for speculative decoding |
| draft / target | The small proposing path (draft) and the verifying body (target) |
| acceptance | Share of drafted tokens the target accepts. Directly sets decode speed |
| PLE | The int4 n-gram lookup table separate from the weights. 51 GiB, Layer 1 only |
| QSA ring | Attention ring buffer; capacity follows MTP depth (20 rows / 12 rows) |
| GDN | State-carrying linear attention layer family |
| block_size | KV page size in tokens. Sets both decode rate and KV memory efficiency |
| KV cache | Key/value store for past tokens during generation. The main window constraint |
| admission | The KV capacity the engine reports at start-up, e.g. `380,966 tokens, 1.45x` |
| 1.45x | KV holds 1.45 times the requested window (262,144) |
| TTFT | Time To First Token; the prefill's completion time |
| tok/s | Tokens per second |
| OOM | Out Of Memory; an allocation failure |
| eGPU | External GPU. Here a 3090 over PCIe Gen4 x4 |
| Non-P2P | The two GPUs cannot address each other's memory directly; transfers cross host RAM |
| owner / helper | The two device roles this fork defines. Owner holds the trunk and KV |
| AutoRound | Intel's learned 4-bit quantization method |
| W4A16 | 4-bit weights, 16-bit activations |
| healing | Retraining step that compensates quantization loss by KL self-distillation |
| envelope | The immutable JSON record sparkDash writes per job |

### 0.10 History so far

| Step | Window | KV | block | batch | graphs | Result |
|---:|---:|---:|---:|---:|---|---|
| Baseline | 17,408 | 4 GiB | 160 | 8192 | PIECEWISE | Baseline recipe. Admission 33,572 tokens, 1.93x |
| Page test | 17,408 | 4 GiB | 1600 / 1632 | 8192 | PIECEWISE | Decode falls to 73-76 tok/s. Page size isolated as the cause |
| First extension | 204,800 | 12 GiB | 160 | 2048 | NONE | Works. Admission 299,323 tokens, 1.46x, zero OOM |
| Second extension | 262,144 | 14 GiB | 160 | 2048 | NONE | Works. Model maximum. Admission 380,966 tokens, 1.45x |
| Third attempt | 409,600 | 16 GiB | 160 | 2048 | NONE | Admission 481,483 tokens, 1.18x, then OOM |
| Restore | 262,144 | 14 GiB | 160 | 2048 | NONE | Re-confirmed. Admission 380,966 tokens, load 220.14 s |

Three rules were established along the way.

1. **Page size decides decode.** With MTP depth and window held fixed, changing block to 1600 or 1632 drops decode
   to 73-76 tok/s, while block 160 reaches 192-194 tok/s. The kernel processes every allocated row of the page, so
   a large page wastes work. `block_size` must also divide the QSA ring capacity, which follows MTP depth: 20 rows
   at MTP 15, so block must be a multiple of 20, and 1632 is MTP-5 only.
2. **Prefill batch is 2048.** At 8192 the workspace does not fit beside KV and the engine core dies at 32K prefill.
   At 2048 every size up to 261K completes.
3. **KV is pinned to device 0.** This fork's worker allocates KV and weights against device 0 regardless of the
   load device, so KV size is capped by the 24 GiB card rather than by the GB10's 121.7 GiB pool.

---

## 1. Hardware and software

| Item | Value |
|---|---|
| Host | JGX10-1, DGX Spark class |
| Device 1 (owner) | RTX 3090, GPU-unlocked, SM86, 24,576 MiB GDDR6X, 833 GB/s |
| Device 0 (vLLM default owner) | GB10, SM121, 121.7 GiB unified, 246 GB/s |
| Link | PCIe Gen4 x4, Non-P2P, 33.14 µs round trip at 20 KB |
| Measured copy bandwidth | host→device 6.20 GB/s, device→host 6.59 GB/s |
| Host RAM | 121 GiB (`MemAvailable` ≈ 118 GiB before load) |
| PLE table | 51 GiB int4 n-gram, mmapped from USB, two identical copies |
| Engine image | `qwen38-flash-dgx:a5b-autoround-bottleneck-q2` |
| vLLM | `0.1.dev20073` (fork build) |
| PyTorch | `2.13.0+cu130` |
| Triton | `3.7.1` |

The engine is a fork: `peer_worker.py` and `benchmark_worker.py` are bind-mounted over the image's copies. The
fork splits work between an owner device and a helper device, moves the trunk to the owner, and keeps the draft
head and PLE lookups where the fork decides.

## 2. Confirmed configuration (262,144)

```
--max-model-len 262144 --max-num-seqs 1 --kv-mib 14336 --block-size 160 \
--max-num-batched-tokens 2048 --mtp 15 --graphs NONE
```

Admission: `GPU KV cache size: 380,966 tokens, Maximum concurrency for 262,144 tokens per request: 1.45x`.
`engine_load_seconds 218.27` (a repeat boot measured 220.14 s).

The 3090 sits at 22,913 of 24,576 MiB (93%) and host RAM at 69.7 of 121.7 GiB. Usable card headroom is 1.6 GiB.

## 3. Measured numbers at 262,144

Decode, C=1:

| Workload | Rate |
|---|---:|
| Code (temp 0, reasoning off) | 194.70 tok/s cold |
| Prose (temp 0.7, reasoning on) | 33.18 tok/s cold, then 76.77 / 73.61 warm |

Prefill at batch 2048:

| Size | promptTokens | samples | prefillTps | TTFT |
|---|---:|---:|---:|---:|
| 4K | 4,136 | 3 | 2,814.22 | 1.58 s |
| 16K | 16,423 | 3 | 2,723.52 | 6.14 s |
| 32K | 32,805 | 3 | 2,649.84 | 12.49 s |
| 200K | 200,041 | 1 | 2,331.81 | 85.90 s |
| 261K | 261,036 | 1 | 2,240.99 | 116.59 s |

The 204,800 configuration (KV 12 GiB) measured code 192.28, prose 76.12 / 68.92 warm, prefill 4K 2,844.53 /
16K 2,727.28 / 32K 2,657.05 / 200K 2,345.07 (TTFT 85.41 s), admission 299,323 tokens at 1.46x, and zero OOM
events. Going from 204,800 to 262,144 cost nothing measurable: every figure sits within run-to-run noise.

Prose is about a third of code because of draft acceptance. The model author publishes 75% overall acceptance at
T=0.5 (85% code, 63% prose, 89% long context), and the forum category table is CODE 82.1%, MATH 76.8%,
LONGCTX 85.5%, TOOL 68.9%, GENERAL 55.7%. Prose rejects roughly two of every five drafted tokens, and the
thinking path spends most of its tokens in exactly that distribution, so the rate lands low. The sourced causal
chain is in `docs/02-why-prose-is-slower.md`.

## 4. The ladder actually run

| Window | KV | block | batch | graphs | Result |
|---:|---:|---:|---:|---|---|
| 17,408 | 4 GiB | 160 | 8192 | PIECEWISE | baseline recipe |
| 204,800 | 12 GiB | 160 | 2048 | NONE | works, 1.46x admission |
| 262,144 | 14 GiB (14,336 MiB) | 160 | 2048 | NONE | works, 1.45x admission |
| 409,600 | 16 GiB (16,384 MiB) | 160 | 2048 | NONE | admitted, then OOM |

## 5. Why 409,600 fails on the card

With `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, the engine accepts `--max-model-len 409600` and reports
`GPU KV cache size: 481,483 tokens, Maximum concurrency for 409,600 tokens per request: 1.18x` at `--kv-mib 16384`.
It then dies during start-up:

```
memory allocation failed with OOM on device 0   (1.28 GB request, 44 MB free)
```

The three residents do not fit together:

| Component | GiB |
|---|---:|
| Draft weights on the 3090 | 9.09 |
| Target weights on the 3090 | 6.19 |
| KV at 16 GiB | 16.00 |
| Sum | 31.28 |

The card holds 24 GiB. At 262,144 the same three components total 29.28 GiB on paper but measure 22.38 GiB
because vLLM releases part of the weight staging buffer after load, so the reservation figure is an upper bound.

## 6. Escape routes tried

**Route 1: move the draft head to the GB10.** `peer_worker.py` was edited so the draft request loads on the
helper device. The measured placement still reported `primary_bytes 9,757,622,784` on device 0, and the engine
log confirmed the draft stayed there. vLLM's worker allocates KV and weights against device 0 regardless of the
load device, so the edit had no effect.

**Route 2: enable prefix caching.** Without it, a window beyond the model maximum is rejected with
`--mamba-block-size can only be set with --enable-prefix-caching`. Enabling prefix caching passes that check and
still reaches the same CUDA OOM.

**Route 3: make the GB10 device 0.** `QWEN38_VISIBLE_DEVICES` was set to GB10 first, with `QWEN38_OWNER=1`,
`QWEN38_HELPER=0` and `QWEN38_SKIP_DEVICE_CHECK=1`. The role assertions pass and the engine reaches model load.
Three errors follow in order:

1. `Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on
   cuda:1`, from `vocab_parallel_embedding.py`'s `embedding`. The trunk move places the embedding table on the
   owner while the input ids are built on device 0.
2. Moving `VocabParallelEmbedding` to the helper clears that, but Triton rejects the table with
   `Pointer argument cannot be accessed from Triton (cpu tensor?)`.
3. Wrapping the embedding forward in `torch.cuda.device(input_.device)` and keeping the table on the 3090 gets
   past Triton, but the first `index_select` then fails with the same device split, because the ids exist on
   cuda:0 before the embedding is entered.

The device split is created before this call site, so it cannot be repaired inside the embedding. The engine was
restored to the default device order afterwards, and the 262,144 configuration was confirmed again
(`GPU KV cache size: 380,966 tokens`, load 220.14 s).

## 7. Upstream state

- The SM121/GB10 support pull request in vLLM (#31740) was closed unmerged on 2026-11-01.
- The installed vLLM is a development build (`0.1.dev20073`), not a tagged release.
- Triton's "cannot be accessed from Triton" class of error is commonly caused by launching a kernel under a
  current device that differs from the tensors' device, not by unified memory itself. Route 3 step 3 supports
  that reading: fixing the current device cleared the Triton error and exposed the earlier device split.
- The fork's `load_device` change had no effect. The worker's device-0 allocation rule lives outside the fork.

## 8. Open questions for an advisor

1. Is there a supported way in this fork to place KV on the unified-memory device while weights stay on the
   3090, without moving device 0? The fork's owner/helper split was not designed for that.
2. Is the 409,600 window reachable by shrinking one of the three residents rather than by moving KV, for example
   a smaller draft head, a tighter KV dtype, or a lower target weight footprint?
3. Does any released vLLM tag after `0.1.dev20073` change the device-0 allocation rule for KV, or accept a
   device-0 that is unified memory?
4. If the window stays at 262,144, is there a measurable benefit to the 204,800 configuration (1.6 GiB more
   headroom) for workloads that never exceed 200K?
5. Given this Non-P2P link (33.14 µs round trip, 6.2/6.6 GB/s), is sharding KV across the two devices ever
   worthwhile, or does the host-RAM hop always cost more than it saves?

## 9. Artifacts

- `artifacts/recipe/sep-d.json` — the 409,600 run record (`status: failed`, `max_model_len: 409600`).
- `artifacts/recipe/sep-d.server.log` — engine log of the last 409,600 attempt, which ended at the device split
  described in section 6 route 3.
- `artifacts/recipe/sep-d.egpu.log` — PCIe link state and AER counters taken during that attempt: Gen4 x4 for
  the 3090, 3 correctable timeouts on the device, 188 correctable errors on the root port, zero fatal errors.
  The `nvbandwidth` copy figures quoted in 0.3 are in this file too.
- `artifacts/recipe/` also holds the sparkDash envelopes for the 17,408 recipe runs (`r5f`, `r5g`), which carry
  the decode and prefill rows quoted in `docs/06-results.md`.
- `artifacts/reproduction/`, `artifacts/longprefill/`, `artifacts/storage/` and `artifacts/verify/` hold the
  reproduction, long-prefill, storage A/B and verification results.

The `481,483` admission line and the `memory allocation failed` line quoted in section 5 were printed by the
engine during the first 409,600 attempt. That attempt's server log was not archived before the engine was
restarted, so the quotes are transcribed from the live run rather than linked to a file. The archived
`sep-d.server.log` is from the later attempt, which stops earlier, at the embedding device split.

- `docs/06-results.md` sections "204,800 window at block 160", "262,144 window", "Forcing 409,600" and
  "Swapping device 0 to the GB10" carry the same numbers with their surrounding context.
- `docs/07-gb10-solo-vs-egpu.md` section "Why the GB10 cannot take over KV in the eGPU configuration" carries
  the device-0 swap barriers.
- `docs/01-measurement-method.md` carries the full measurement contract and
  `docs/02-why-prose-is-slower.md` carries the sourced prose analysis.

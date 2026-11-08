# 3. Cold-start analysis (from PDF 9)

This document is a transcription and commentary of `analysis/GB10-and-RTX3090-decoding-optimization-9.pdf`.
The PDF is the authoritative artifact; this page restates its structure in English and marks which parts
this repository's engine actually implements.

PDF 9's own convention is preserved: numbers it models rather than measures carry `[INFERENCE]`. Do not
read those as measured throughput.

## The problem PDF 9 addresses

On this machine the same engine produces very different prose rates depending on how warm it is. A first
run after boot lands well below the target; only after several runs does the rate reach its steady value.
PDF 9 calls the two regimes *short-warmup* and *long-warmup* and argues that the gap is not a hardware
limit but five specific cold-start penalties that can be moved from measurement time to boot time.

The observation is consistent with what this repository measured independently: prose decode went
9.17 → 43.81 → 46.30 tok/s across three cold jobs, then 95.78 → 87.35 → 82.50 tok/s on the same engine
once the kernels were compiled (see `06-results.md`). That is the same shape PDF 9 describes.

## The five cold bottlenecks

| Rank | Bottleneck | What happens on a short warmup | Why a long warmup removes it | PDF 9's stated loss, Run 1 |
|---|---|---|---|---|
| Cold-1 (largest) | Cold PLE page faults on 10 Gbps USB storage | With `PREWARM=0`, `MADV_RANDOM=1`, `FAST_ROWS=512`, the first prose request hashes n-grams that are not in the page cache and issues hundreds of 4 KiB random reads on the USB SSD, each stalling 300–500 µs | Repeated prose runs accumulate the common natural-language n-gram pages in the idle GB10 page cache, and the disk I/O disappears | −8.0 to −14.0 tok/s (3–5 ms per round) |
| Cold-2 | Delayed CUDA graph capture and Triton JIT for partial-acceptance row sizes | A code warmup only exercises the full-acceptance path (`M=6`). The prose run then meets 0-, 1- and 2-acceptance row counts for the first time, and helper CUDA graph capture plus Triton JIT/autotuning run *inside* the measurement | After enough prose runs every row combination's graph and kernel is registered and the stalls stop | −6.0 to −12.0 tok/s (0.8–2.0 s hitch) |
| Cold-3 | Target INT4 `lm_head` vs draft BF16 `lm_head`, plus unhealed MTP rejecting the `<think>` opening | The target verifies with an AutoRound INT4 head (97.2% top-1 agreement) while the draft head is BF16, so quantisation noise points different ways. In the first prose request's `<think>` opening, the mismatch plus a `DRAFT_SCALE` distortion concentrates immediate rejections | As the context grows and the text enters low-entropy body prose, the logit gap widens and the mismatch stops mattering | −5.0 to −9.0 tok/s (0.25 tokens per round) |
| Cold-4 | Prose experts absent from the initial hot-expert cache | After a code-only warmup the RTX 3090 hot-expert cache holds Python-syntax experts; the prose reasoning experts are missing, so all four rows are shipped to GB10 for the whole first run | Accumulated prose runs warm the prose experts into the cache and L2 | −4.0 to −7.0 tok/s |
| Cold-5 | CUDA allocator block splitting and clock ramp after a short warmup | A 64–128 token warmup allocates KV and GDN state buffers only to that length; when the measured prose run passes token 129 toward 400–600, the PyTorch caching allocator allocates and splits segments mid-measurement | One 600-token generation leaves the memory pool expanded and the GPU at P0 | −2.0 to −4.0 tok/s |

PDF 9 also cites MiaAI-Lab's own documentation (`docs/tp4.md`), which tells benchmarkers to discard the
first two runs after boot because of the cold Engram row cache. So the cold-start penalty is a known
property of this model family, not a defect of this machine.

## The six pre-bake recipes

| Recipe | Move | What it does | PDF 9's effect `[INFERENCE]` |
|---|---|---|---|
| ZW-1 | Exhaustive shape pre-compilation | At model-load time, force one forward pass and graph capture for every verification row size `M ∈ {1..6}` and every GB10 helper row size `M_cold ∈ {1..6}`, plus the temp-0.7 `RejectionSampler` kernels (Top-64 subspace softmax, `generate_uniform_probs`, entropy/block verify) and GDN state rollback (`RecoverSSM`), so Triton JIT finishes at boot and the kernel cache is frozen | +8 to +12 tok/s on the first prose run, at a cost of about 1.5 s of boot time |
| ZW-2 | Boot-time PLE pre-warm | Raise `VLLM_PLE_MMAP_FAST_ROWS` from 512 (80 KiB) to 262,144 rows (41.94 MiB) and fill the RTX 3090 copy with one sequential 42 MB read (≈0.05 s even on the T7), and pre-load the hot 40–45 GiB region of the 51.2 GiB table into idle GB10 RAM with `posix_fadvise(POSIX_FADV_WILLNEED)` or `vmtouch -t`, with `MADV_RANDOM` cleared, so the first prose request has zero USB page faults | Removes Cold-1 |
| ZW-3 | Shared INT4 lm_head pointer | Alias the draft head's `lm_head` to the target's INT4 GPTQ-Marlin instance on the RTX 3090, so both compute logits from bit-identical int4 weights; the draft also stops reading a 1.27 GB BF16 head each step, and 1.27 GiB of VRAM is freed — enough to keep one whole extra layer's 512 experts (1.21 GiB) resident | Removes Cold-3; PDF 9 models 105.9 tok/s with ZW-3 + ZW-5 |
| ZW-4 | Static prose-first hot-expert map | Build an offline 48-layer expert-ID table weighted 70% prose (thinking + body) and 30% code, save it as `static_prose_hot_map.pt`, and pin it into the RTX 3090 hot cache at `t = 0`, so the first prose token already sees 65–72% hot-expert hit rate | Removes Cold-4 |
| ZW-5 | Undistorted temp-0.7 sampler | For temperature > 0, draft selection stays argmax but the `draft_probs` fed to the rejection sampler are computed with `SCALE=1.0` and `T=0.7` over the target's top-64 subspace instead of the full 248,320 vocabulary; `<think>` spans use entropy verify (`posterior_threshold=0.07`, `alpha=0.3`) or block verify; BF16 KV cache and BF16 GDN state (`MAMBA_SSM_CACHE_DTYPE=bfloat16`) keep deep-draft acceptance and make `RecoverSSM` a pointer swap | Improves `<think>`-span acceptance |
| ZW-6 | 600-token pool pre-touch and clock lock | With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, allocate and free a 1,024-token dummy KV/GDN slot during the ZW-1 sweep so the allocator pool and GPU clock are already at P0 before measurement | Removes Cold-5 |

## PDF 9's projected short-warmup table `[INFERENCE]`

| Stage | Short-warmup round time | Confirmed tokens per round | Short-warmup prose (Run 1) | Warm prose (Run 5+) | Code (Run 1) |
|---|---:|---:|---:|---:|---:|
| Current cold state | 31.0 ms | 2.125 | 68.5 tok/s | 85.0 tok/s | ~220.0 tok/s |
| + ZW-1 & ZW-6 | 27.2 ms | 2.125 | 78.1 tok/s | 85.5 tok/s | 222.0 tok/s |
| + ZW-2 & ZW-4 | 24.5 ms | 2.150 | 87.8 tok/s | 88.5 tok/s | 225.0 tok/s |
| + ZW-3 & ZW-5 | 22.2 ms | 2.350 | 105.9 tok/s | 106.5 tok/s | 238.0+ tok/s |

None of these four rows has been measured on this machine. They are PDF 9's model.

## What this repository actually implements

The engine carries a boot-time pre-bake hook, `_zw_prebake` in `engine/resident_bench.py`, gated by
`QWEN38_ZW_PREBAKE`. It runs ZW-1 dummy sweeps (`M = 1..6`), the temp-0.7 lane, the ZW-6 600-token KV
pre-touch and a ZW-4 prose hot seed.

Two facts matter for reading the results in `06-results.md`:

1. `run_engine_hostnet.sh` does not pass `QWEN38_ZW_PREBAKE` into the container. The variable must be
   added both to the launcher's `export` list and to its `docker run -e` list. The reproduction and
   recipe runs in this repository therefore ran with the pre-bake **off**, which is exactly why the
   first prose jobs show the Cold-2 JIT penalty.
2. The reproduction run's engine log confirms it directly: `Triton kernel JIT compilation during
   inference: _qsa_mqa_paged_kernel`, `_compute_local_logits_stats_kernel`, `_rejection_kernel`,
   `_resample_kernel`, `_gumbel_sample_kernel`, `_qsa_sparse_paged_gqa_splitk_kernel`,
   `_qsa_pre_indexer_kernel` and `_expand_qsa_indices_kernel`, all firing between 17:01 and 17:04 while
   the prose jobs were running. The code jobs ran after the last of those warnings and reproduced the
   earlier numbers almost exactly.

So the practical recipe for a stable first-run prose number on this machine is: enable the pre-bake
hook, or accept that the first two prose jobs are warmup and discard them, which is what MiaAI-Lab's
own docs say to do.

## Not claimed

- ZW-2's 262,144-row VRAM seed and GB10 RAM pre-load are not implemented in this repository.
- ZW-3's shared INT4 head pointer is not implemented; the draft head stays BF16.
- ZW-5's top-64 undistorted sampler and entropy verify are not implemented.
- The projected numbers above are PDF 9's, not measurements from this machine.

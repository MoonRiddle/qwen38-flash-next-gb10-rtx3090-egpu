# Third-party notices

This repository contains code and data derived from, or required by, the following projects. Each keeps
its own license. Nothing in this repository relicenses them.

## sparkDash — Apache License 2.0 (MIT for pre-2.0.0 code)

- Upstream: https://github.com/MiaAI-Lab/sparkDash
- Copyright 2026 Mia's AI Lab
- Pinned commit: `7c2b9638b025cd39fb43f95c225de62980f130ee`
- Scope here: the nine measurement files vendored byte-for-byte in `bench/sparkdash-exact/`
  (`server/collectors/DecodeBench.js`, `server/collectors/PrefillBench.js`,
  `server/collectors/LlmStreaming.js`, `server/collectors/contextFill.js`,
  `server/util/atomicWrite.js`, `src/shared/llmPrompts.js`, `src/shared/llmTarget.js`,
  `src/shared/prefillBench.js`, `src/shared/contextFit.js`), plus its `LICENSE`, `LICENSE-MIT` and
  `NOTICE`, which are copied unchanged.

Upstream's own NOTICE reads:

> sparkDash / Copyright 2026 Mia's AI Lab / This product is licensed under the Apache License,
> Version 2.0 (see LICENSE) from version 2.0.0 onward. Versions up to and including 1.9.0 were
> released under the MIT License. The MIT License text and its copyright notice are kept in
> LICENSE-MIT and continue to apply to the code written for those versions, which remains part of
> this codebase.

`bench/sparkdash-exact/runner.mjs` is this repository's own MIT-licensed invocation wrapper. It is not
an upstream file; it calls the upstream managers through their public API and does not modify the
vendored collectors.

## TensorFold recipe (MiaAI-Lab) — MIT License

- Upstream: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
- Copyright (c) 2026 MiaAI-Lab
- Pinned commit: `20e15e41f2f514a0d287710f4665ee1b5009ba41`
- Scope here: `verify/client.py`, `verify/exact.py`, `verify/needle.py`, `verify/toolcheck.py`,
  `verify/prompt_reuse.py`, `verify/context_boundary.py`, adapted as described in
  `docs/05-verification.md`. `verify/exactness_cross.py` is this repository's own MIT code.

That upstream project also builds on TensorFold (https://github.com/ashhart/TensorFold, Apache-2.0
from 0.6.0, MIT before that). This repository does not build or redistribute TensorFold; it only uses
MiaAI-Lab's verification tools.

## Qwen3.8-Flash-Next model weights — Qwen Community License 1.0

- Upstream: Alibaba Qwen / Qwen3.8-Flash-Next, model card and `LICENSE` on the Hugging Face repo
- License: **Qwen Community License 1.0**, not Apache-2.0. The license carries use and redistribution
  conditions that differ from Apache-2.0, so read the official text before any commercial use or
  redistribution: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
- Scope here: not redistributed. The recipe expects the weights to be present locally.

## azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound — quantized checkpoint

- Upstream: https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound
- Base model: `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`
- License: the card declares `license: other`, `license_name: qwen`, and links the Qwen Community
  License 1.0: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE. That license governs
  the checkpoint, and it is not Apache-2.0. Read it before redistributing the weights.
- Scope here: not redistributed. Referenced as the checkpoint this recipe serves.

## Saren-Arterius/qwen3.8-Flash-DGX-AutoRound — engine build base

- Upstream: https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound
- Scope here: not redistributed as source. `engine/Dockerfile.bottleneck` builds `FROM` an image
  produced from that project. Its README notes active development on the `magi-v5-public` branch.
  Consult that project for its license before building or redistributing the image.

## vLLM — Apache License 2.0

- Upstream: https://github.com/vllm-project/vllm
- Scope here: the engine image contains a vLLM build (`0.1.dev20073+g8e685d198`). The patches in
  `engine/patches/` are this repository's MIT work applied to that tree; the vLLM code they modify
  stays under Apache-2.0.

## Zagitta/gpu-unlock

- Scope here: not redistributed. The RTX 3090's BF16 unlock is a property of the machine, recorded in
  the hardware table so a reader knows why an SM86 card reports BF16 throughput. Apply that project's
  own instructions and license if you reproduce it.

## Text and data

`analysis/GB10-and-RTX3090-decoding-optimization-9.pdf` is the author's own analysis document,
included byte-for-byte. Its cleaned transcription in the same directory is derived from it and keeps
the same authorship. Numbers in that document marked `[INFERENCE]` are its models, not measurements
from this repository.

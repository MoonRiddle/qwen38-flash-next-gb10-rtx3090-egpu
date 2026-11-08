# Deployment map: what runs on the GB10 and what runs on the RTX 3090

`docs/18-deployment-map.svg` is the diagram. This page explains the two placements it draws.

## Fixed placement in both configurations

| Component | Device | Reason |
|---|---|---|
| Target weights, helper side | GB10, 62.8 GB resident | The body does not fit on 24 GiB, and the GB10 has 121.7 GiB of unified memory |
| PLE n-gram table | GB10, 51 GiB mmapped from USB | It is read-mostly and lives outside the weights |
| KV cache | RTX 3090, 14 GiB | The engine reserves KV on device 0, which is the 3090 |
| Draft MTP weights | RTX 3090 | The draft runs on the same device as the target's owner step |
| Hot-expert cache | RTX 3090, 4 GiB owner budget | Hot routed experts stay close to the decode step |

The link is PCIe Gen4 x4 and non-P2P, so every activation that crosses devices goes through
host memory. That is why the target body stays on the GB10 and only the per-step working set
stays on the 3090.

## A. 409,600 single request

- `--max-model-len 409600`, `--max-num-seqs 1`, `--kv-mib 14336`.
- KV admits `427,468 tokens, 1.04x`, which is exactly one full request.
- Decode medians: code 185.89, prose 67.28.
- The 3090 holds KV 14 GiB plus the draft and the hot cache. The GB10 holds the body and PLE.

## B. 340,000 two requests

- `--max-model-len 340000`, `--max-num-seqs 2`, `--kv-mib 15360` for the decode medians,
  and `--kv-mib 14336` for the long-prompt probe.
- KV admits `677,901 tokens, 1.99x` at 15 GiB and `631,728 tokens, 1.86x` at 14 GiB.
- Decode medians at C=2, measured at KV 15 GiB: code 86.79 per request (163.46 aggregate),
  prose 40.16 per request (78.35 aggregate).
- Two *full-length* 340,000 requests do not fit: a two-stream probe at 339,967 input tokens
  each was rejected with `HTTP 400`.

## What the diagram does not claim

The long-prompt probe completed two 315,000-token requests, but it was non-streaming and
recorded only wall time. It did not record TTFT overlap, scheduler KV occupancy or
preemption, so the diagram shows configured slots, not proven simultaneous residency.

Short-prompt concurrency is separate and measured: C=2 aggregate 260.81 for code and 86.60
for prose, C=4 aggregate 302.98 and 115.28, all with `streamsFailed == 0`.

## Regenerating

The SVG is hand-written and static. Edit `docs/18-deployment-map.svg` directly. It uses
literal hex colors rather than `currentColor` or `var(--…)`, because the rasterizer used to
preview it ignores theme variables.

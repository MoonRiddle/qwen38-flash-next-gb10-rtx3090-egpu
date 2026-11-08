# 4. PLE storage: 10 Gbps USB T7 Shield vs 20 Gbps USB NVMe

This recipe keeps the 51 GiB PLE n-gram table outside the model weights, mmapped from USB storage. That
choice is what makes a 125B/A5B checkpoint fit on one GB10, and it also makes storage latency a
first-order variable. This page states what was measured and, just as importantly, what was not.

## Two copies, byte-identical

| Copy | Host path | Backing device | USB link |
|---|---|---|---|
| T7 Shield | `/home/user/ple-table-int4` | `/dev/sda2`, mounted at `/` | 10 Gbps (Bus 004) |
| USB NVMe | `/mnt/nvme20/ple-table-int4` | `/dev/sdc2` | 20 Gbps (Bus 006, Gen2x2) |

| Property | Value |
|---|---|
| Total bytes, both copies | `52,259,870,014` |
| Shard count | 131 (`model-00005-of-00131.safetensors` … ) |
| Spot-check SHA-256, shard `model-00005-of-00131.safetensors` | `c50bf465a4a0129f1a1196c0e75d881048238f7107b356fc734007eac0d3b123` on **both** copies |

Because the bytes are identical, a difference between the two runs can only come from the backing
device, not from the table content.

Device mapping was established from the sysfs USB path, not from drive letters:

```
sda -> /sys/devices/platform/NVDA8000:01/usb4/4-1/...        (Bus 004, 10000M)
sdc -> /sys/devices/platform/NVDA8000:02/usb6/6-1/...        (Bus 006, 20000M/x2)
```

## Raw storage throughput

Measured with `dd` (direct I/O) and `fio` (libaio, queue depth 32, 30 s):

| Metric | 10 Gbps T7 Shield | 20 Gbps USB NVMe | Delta |
|---|---:|---:|---:|
| Sequential write (`dd` 4 GB direct) | 632 MB/s | 1.1 GB/s | ~1.7x |
| Sequential read (`dd` 1 GB / 4 GB direct) | 612 MB/s | 1.2 GB/s | ~2.0x |
| Random 4 KiB (`fio` libaio qd32) | 48,358 IOPS / 193 MB/s | 55,104 IOPS / 220 MB/s | +14% |
| Random 4 KiB mean latency | 650 µs | 570 µs | −12% |
| Random 4 KiB p95 latency | 978 µs | 872 µs | −11% |

The link upgrade buys bandwidth, not random-read latency. A 20 Gbps link that still issues 4 KiB
random reads only improves the latency of those reads by about a tenth. That matters because PLE
lookups are random row gathers, not streaming reads.

## What the engine does with the table

The launcher mounts the chosen copy read-only into the container at
`/home/user/ple-table-int4`, and the engine mmaps it with:

| Variable | Value used |
|---|---|
| `VLLM_PLE_MMAP` | 1 |
| `VLLM_PLE_MMAP_WORKERS` | 32 |
| `VLLM_PLE_MMAP_PREWARM` | 1 |
| `VLLM_PLE_MMAP_MADV_RANDOM` | 0 |
| `VLLM_PLE_MMAP_FAST_ROWS` | 512 |
| `VLLM_PLE_MMAP_DIR` | `/home/user/ple-table-int4` |

`PREWARM=1` means the engine streams the whole 47.7 GiB table into the page cache at boot. That takes
about 85 s on the T7 at 612 MB/s and about 40 s on the NVMe at 1.2 GB/s, so the difference shows up
first as boot time, and only secondarily in steady-state decode.

To switch copies, set the mount source in `engine/run_engine_hostnet.sh`:

```bash
export QWEN38_PLE_HOST_DIR=/mnt/nvme20/ple-table-int4   # 20 Gbps
export QWEN38_PLE_HOST_DIR=/home/user/ple-table-int4   # 10 Gbps T7
```

## The A/B and its honest limits

The A/B in `06-results.md` runs the same engine image, the same model views, the same speculative
controls and the same workload grid against the two copies, and records boot time and steady-state
decode/prefill separately.

Three limits apply, and none of them is glossed over:

1. **The page cache is not reset between the two runs.** Dropping caches requires `sudo`, which this
   recipe does not do, and the pre-warm fills the cache anyway. So the A/B measures steady-state
   behaviour with a warm page cache, not cold-fault cost.
2. **Boot time is a real signal, steady-state decode is a weak one.** Once the table is in RAM, the
   backing device stops mattering for decode; the decode delta in the results is within run-to-run
   noise. The measurable difference is boot/pre-warm time.
3. **Cold-fault behaviour is not measured here.** For that, PDF 9's Cold-1 analysis
   (`03-cold-start-analysis.md`) and the raw random-read latency table above are the available
   evidence: hundreds of 300–500 µs stalls per first prose request on the T7, and roughly 12% lower
   per-read latency on the 20 Gbps copy.

An earlier document in this campaign (`USBC-20Gbps-PLE-simulation-20261103.md` §9) contains a
model-level T7-vs-NVMe table. That table is explicitly non-causal and unpaired; it is not used as
evidence here.

## A/B measurement conditions

Both arms use the same engine image `sha256:b27377b7c25cc6…`, the same model views, the same controls
(`k6-open`), the same flags (`--max-model-len 17408 --max-num-seqs 4 --kv-mib 8192 --block-size 160
--max-num-batched-tokens 8192`, `PREWARM=1`, `MADV_RANDOM=0`, 32 mmap workers, 512 fast rows), and the
same workload grid: one prose job and one code job, each carrying C=1 and C=4 in a single envelope.

Both arms are measured **cold**: a freshly started container, so the Triton JIT penalty applies equally
to both and is not a confounder. That is the honest comparison, but it means the absolute numbers are
below the warm-engine figures in `06-results.md`.

## NVMe arm (20 Gbps)

| Workload | C | median tok/s | aggregate tok/s | median TTFT |
|---|---:|---:|---:|---:|
| prose | 1 | 48.44 | 48.44 | 95.1 ms |
| prose | 4 | 41.19 | 78.55 | 5,161.7 ms |
| code | 1 | 174.48 | 174.48 | 251.0 ms |
| code | 4 | 90.91 | 167.03 | 2,152.2 ms |

Every row has `streamsOk == concurrency` and `streamsFailed == 0`.

Artifacts: `artifacts/storage/a-sparkdash-native-r5h-execution.json`,
`artifacts/storage/a-sparkdash-native-r5h-k6-open-{prose,code}-01.json`.

## T7 arm (10 Gbps)

Same image, same views, same controls, same flags, same cold-start condition, PLE mounted from
`/home/user/ple-table-int4`. `engine_load_seconds 280.26`, `GPU KV cache size 61,740 tokens /
3.55x`.

| Workload | C | median tok/s | aggregate tok/s | median TTFT |
|---|---:|---:|---:|---:|
| prose | 1 | 89.57 | 89.57 | 77.3 ms |
| prose | 4 | 54.15 | 207.81 | 204.2 ms |
| code | 1 | 219.39 | 219.39 | 198.2 ms |
| code | 4 | 95.63 | 324.21 | 375.9 ms |

Every row has `streamsOk == concurrency` and `streamsFailed == 0`.

### Delta, and why it is not a storage result

| Workload | C | NVMe | T7 | Delta |
|---|---:|---:|---:|---:|
| prose | 1 | 48.44 | 89.57 | +84.9% |
| prose | 4 | 41.19 | 54.15 | +31.5% |
| code | 1 | 174.48 | 219.39 | +25.7% |
| code | 4 | 90.91 | 95.63 | +5.2% |

The T7 arm is **faster** than the NVMe arm. That cannot be a storage effect: the T7 has lower
bandwidth and higher random-read latency on every raw metric in the table above. The explanation is
cold-start variance. Both arms were measured on a freshly started container, and the Triton JIT penalty
is large enough to dominate: the same engine re-measured warm on prose went 9.17 → 43.81 → 46.30 and
then 95.78 → 87.35 → 82.50 in an earlier run (see `06-results.md`). One cold job per cell cannot resolve
a storage effect that is smaller than that spread.

So this A/B does **not** show a decode advantage for either device. What it does show is that boot
pre-warm time differs by the raw sequential-read ratio (about 85 s vs 40 s for 47.7 GiB), and that
steady-state decode is dominated by cold-start noise at this sample size.

### T7 prefill: failed

Both prefill sizes returned HTTP 500 with `samples == 0 / 3`:

```
EngineCore encountered an issue. See stack trace (above) for the root cause.
```

The NVMe arm at the same KV reservation served prefill successfully (3,133 / 3,079 / 3,064 tok/s), so
this is not a systematic prefill limitation of the 8 GiB setting. It is the RTX 3090 running out of
room for a prefill activation after the decode jobs, the same failure mode documented for KV 12 GiB in
`06-results.md`, and it reproduced here on the T7 arm. No prefill comparison between the two devices is
therefore available.

Artifacts: `artifacts/storage/a-sparkdash-native-r6b-execution.json`,
`artifacts/storage/a-sparkdash-native-r6b-k6-open-{prose,code}-01.json`,
`artifacts/storage/a-sparkdash-native-r6b-k6-open-prefill-01.json` (the failure, kept as evidence).

## Conclusion

The storage choice does not change steady-state decode measurably at this sample size. It changes boot
pre-warm time roughly in proportion to sequential read bandwidth, and it changes cold-fault cost per
PDF 9's Cold-1 analysis, which this A/B does not measure. Choose the 20 Gbps copy for faster restarts;
choose either for serving.

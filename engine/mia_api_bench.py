#!/usr/bin/env python3
"""Exercise the real vLLM OpenAI/SSE path inside the existing benchmark container."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

import mia_bench
from bench_engine import host_memory


def build_api_server_command(
    *,
    model: str,
    draft: str,
    mtp: int,
    graphs: str,
    single_gb10: bool,
    kv_mib: int | None,
    draft_sample_method: str = "greedy",
    rejection_sample_method: str = "standard",
    block_size: int | None = None,
    worker_extension_cls: str | None = None,
    served_model_name: str = "qwen38-mia-a5b",
    host: str = "127.0.0.1",
    port: int = 18800,
    max_model_len: int = 1024,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 8192,
    override_generation_config: dict[str, object] | None = None,
    async_scheduling: bool | None = None,
    reasoning_parser: str | None = None,
    gpu_memory_utilization: float = 0.01,
) -> tuple[list[str], dict[str, object], dict[str, object]]:
    split_ops = ["vllm::unified_attention_with_output", "vllm::unified_mla_attention_with_output",
                 "vllm::mamba_mixer2", "vllm::mamba_mixer", "vllm::short_conv",
                 "vllm::qwen3_8_flash_next_ple_short_conv", "vllm::qwen3_8_flash_next_qsa_with_output",
                 "vllm::linear_attention", "vllm::qwen_gdn_attention_core",
                 "vllm::qwen_gdn_attention_core_fused_norm_packed", "vllm::sparse_attn_indexer",
                 "vllm::ple_mmap_lookup"]
    if not single_gb10:
        split_ops.extend(["vllm::moe_forward", "vllm::moe_forward_shared"])
    worker = worker_extension_cls or (
        "benchmark_worker.BenchmarkWorkerExtension" if single_gb10 else "peer_worker.PeerWorkerExtension"
    )
    speculative = {
        "method": "mtp",
        "num_speculative_tokens": mtp,
        "model": draft,
        "rejection_sample_method": rejection_sample_method,
        "draft_sample_method": draft_sample_method,
    }
    if os.environ.get("QWEN38_INDEX_SHARE", "0") == "1":
        speculative["index_share_for_mtp_iteration"] = True
    compile_config = {"cudagraph_mode": graphs, "splitting_ops": split_ops}
    if graphs != "NONE":
        # A target verification step has at most K+1 rows per scheduled
        # request. Larger prompts retain the normal eager/piecewise fallback.
        compile_config["max_cudagraph_capture_size"] = max_num_seqs * (mtp + 1)
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", served_model_name,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--no-enable-prefix-caching", "--enable-chunked-prefill", "--load-format", "fastsafetensors",
        "--no-enable-flashinfer-autotune", "--kv-cache-dtype", "auto", "--seed", "42",
        "--language-model-only", "--worker-extension-cls", worker,
        "--speculative-config", json.dumps(speculative),
        "--compilation-config", json.dumps(compile_config),
        "--generation-config", "vllm", "--no-enable-log-requests",
        "--enable-prompt-tokens-details",
    ]
    if kv_mib is not None:
        command.extend(["--kv-cache-memory-bytes", str(kv_mib * 1024**2)])
    mamba_dtype = os.environ.get("QWEN38_MAMBA_DTYPE")
    if mamba_dtype:
        command.extend(["--mamba-ssm-cache-dtype", mamba_dtype])
    profiler_config = os.environ.get("QWEN38_PROFILER_CONFIG")
    if profiler_config:
        parsed_profiler = json.loads(profiler_config)
        if not isinstance(parsed_profiler, dict):
            raise ValueError("QWEN38_PROFILER_CONFIG must be a JSON object")
        command.extend(["--profiler-config", json.dumps(parsed_profiler)])
    medium_template = Path(model) / "medium_chat_template.jinja"
    if medium_template.is_file():
        command.extend(["--chat-template", str(medium_template)])
    if override_generation_config is not None:
        command.extend(["--override-generation-config", json.dumps(override_generation_config)])
    if reasoning_parser is not None:
        command.extend(["--reasoning-parser", reasoning_parser])
    if async_scheduling is False:
        command.append("--no-async-scheduling")
    elif async_scheduling is True:
        command.append("--async-scheduling")
    if block_size is not None:
        command.extend(["--block-size", str(block_size)])
    if graphs == "NONE":
        command.append("--enforce-eager")
    return command, compile_config, speculative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--mtp", type=int, required=True)
    parser.add_argument("--graphs", choices=("PIECEWISE", "NONE"), default="PIECEWISE")
    parser.add_argument("--out", required=True)
    parser.add_argument("--outputs", type=int, default=400)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--warmup-measured-prompt", action="store_true")
    parser.add_argument("--single-gb10", action="store_true")
    parser.add_argument("--kv-mib", type=int, default=2048)
    parser.add_argument("--draft-sample-method", choices=("greedy", "probabilistic"), default="greedy")
    parser.add_argument("--rejection-sampling", choices=("standard", "block"), default="standard")
    parser.add_argument("--block-size", type=int, help="Optional legal hybrid KV alignment unit")
    parser.add_argument("--workloads", choices=("code", "prose", "code_ts", "json_schema", "prose_pure", "prose_pure_greedy"), nargs="+", default=["code", "prose"])
    args = parser.parse_args()
    out = Path(args.out)
    command, compile_config, speculative = build_api_server_command(
        model=args.model,
        draft=args.draft,
        mtp=args.mtp,
        graphs=args.graphs,
        single_gb10=args.single_gb10,
        kv_mib=args.kv_mib,
        block_size=args.block_size,
        max_model_len=((args.outputs + args.mtp + 767) // 256) * 256,
        draft_sample_method=args.draft_sample_method,
        rejection_sample_method=args.rejection_sampling,
    )
    process = None
    with out.with_suffix(".server.log").open("w") as log:
        try:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parent,
                                       stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 450
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Isolated vLLM API server exited before readiness")
                try:
                    with urlopen("http://127.0.0.1:18800/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except (URLError, HTTPError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Isolated vLLM API server readiness timed out")
                time.sleep(2)
            bench_args = ["--base-url", "http://127.0.0.1:18800", "--model", "qwen38-mia-a5b",
                          "--out", str(out), "--runs", str(args.runs), "--outputs", str(args.outputs),
                          "--warmup-tokens", str(args.warmup_tokens), "--warmup-rounds", str(args.warmup_rounds),
                          "--workloads", *args.workloads]
            if args.warmup_measured_prompt:
                bench_args.append("--warmup-measured-prompt")
            memory_before = host_memory()
            status = mia_bench.main(bench_args)
            memory_after = host_memory()
            result = json.loads(out.read_text())
            result["host_memory_before_measurement"] = memory_before
            result["host_memory_after_measurement"] = memory_after
            result["swap_out_pages_during_measurement"] = memory_after["pswpout"] - memory_before["pswpout"]
            result["swap_in_pages_during_measurement"] = memory_after["pswpin"] - memory_before["pswpin"]
            result["host_memory_window_scope"] = "System-wide counters; API readiness to end of warmup and measured requests, excluding model loading; not process-exclusive attribution"
            draft_config = json.loads((Path(args.draft) / "config.json").read_text())
            result["engine_runtime"] = {"model": args.model, "draft": args.draft,
                                        "draft_depth": args.mtp, "kv_dtype": "BF16", "kv_bytes": args.kv_mib * 1024**2,
                                        "precision": "target A5B INT4; MTP RTN INT4" if "int4" in args.draft else "target A5B INT4; MTP BF16",
                                        "prefix_cache": False, "context_limit_tokens": 1024,
                                        "requested_kv_block_size": args.block_size,
                                        "draft_experts_per_token_config": draft_config["text_config"]["num_experts_per_tok"],
                                        "draft_head_mode": os.environ.get("VLLM_MTP_DRAFT_HEAD", "shared"),
                                        "draft_vocab_file": os.environ.get("VLLM_MTP_DRAFT_VOCAB"),
                                        "draft_head_source": os.environ.get("VLLM_MTP_DRAFT_VOCAB_CKPT"),
                                        "draft_scale_config": float(os.environ.get("VLLM_MTP_DRAFT_SCALE", "1")),
                                        "hot_cache_profile": os.environ.get("QWEN38_HOT_CACHE_PROFILE"),
                                        "qsa_exact_topk": os.environ.get("VLLM_QSA_EXACT_TOPK") == "1",
                                        "qsa_stable_order": os.environ.get("QWEN38_QSA_STABLE_TOPK") == "1",
                                        "moe_stable_order": os.environ.get("QWEN38_MOE_STABLE_ALIGN") == "1",
                                        "ple_madv_random": os.environ.get("VLLM_PLE_MMAP_MADV_RANDOM") == "1",
                                        "ple_prewarm": os.environ.get("VLLM_PLE_MMAP_PREWARM") == "1",
                                        "ple_fast_rows": int(os.environ.get("VLLM_PLE_MMAP_FAST_ROWS", "512")),
                                        "helper_graph_mode": os.environ.get("QWEN38_HELPER_GRAPHS", "0"),
                                        "shared_overlap_requested": os.environ.get("QWEN38_SHARED_OVERLAP") == "1",
                                        "local_expert_layers": int(os.environ.get("QWEN38_LOCAL_EXPERT_LAYERS", "0")),
                                        "compilation_config": compile_config, "speculative_config": speculative,
                                        "transport": "real OpenAI HTTP/SSE over container loopback"}
            temporary = out.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(out)
            return status
        except Exception as error:
            result = {"status": "failed", "schema": "mia-api-startup-failure-v1", "error_type": type(error).__name__,
                      "protocol_id": mia_bench.PROTOCOL_ID,
                      "requested": {"workloads": list(args.workloads)},
                      "failures": [{"stage": "server_startup_or_measurement", "reason": type(error).__name__}]}
            out.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            return 1
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Persistent vLLM API benchmark with versioned Mia and cold-prefill protocols."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

import mia_bench
import resident_campaign
from bench_engine import host_memory
from mia_api_bench import build_api_server_command



_SPARKDASH_COMMIT = "693fa8016ea3a24b29fe36a98367669fd316393b"
_RECIPE_COMMIT = "fd3f43e11238553685ebb68b6d8ab2b112818e3c"
_SERVE_MODEL_NAME = "qwen38-mia-a5b-resident"
_WORKER_EXTENSION = "resident_bench.ResidentWorkerExtension"
_DEFAULT_STATE_FILE = os.environ.get("QWEN38_RESIDENT_STATE_FILE", "/work/resident-server-state.json")
_SUPPORTED_RPCS = {
    "resident_engine_identity",
    "resident_spec_configure",
    "resident_spec_status",
    "resident_ple_mmap_diagnostics",
    "resident_ple_mmap_prewarm",
    "resident_qad_diagnostics",
    "resident_nvfp4_variant_diagnostics",
    "peer_inventory",
    "peer_async_transfers",
    "peer_shared_overlap",
    "benchmark_memory",
    "peer_route_calibration",
}
_PEER_CONTROL_METHODS = ("peer_async_transfers", "peer_shared_overlap")
_SPEC_METRICS = {
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
}
_METRIC_LINE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
_CODE_STREAMS = (
    ("binary_search", "def binary_search(nums, target) -> int: index of target in a sorted list, or -1."),
    ("merge_sort", "def merge_sort(nums) -> list: stable sort of a list of ints, returning a new list."),
    ("lru_cache", "class LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used."),
    ("token_bucket", "class TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False."),
)
_STRUCTURED_PROMPT = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
_DEFAULT_TEMPERATURE = {"code": 0.0, "prose": 0.7, "structured": 0.7, "prose_pure": 0.7}
_DEFAULT_THINKING = {"code": False, "prose": True, "structured": True, "prose_pure": False}
_PREFILL_WORDS = (
    "the of and to in is you that it he was for on are as with his they at be this have from or one had by "
    "word but not what all were we when your can said there use an each which she do how their if will up "
    "other about out many then them these so some her would make like him into time"
).split()
_EXACT_INPUT_TARGET_SEMANTICS = (
    "exact final chat input-token count from resident API POST /tokenize for decode/prefill "
    "size-sweep prompts; includes the chat template and generation/thinking header; warmup and "
    "calibration prompts stay unchanged; each size-sweep usage.prompt_tokens must match or the "
    "request is throughput-ineligible"
)
_LEGACY_INPUT_TARGET_SEMANTICS = (
    "legacy nominal filler-plus-task text target; chat-template tokens are not included and "
    "actual usage.prompt_tokens remains authoritative"
)


def _worker_extension_base():
    if os.environ.get("QWEN38_RESIDENT_WORKER_PROCESS") != "1":
        return object
    probe_path = os.environ.get("QWEN38_CACHE_PROBE_PATH")
    if probe_path:
        from hybrid_cache_probe import install_cache_probe

        install_cache_probe(probe_path)
    if os.environ.get("QWEN38_SINGLE_GB10") == "1":
        from benchmark_worker import BenchmarkWorkerExtension

        return BenchmarkWorkerExtension
    from peer_worker import PeerWorkerExtension

    return PeerWorkerExtension


class ResidentWorkerExtension(_worker_extension_base()):
    def resident_engine_identity(self) -> dict[str, Any]:
        # API health has passed before this RPC: the shared base worker's
        # startup-only watchdog must not terminate either resident placement.
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        return {
            "engine_instance_id": os.environ.get("QWEN38_RESIDENT_ENGINE_ID"),
            "pid": os.getpid(),
            "worker_class": type(self).__name__,
        }

    def resident_spec_configure(self, settings: dict[str, Any]) -> dict[str, Any]:
        from resident_spec_control import configure_resident_spec

        return configure_resident_spec(self, settings)

    def resident_spec_status(self) -> dict[str, Any]:
        from resident_spec_control import resident_spec_status

        return resident_spec_status(self)

    def resident_ple_mmap_diagnostics(self) -> dict[str, Any]:
        from resident_ple_control import ple_mmap_diagnostics

        return ple_mmap_diagnostics()

    def resident_ple_mmap_prewarm(self) -> dict[str, Any]:
        from resident_ple_control import guarded_prewarm_registered_tables

        return guarded_prewarm_registered_tables()

    def resident_qad_diagnostics(self) -> dict[str, Any]:
        import importlib
        import resident_qad_diagnostics

        return importlib.reload(resident_qad_diagnostics).run(self)

    def resident_nvfp4_variant_diagnostics(
        self, candidate_library: str, *, warmups: int = 3, repeats: int = 21,
    ) -> dict[str, Any]:
        from resident_qad_diagnostics import run_native_variant

        return run_native_variant(self, candidate_library, warmups=warmups, repeats=repeats)



class ResidentApi:
    """Narrow loopback client for OpenAI SSE, worker RPC, and vLLM counters."""

    def __init__(self, base_url: str, api_key: str | None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _open(self, path: str, *, method: str = "GET", data: bytes | None = None, timeout: float = 5):
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        return urlopen(request, timeout=timeout)

    def health(self) -> bool:
        with self._open("/health", timeout=3) as response:
            return response.status == 200

    def tokenize_chat_prompt(
        self,
        model_id: str,
        prompt: str,
        *,
        thinking: bool,
        timeout: float = 360,
    ) -> int:
        generation_body = mia_bench._request_body(
            model_id, prompt, 1, 0.0, thinking=thinking,
        )
        body = {
            key: generation_body[key]
            for key in ("model", "messages", "chat_template_kwargs")
        }
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            with self._open("/tokenize", method="POST", data=data, timeout=timeout) as response:
                raw_payload = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if self.api_key:
                detail = detail.replace(self.api_key, "[REDACTED]")
            raise RuntimeError(f"POST /tokenize returned HTTP {exc.code}: {detail}") from exc
        try:
            payload = json.loads(raw_payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"POST /tokenize returned invalid JSON ({type(exc).__name__}: {exc})"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("POST /tokenize response must be a JSON object")
        count = payload.get("count")
        tokens = payload.get("tokens")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            if isinstance(tokens, list) and len(tokens) != count:
                raise RuntimeError(
                    f"POST /tokenize count={count} disagrees with returned token array length={len(tokens)}"
                )
            return count
        if isinstance(tokens, list):
            return len(tokens)
        raise RuntimeError(
            "POST /tokenize response has neither a non-negative integer count nor a tokens array"
        )

    def worker_rpc(
        self,
        method: str,
        *,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> Any:
        if method not in _SUPPORTED_RPCS:
            raise ValueError(f"worker RPC is not allowlisted: {method}")
        body = json.dumps(
            {"method": method, "args": args or [], "kwargs": kwargs or {}, "timeout": timeout},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            with self._open("/collective_rpc", method="POST", data=body, timeout=timeout + 5) as response:
                payload = response.read()
        except HTTPError as exc:
            detail = exc.read(512).decode("utf-8", errors="replace")
            raise RuntimeError(f"worker RPC {method} returned HTTP {exc.code}: {detail}") from exc
        if not payload:
            return None
        result = json.loads(payload)
        return result.get("results") if isinstance(result, dict) and "results" in result else result

    def speculative_metrics(self) -> dict[str, Any]:
        try:
            with self._open("/metrics", timeout=5) as response:
                text = response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            return {
                "source": "vLLM /metrics Prometheus endpoint",
                "counters": {},
                "metric_series": {},
                "missing": sorted(_SPEC_METRICS),
                "reason": f"metrics_endpoint_unavailable: {type(exc).__name__}",
            }
        counters: dict[str, float] = {}
        metric_series: dict[str, dict[str, float]] = {}
        raw_names: list[str] = []
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            match = _METRIC_LINE.match(line)
            if not match:
                continue
            name, labels, raw_value = match.groups()
            if name not in _SPEC_METRICS:
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            counters[name] = counters.get(name, 0.0) + value
            label_key = labels or ""
            series = metric_series.setdefault(name, {})
            series[label_key] = series.get(label_key, 0.0) + value
            raw_names.append(name)
        missing = sorted(_SPEC_METRICS.difference(counters))
        return {
            "source": "vLLM /metrics Prometheus endpoint; no estimated values",
            "counters": counters,
            "metric_series": metric_series,
            "raw_metric_names": sorted(set(raw_names)),
            "missing": missing,
            "reason": "speculative_metrics_not_exported_by_server" if missing else None,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _extract_worker_pids(value: Any) -> list[int]:
    found: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in ("worker_pid", "pid", "engine_pid"):
                candidate = item.get(key)
                if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                    found.add(candidate)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(found)


def _explicit_control_rejection(value: Any) -> list[str]:
    reasons: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            if item.get("accepted") is False or item.get("success") is False or item.get("applied") is False:
                reasons.append(f"{path}: {item.get('reason') or item.get('status') or 'explicit rejection'}")
            rejected = item.get("rejected")
            if rejected:
                reasons.append(f"{path}: rejected={rejected!r}")
            for key, nested in item.items():
                visit(nested, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")

    visit(value, "control")
    return reasons


def _code_prompt(stream_index: int) -> str:
    if stream_index == 0:
        return mia_bench.PROMPTS["code"]
    name, spec = _CODE_STREAMS[stream_index]
    return f"{name}\n{spec}\n{mia_bench._CODE_TAIL}"


def _decode_prompts(workload: str, concurrency: int) -> list[str]:
    if workload == "code":
        return [_code_prompt(index) for index in range(concurrency)]
    if workload == "structured":
        prompt = _STRUCTURED_PROMPT
    elif workload == "prose_pure":
        prompt = mia_bench.PROMPTS["prose_pure"]
    else:
        prompt = mia_bench.PROMPTS["prose"]
    if concurrency == 1:
        return [prompt]
    return [f"Request ID: decode-{index + 1}\n{prompt}" for index in range(concurrency)]


def _prefill_filler(count: int, salt: str = "") -> str:
    if salt:
        seed_bytes = hashlib.sha256(salt.encode("utf-8")).digest()
        state = int.from_bytes(seed_bytes[:4], "big")
    else:
        state = 0x2545F491
    words: list[str] = []
    for _ in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        words.append(_PREFILL_WORDS[(state >> 8) % len(_PREFILL_WORDS)])
    return " " + " ".join(words)


def _token_estimate(text: str) -> int:
    return max(1, math.floor(len(text) / 4 + 0.5))


def _build_prefill_prompt(target_tokens: int, salt: str) -> str:
    header = f"[prefill-bench {salt}]\nIgnore the filler below. Reply with the single word OK.\n"
    footer = "\nReply OK."
    reserved = _token_estimate(header + footer)
    fill_tokens = max(1, target_tokens - reserved)
    return header + _prefill_filler(fill_tokens, salt) + footer


def _build_workload_context_prompt(prompt: str, target_tokens: int, salt: str) -> str:
    if target_tokens <= 0:
        return prompt
    header = (
        f"[decode-context {salt}]\n"
        "Background filler follows. It carries no instructions and can be ignored.\n"
    )
    task_tail = f"\nEnd of background filler.\n\n{prompt}"
    reserved = _token_estimate(header + task_tail)
    fill_tokens = max(1, target_tokens - reserved)
    return header + _prefill_filler(fill_tokens, salt) + task_tail


def _build_exact_context_prompt(
    api: ResidentApi,
    model_id: str,
    target_tokens: int,
    *,
    header: str,
    task_tail: str,
    thinking: bool,
    request_timeout: float,
) -> tuple[str, dict[str, Any]]:
    if target_tokens <= 0:
        raise ValueError("exact input-token targets must be positive")
    counts_by_filler_words: dict[int, int] = {}

    def measure(filler_words: int) -> tuple[str, int]:
        prompt = header + (_prefill_filler(filler_words, header) if filler_words else "") + task_tail
        if filler_words not in counts_by_filler_words:
            counts_by_filler_words[filler_words] = api.tokenize_chat_prompt(
                model_id, prompt, thinking=thinking, timeout=request_timeout,
            )
        return prompt, counts_by_filler_words[filler_words]

    prompt, minimum_tokens = measure(0)
    if minimum_tokens > target_tokens:
        raise ValueError(
            f"exact input-token target {target_tokens} is below the {minimum_tokens}-token "
            "minimum required by the unchanged task, nonce, chat template, and thinking header"
        )
    filler_words = 0
    if minimum_tokens < target_tokens:
        token_gap = target_tokens - minimum_tokens
        sample_words = min(256, token_gap)
        _, sample_tokens = measure(sample_words)
        sample_delta = sample_tokens - minimum_tokens
        initial_words = (
            max(1, (token_gap * sample_words + sample_delta // 2) // sample_delta)
            if sample_delta > 0 else token_gap
        )
        initial_words = min(initial_words, target_tokens * 4)
        low_words, low_tokens = 0, minimum_tokens
        high_words: int | None = None
        high_tokens: int | None = None
        filler_words = initial_words
        for _ in range(16):
            prompt, token_count = measure(filler_words)
            if token_count == target_tokens:
                break
            if token_count < target_tokens:
                low_words, low_tokens = filler_words, token_count
            else:
                high_words, high_tokens = filler_words, token_count
            if high_words is not None:
                if high_words - low_words <= 1:
                    break
                assert high_tokens is not None
                if high_tokens > low_tokens:
                    guess = low_words + (
                        (target_tokens - low_tokens) * (high_words - low_words)
                        // (high_tokens - low_tokens)
                    )
                else:
                    guess = (low_words + high_words) // 2
                if not low_words < guess < high_words:
                    guess = (low_words + high_words) // 2
                filler_words = guess
            else:
                filler_words += max(1, target_tokens - token_count)
        prompt, token_count = measure(filler_words)
        if token_count != target_tokens:
            if high_words is None:
                candidate = max(1, low_words * 2)
                for _ in range(24):
                    candidate_prompt, candidate_tokens = measure(candidate)
                    if candidate_tokens == target_tokens:
                        filler_words, prompt, token_count = candidate, candidate_prompt, candidate_tokens
                        break
                    if candidate_tokens < target_tokens:
                        low_words, low_tokens = candidate, candidate_tokens
                        candidate = min(candidate * 2, target_tokens * 4)
                        if candidate <= low_words:
                            break
                    else:
                        high_words, high_tokens = candidate, candidate_tokens
                        break
            if token_count != target_tokens and high_words is not None:
                while high_words - low_words > 1:
                    filler_words = (low_words + high_words) // 2
                    prompt, token_count = measure(filler_words)
                    if token_count == target_tokens:
                        break
                    if token_count < target_tokens:
                        low_words, low_tokens = filler_words, token_count
                    else:
                        high_words, high_tokens = filler_words, token_count
            if token_count != target_tokens:
                raise RuntimeError(
                    f"exact input-token target {target_tokens} is not reachable with the "
                    f"unchanged task and filler-only adjustment; nearest checked counts were "
                    f"{low_tokens} and {high_tokens}"
                )
    final_tokens = counts_by_filler_words[filler_words]
    return prompt, {
        "method": "resident API POST /tokenize with the exact chat messages and chat_template_kwargs",
        "target_input_tokens": target_tokens,
        "minimum_prompt_tokens_without_filler": minimum_tokens,
        "final_tokenizer_prompt_tokens": final_tokens,
        "filler_word_count": filler_words,
        "tokenization_call_count": len(counts_by_filler_words),
        "tokenization_counts_by_filler_word_count": [
            {"filler_word_count": words, "prompt_tokens": count}
            for words, count in sorted(counts_by_filler_words.items())
        ],
        "verified_exact_before_generation": final_tokens == target_tokens,
    }


def _build_exact_workload_context_prompt(
    api: ResidentApi,
    model_id: str,
    prompt: str,
    target_tokens: int,
    salt: str,
    *,
    thinking: bool,
    request_timeout: float,
) -> tuple[str, dict[str, Any]]:
    header = (
        f"[decode-context {salt}]\n"
        "Background filler follows. It carries no instructions and can be ignored.\n"
    )
    task_tail = f"\nEnd of background filler.\n\n{prompt}"
    return _build_exact_context_prompt(
        api, model_id, target_tokens,
        header=header, task_tail=task_tail, thinking=thinking,
        request_timeout=request_timeout,
    )


def _build_exact_prefill_context_prompt(
    api: ResidentApi,
    model_id: str,
    prompt: str,
    target_tokens: int,
    salt: str,
    *,
    request_timeout: float,
) -> tuple[str, dict[str, Any]]:
    header = (
        f"[decode-context {salt}]\n"
        "Background filler follows. It carries no instructions and can be ignored.\n"
    )
    task_tail = f"\nEnd of background filler.\n\n{prompt}"
    return _build_exact_context_prompt(
        api, model_id, target_tokens,
        header=header, task_tail=task_tail, thinking=False,
        request_timeout=request_timeout,
    )


def _code_task_name(prompt: str) -> str | None:
    candidates = ["binary_search", *(name for name, _ in _CODE_STREAMS)]
    for name in candidates:
        if prompt.startswith(f"{name}\n") or f"\n{name}\n" in prompt:
            return name
    return None


def _parse_experiments(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return [{"name": "baseline", "settings": {}}]
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "experiments" in raw:
        unknown = set(raw) - {"experiments"}
        if unknown:
            raise ValueError(f"experiment file has unrecognized top-level fields: {sorted(unknown)!r}")
        raw = raw["experiments"]
    elif isinstance(raw, dict):
        raw = [{"name": name, "settings": settings} for name, settings in raw.items()]
    if not isinstance(raw, list) or not raw:
        raise ValueError("experiment file must contain a non-empty list or name-to-settings map")
    experiments: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"experiment {index} must be an object")
        unknown = set(item) - {"name", "settings", "peer_controls"}
        if unknown:
            raise ValueError(
                f"experiment {index} has unrecognized fields {sorted(unknown)!r}; "
                "runtime controls must be nested under 'settings' or 'peer_controls'"
            )
        raw_name = item.get("name")
        if raw_name is not None and not isinstance(raw_name, str):
            raise ValueError(f"experiment {index} name must be a string")
        name = (raw_name or f"experiment-{index}").strip()
        settings = item.get("settings", {})
        peer_controls = item.get("peer_controls")
        if not name or name in names:
            raise ValueError(f"experiment name is empty or duplicated: {name!r}")
        if not isinstance(settings, dict):
            raise ValueError(f"experiment {name!r} settings must be an object")
        normalized = {"name": name, "settings": settings}
        if "peer_controls" in item:
            if not isinstance(peer_controls, dict):
                raise ValueError(f"experiment {name!r} peer_controls must be an object")
            unknown_peer_controls = set(peer_controls) - set(_PEER_CONTROL_METHODS)
            if unknown_peer_controls:
                raise ValueError(
                    f"experiment {name!r} has unsupported peer controls: "
                    f"{sorted(unknown_peer_controls)!r}"
                )
            if len(peer_controls) > 1 or any(type(value) is not bool for value in peer_controls.values()):
                raise ValueError(
                    f"experiment {name!r} must select at most one peer control with an exact boolean value"
                )
            normalized["peer_controls"] = peer_controls
        names.add(name)
        experiments.append(normalized)
    peer_control_mode = any("peer_controls" in item for item in experiments)
    if peer_control_mode:
        if any("peer_controls" not in item for item in experiments):
            raise ValueError("peer-control experiments must specify peer_controls for every A/B position")
        if any(item["settings"] for item in experiments):
            raise ValueError("peer controls cannot be mixed with speculative-decoder settings")
        if not any(item["peer_controls"] for item in experiments):
            raise ValueError("peer-control experiment set has no candidate setting")
    return experiments


def _parse_workload_overrides(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        workload, separator, raw_cap = value.partition("=")
        if not separator or workload not in ("code", "prose", "structured", "prose_pure"):
            raise ValueError(f"invalid --output-cap {value!r}; use workload=positive_integer")
        cap = int(raw_cap)
        if cap < 1 or workload in out:
            raise ValueError(f"invalid or duplicate output cap for {workload!r}")
        out[workload] = cap
    return out


def _profile(args: argparse.Namespace) -> dict[str, Any]:
    selected = mia_bench.RESIDENT_PROTOCOLS[args.protocol_profile]
    caps = dict(selected["max_tokens_by_workload"])
    caps.update(_parse_workload_overrides(args.output_cap))
    force_to_cap = bool(selected["force_to_cap"])
    ignore_eos_requested = True if force_to_cap else selected.get("explicit_natural_ignore_eos")
    base_profile_id = selected["protocol_id"]
    protocol_id = base_profile_id
    if args.output_cap or args.top_k is not None or args.top_p != 1.0 or args.warmup_tokens != 32:
        protocol_id = f"{protocol_id}-custom-v1"
    if any(size > 0 for size in args.decode_context_sizes):
        protocol_id = f"{protocol_id}-context-sweep-v1"
    if args.exact_input_tokens:
        protocol_id = f"{protocol_id}-exact-input-tokens-v1"
    metadata = dict(mia_bench.RESIDENT_PROTOCOL_METADATA)
    metadata.update({
        "protocol_id": protocol_id,
        "base_profile_id": base_profile_id,
        "decode_context_tokens": list(args.decode_context_sizes),
        "positive_decode_context_is_not_headline": any(size > 0 for size in args.decode_context_sizes),
        "selected_profile": args.protocol_profile,
        "protocol_class": selected.get("protocol_class", "diagnostic-or-unclassified"),
        "exact_input_tokens": args.exact_input_tokens,
        "input_target_semantics": (
            _EXACT_INPUT_TARGET_SEMANTICS if args.exact_input_tokens
            else _LEGACY_INPUT_TARGET_SEMANTICS
        ),
        "source_commit": _SPARKDASH_COMMIT,
        "recipe_commit": _RECIPE_COMMIT,
        "max_tokens_by_workload": caps,
        "warmup_max_tokens": args.warmup_tokens,
        "temperature_by_workload": dict(_DEFAULT_TEMPERATURE),
        "thinking_by_workload": dict(_DEFAULT_THINKING),
        "top_p": args.top_p,
        "top_k_in_request": args.top_k if args.top_k is not None else "omitted",
        "pinned_sparkdash_request_top_k": "omitted",
        "pinned_tensorfold_server_default_top_k": 20,
        "server_generation_config_top_k_override": 20,
        "effective_top_k": args.top_k if args.top_k is not None else 20,
        "installed_vllm_default_top_k": "not relied on; server configuration explicitly overrides it",
        "force_cap_fields": force_to_cap,
        "forced_min_tokens": force_to_cap,
        "ignore_eos": ignore_eos_requested if ignore_eos_requested is not None else "omitted",
        "ignore_eos_requested": ignore_eos_requested if ignore_eos_requested is not None else "omitted",
        "min_tokens": "max_tokens" if force_to_cap else "omitted",
        "stop": [] if force_to_cap else "omitted",
        "natural_eos": "only after HTTP 400 force-field fallback" if force_to_cap else True,
        "natural_eos_request": (
            "explicit ignore_eos=false; min_tokens and stop omitted"
            if ignore_eos_requested is False else
            "natural EOS; ignore_eos/min_tokens/stop omitted"
            if not force_to_cap else "natural EOS only after HTTP 400 force-field fallback"
        ),
        "thinking_http_400_fallback": {
            "thinking_on": "strip thinking flags once and retry; backend effective mode then unknown",
            "thinking_off": "retry with explicit disable flags, then minimal enable_thinking=false body",
            "source": "pinned LlmStreaming.runStreamingRequest behavior",
        },
        "forced_cap_http_400_retry_max": 1 if force_to_cap else 0,
        "max_attempts_per_request": 6 if force_to_cap else 3,
        "raw_sse_encoding": "base64 (exact response bytes)",
        "timing_source": "time.perf_counter monotonic marks around first/last non-empty visible reasoning or answer fragments",
        "decode_tps_formula": "(usage.completion_tokens - 1) / (last visible - first visible)",
        "aggregate_decode_tps_formula": "sum of post-first completion tokens / (latest last-visible - earliest first-visible); pinned DecodeBench implementation, not the README's per-request-rate-sum wording",
        "historical_published_execution_reproduced": False,
        "source_version_scope": "Pinned sparkDash client request contract; the exact client commit and request transcript of the published run are unavailable",
        "mtp_draft_sample_method_at_server_start": args.draft_sample_method,
        "mtp_rejection_sample_method_at_server_start": args.rejection_sampling,
        "resident_spec_control": {
            "configure_rpc": "resident_spec_configure(settings) delegates atomically to configure_resident_spec(self, settings)",
            "status_rpc": "resident_spec_status() delegates to resident_spec_status(self)",
            "supported_live_settings": {
                "draft_scale": "finite >0; probabilistic/standard and PIECEWISE or NONE only",
                "prose_max_draft_tokens": "integer 1..Kmax; all-positive-temperature batch uses configured prose K, mixed/empty batch uses Kmax; T=0 follows the supported code-depth policy",
                "prose_confidence_threshold": "null/0 disables or finite [0,1]; probabilistic and all-positive-temperature batch only",
                "prose_draft_top_k": "null disables or exact integer 1..draft-vocab width; proposal logits and cached q are filtered together only for all-positive-temperature batches",
                "ple_madvise": "normal/random on already loaded original mmap planes between requests; no pin or numerical changes",
                "qad_hot_cache_enabled": "exact boolean; toggles only a ready, loaded 48-layer native QAD HotCache without reloading the engine",
                "compact_draft_head_enabled": (
                    "exact boolean; defaults false; first enable verifies the loaded tokenizer SHA256 and unique in-range IDs, "
                    "then prepares selected rows from the actual dense BF16 MTP lm_head without reload or INT4 requantization; "
                    "standard rejection; applies to all workloads"
                ),
            },
            "async_scheduling_required": False,
            "unsupported_or_invalid_settings": "whole configuration rejected atomically; no partial apply",
            "status_fields": [
                "max K", "last active/valid K", "sync-check count", "graph/async gating", "metrics source",
                "qad_hot_cache",
                "compact_draft_head",
            ],
        },
    })
    if args.protocol_profile == "official-headline":
        metadata["official_recipe_headline"] = {
            "published_code_decode_tps": 110.5,
            "published_prose_decode_tps": 55.6,
            "published_sampling": {
                "code": {"temperature": 0.0, "thinking": False},
                "prose": {"temperature": 0.7, "thinking": True},
            },
            "effective_top_k": args.top_k if args.top_k is not None else 20,
            "effective_top_k_source": "explicit request top_k" if args.top_k is not None
            else "resident API server --override-generation-config {\"top_k\":20}; request top_k omitted",
            "pinned_client_request_shape_matches": (
                args.top_k is None and args.top_p == 1.0 and args.warmup_tokens == 32
                and not args.output_cap and args.decode_context_sizes == [0]
                and not args.exact_input_tokens
            ),
            "same_checkpoint_engine_hardware_context_claimed": False,
            "published_workloads": ["code", "prose"],
            "structured_scope": "Pinned DecodeBench Count1to200 task; not a published code/prose headline row",
            "repetition_scope": "Local repeated measurements; not proof of the historical published run's repetition settings",
        }
    else:
        metadata["official_recipe_headline"] = {
            "published_code_decode_tps": 110.5,
            "published_prose_decode_tps": 55.6,
            "direct_headline_comparison": False,
            "reason": "requested caps and natural-EOS policy differ from the pinned 400-token forced-cap request",
        }
    return {
        "protocol_id": protocol_id,
        "caps": caps,
        "force_to_cap": force_to_cap,
        "ignore_eos_requested": ignore_eos_requested,
        "metadata": metadata,
    }


def _metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_values = before.get("counters", {}) if isinstance(before, dict) else {}
    after_values = after.get("counters", {}) if isinstance(after, dict) else {}
    deltas: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    for name in sorted(_SPEC_METRICS):
        left = before_values.get(name)
        right = after_values.get(name)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            deltas[name] = None
            reasons[name] = "counter_missing_before_or_after; no estimate substituted"
        elif right < left:
            deltas[name] = None
            reasons[name] = "counter_reset_during_window"
        else:
            deltas[name] = right - left
    drafts = deltas.get("vllm:spec_decode_num_drafts_total")
    proposed = deltas.get("vllm:spec_decode_num_draft_tokens_total")
    accepted = deltas.get("vllm:spec_decode_num_accepted_tokens_total")
    before_series = before.get("metric_series", {}) if isinstance(before, dict) else {}
    after_series = after.get("metric_series", {}) if isinstance(after, dict) else {}
    position_before = before_series.get("vllm:spec_decode_num_accepted_tokens_per_pos_total", {})
    position_after = after_series.get("vllm:spec_decode_num_accepted_tokens_per_pos_total", {})
    position_deltas: dict[str, float | None] = {}
    position_reasons: dict[str, str] = {}
    for label_key in sorted(set(position_before) | set(position_after)):
        left = position_before.get(label_key)
        right = position_after.get(label_key)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            position_deltas[label_key] = None
            position_reasons[label_key] = "position counter missing before or after; no estimate substituted"
        elif right < left:
            position_deltas[label_key] = None
            position_reasons[label_key] = "position counter reset during window"
        else:
            position_deltas[label_key] = right - left
    derived: dict[str, Any] = {
        "accepted_draft_tokens_per_round": accepted / drafts if accepted is not None and drafts else None,
        "accepted_fraction_of_proposed_tokens": accepted / proposed if accepted is not None and proposed else None,
        "pre_stop_acceptance_length_including_target_sample": (accepted + drafts) / drafts
        if accepted is not None and drafts else None,
        "position_acceptance_p1_p5": None,
    }
    return {
        "counter_delta": deltas,
        "counter_scope": "scheduler_before_request_stop_truncation",
        "derived_from_actual_counters_only": derived,
        "position_metric_name": "vllm:spec_decode_num_accepted_tokens_per_pos_total",
        "position_acceptance_counter_delta_by_exact_prometheus_labels": position_deltas,
        "position_counter_missing_reasons": position_reasons,
        "position_acceptance_reason": "actual per-position counters are preserved by label; p1-p5 rates are not inferred without verified position/denominator semantics",
        "missing_reasons": reasons,
    }


def _prefill_rate(event: dict[str, Any], overhead_ms: float | None) -> dict[str, Any]:
    prompt_tokens = event.get("prompt_tokens")
    cached_tokens = event.get("cached_prompt_tokens") or 0
    server_ms = event.get("server_prompt_ms")
    server_tokens = event.get("server_prompt_tokens")
    if isinstance(server_ms, (int, float)) and server_ms > 0 and isinstance(server_tokens, int) and server_tokens > 0:
        return {"tps": round(server_tokens / server_ms * 1000, 6), "method": "server", "low_confidence": False}
    if not isinstance(prompt_tokens, int) or not isinstance(event.get("ttft_ms"), (int, float)) or event["ttft_ms"] <= 0:
        return {"tps": None, "method": "ttft", "low_confidence": False,
                "reason": "actual_usage_prompt_tokens_and_positive_ttft_required"}
    computed = max(0, prompt_tokens - int(cached_tokens))
    ttft = float(event["ttft_ms"])
    if computed <= 0:
        return {"tps": None, "method": "ttft", "low_confidence": False,
                "reason": "prompt_fully_cached; no cold-prefill work measured"}
    if overhead_ms is None:
        return {"tps": None, "method": "ttft", "low_confidence": False,
                "reason": "no successful shared fixed-overhead 16-token calibration"}
    effective_ms = max(ttft - max(0.0, overhead_ms), ttft * 0.5)
    return {"tps": round(computed / (effective_ms / 1000), 6), "method": "ttft",
            "low_confidence": overhead_ms > ttft * 0.3, "computed_prompt_tokens": computed,
            "effective_ttft_ms": round(effective_ms, 6)}


def _prefill_repeats(size: int) -> int:
    return 3 if size <= 32768 else (2 if size <= 131072 else 1)


def _aggregate_wave(
    concurrency: int,
    events: list[dict[str, Any]],
    duration_ms: float,
    *,
    workload: str,
    phase: str,
) -> dict[str, Any]:
    throughput_success = len(events) == concurrency and all(event.get("throughput_eligible") for event in events)
    if phase == "prefill-measurement":
        throughput_success = throughput_success and all(
            isinstance(event.get("prefill_measurement", {}).get("tps"), (int, float))
            for event in events
        )
    quality_applicable_count = sum(event.get("quality_applicable", True) is True for event in events)
    quality_unassessed_count = sum(
        event.get("quality_applicable", True) is True and event.get("quality_valid") is None
        for event in events
    )
    quality_success: bool | None = (
        None
        if quality_applicable_count == 0
        or quality_applicable_count != concurrency
        or len(events) != concurrency
        or quality_unassessed_count > 0
        else all(event.get("quality_valid") is True for event in events)
    )
    per_request = [event.get("decode_tps") for event in events]
    valid_request_rates = (
        len(per_request) == concurrency and throughput_success
        and all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                for value in per_request)
    )
    summed_request_rates = sum(per_request) if valid_request_rates else None
    request_ttfts = [event.get("ttft_ms") for event in events]
    valid_ttfts = len(request_ttfts) == concurrency and all(
        isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        for value in request_ttfts
    )
    if throughput_success:
        firsts = [event["first_visible_perf_counter"] for event in events]
        lasts = [event["last_visible_perf_counter"] for event in events]
        window_ms = (max(lasts) - min(firsts)) * 1000
        total_decode_tokens = sum(int(event.get("decode_tokens") or 0) for event in events)
        aggregate_decode = total_decode_tokens / (window_ms / 1000) if window_ms > 0 else None
        request_starts = [event["request_start_perf_counter"] for event in events]
        prefill_window_ms = (max(firsts) - min(request_starts)) * 1000
        prompt_tokens = sum(int(event.get("prompt_tokens") or 0) for event in events)
        aggregate_prefill = prompt_tokens / (prefill_window_ms / 1000) if prefill_window_ms > 0 else None
    else:
        window_ms = None
        total_decode_tokens = None
        prefill_window_ms = None
        prompt_tokens = None
        aggregate_decode = None
        aggregate_prefill = None
    return {
        "concurrency": concurrency,
        "streams_completed": len(events),
        "streams_throughput_eligible": sum(bool(event.get("throughput_eligible")) for event in events),
        "streams_quality_valid": sum(event.get("quality_valid") is True for event in events),
        "streams_quality_invalid": sum(
            event.get("quality_applicable", True) is True and event.get("quality_valid") is False
            for event in events
        ),
        "streams_quality_not_applicable": sum(event.get("quality_applicable") is False for event in events),
        "all_streams_required_for_aggregate": True,
        "per_request_decode_tps": per_request,
        "per_request_ttft_ms": [event.get("ttft_ms") for event in events],
        "sum_per_request_decode_tps": round(summed_request_rates, 6) if summed_request_rates is not None else None,
        "mean_per_request_decode_tps": statistics.fmean(per_request) if valid_request_rates else None,
        "mean_per_request_ttft_ms": statistics.fmean(request_ttfts) if throughput_success and valid_ttfts else None,
        "aggregate_metric_scope": "aggregate_decode_tps is the pinned client common window; sum_per_request_decode_tps is the distinct README-style sum, not proof of overlap",
        "median_per_request_decode_tps": statistics.median(
            [value for value in per_request if isinstance(value, (int, float))]
        ) if any(isinstance(value, (int, float)) for value in per_request) else None,
        "aggregate_decode_tps": round(aggregate_decode, 6) if aggregate_decode is not None else None,
        "aggregate_decode_window_ms": round(window_ms, 6) if window_ms is not None else None,
        "aggregate_decode_tokens": total_decode_tokens,
        "per_request_prefill_tps_uncorrected": [event.get("prefill_tps") for event in events],
        "aggregate_prefill_tps_uncorrected": round(aggregate_prefill, 6) if aggregate_prefill is not None else None,
        "aggregate_prefill_window_ms": round(prefill_window_ms, 6) if prefill_window_ms is not None else None,
        "aggregate_prefill_prompt_tokens": prompt_tokens,
        "wave_elapsed_ms": round(duration_ms, 6),
        "throughput_eligible": throughput_success,
        "quality_valid_all_streams": quality_success,
        "quality_applicable_all_streams": quality_applicable_count == concurrency,
        "quality_not_applicable_all_streams": len(events) == concurrency and quality_applicable_count == 0,
        "failed_streams_excluded_from_aggregate": not throughput_success,
        "quality_and_throughput_are_independent": True,
    }


def _summaries(record: dict[str, Any]) -> dict[str, Any]:
    decode_groups: dict[tuple[str, int, str, int, str], list[dict[str, Any]]] = {}
    for wave in record.get("waves", []):
        if wave.get("phase") != "measurement":
            continue
        key = (
            wave["experiment"],
            int(wave.get("requested_context_tokens") or 0),
            wave["workload"],
            wave["concurrency"],
            wave.get("request_policy", {}).get("comparison_key", "unknown"),
        )
        decode_groups.setdefault(key, []).append(wave)

    decode_summary: list[dict[str, Any]] = []
    quality_summary: list[dict[str, Any]] = []
    for key in sorted(decode_groups):
        experiment, context_tokens, workload, concurrency, policy_key = key
        waves = decode_groups[key]
        rates = [
            wave["metrics"]["aggregate_decode_tps"] for wave in waves
            if wave.get("metrics", {}).get("throughput_eligible")
            and isinstance(wave.get("metrics", {}).get("aggregate_decode_tps"), (int, float))
        ]
        summed_rates = [
            wave["metrics"]["sum_per_request_decode_tps"] for wave in waves
            if wave.get("metrics", {}).get("throughput_eligible")
            and isinstance(wave.get("metrics", {}).get("sum_per_request_decode_tps"), (int, float))
        ]
        events = [event for wave in waves for event in wave.get("events", [])]
        applicable_events = [event for event in events if event.get("quality_applicable", True) is True]
        valid_events = [event for event in applicable_events if event.get("quality_valid") is True]
        invalid_events = [event for event in applicable_events if event.get("quality_valid") is False]
        decode_summary.append({
            "experiment": experiment,
            "requested_context_tokens": context_tokens,
            "workload": workload,
            "concurrency": concurrency,
            "request_policy_comparison_key": policy_key,
            "throughput_eligible_waves": len(rates),
            "throughput_ineligible_waves": len(waves) - len(rates),
            "quality_valid_waves": sum(wave["metrics"].get("quality_valid_all_streams") is True for wave in waves),
            "quality_invalid_waves": sum(
                wave["metrics"].get("quality_applicable_all_streams") is True
                and wave["metrics"].get("quality_valid_all_streams") is False
                for wave in waves
            ),
            "quality_unassessed_waves": sum(
                wave["metrics"].get("quality_valid_all_streams") is None
                and any(
                    event.get("quality_applicable", True) is True
                    for event in wave.get("events", [])
                )
                for wave in waves
            ),
            "quality_not_applicable_waves": sum(
                wave["metrics"].get("quality_not_applicable_all_streams") is True for wave in waves
            ),
            "actual_usage_prompt_tokens_by_request": [
                event.get("actual_usage_prompt_tokens") for event in events
            ],
            "per_request_decode_tps_by_wave": [
                {"run_index": wave.get("run_index"), "values": [
                    event.get("decode_tps") for event in wave.get("events", [])
                ]}
                for wave in waves
            ],
            "per_request_ttft_ms_by_wave": [
                {"run_index": wave.get("run_index"), "values": [
                    event.get("ttft_ms") for event in wave.get("events", [])
                ]}
                for wave in waves
            ],
            "per_request_decode_window_ms_by_wave": [
                {"run_index": wave.get("run_index"), "values": [
                    event.get("decode_window_ms") for event in wave.get("events", [])
                ]}
                for wave in waves
            ],
            "per_request_e2e_ms_by_wave": [
                {"run_index": wave.get("run_index"), "values": [
                    event.get("e2e_ms") for event in wave.get("events", [])
                ]}
                for wave in waves
            ],
            "aggregate_decode_tps_by_wave": [
                {
                    "run_index": wave.get("run_index"),
                    "value": wave.get("metrics", {}).get("aggregate_decode_tps"),
                    "eligible": wave.get("metrics", {}).get("throughput_eligible"),
                }
                for wave in waves
            ],
            "median_aggregate_decode_tps": statistics.median(rates) if rates else None,
            "mean_aggregate_decode_tps": statistics.fmean(rates) if rates else None,
            "sample_stddev_aggregate_decode_tps": statistics.stdev(rates) if len(rates) > 1 else None,
            "median_sum_per_request_decode_tps": statistics.median(summed_rates) if summed_rates else None,
            "quality_does_not_gate_timing_summary": True,
        })
        quality_summary.append({
            "experiment": experiment,
            "requested_context_tokens": context_tokens,
            "workload": workload,
            "concurrency": concurrency,
            "request_policy_comparison_key": policy_key,
            "requests": len(events),
            "quality_applicable_requests": len(applicable_events),
            "quality_not_applicable_requests": len(events) - len(applicable_events),
            "quality_valid_requests": len(valid_events),
            "quality_invalid_requests": len(invalid_events),
            "quality_unassessed_requests": sum(
                event.get("quality_valid") is None for event in applicable_events
            ),
            "structured_complete_1_to_200_requests": sum(
                bool(event.get("quality_details", {}).get("complete_count_1_to_200"))
                for event in applicable_events
            ) if workload == "structured" else None,
            "format_warning_requests": sum(
                bool(event.get("quality_details", {}).get("format_warnings"))
                for event in applicable_events
            ),
            "format_noncompliant_requests": sum(
                not bool(event.get("validation", {}).get("format_compliant", True))
                for event in applicable_events
            ),
            "structured_requests_with_missing_numbers": sum(
                bool(event.get("quality_details", {}).get("missing_numbers"))
                for event in applicable_events
            ) if workload == "structured" else None,
            "structured_requests_with_duplicates": sum(
                bool(event.get("quality_details", {}).get("duplicate_numbers"))
                for event in applicable_events
            ) if workload == "structured" else None,
            "structured_requests_with_out_of_range_numbers": sum(
                bool(event.get("quality_details", {}).get("out_of_range_numbers"))
                for event in applicable_events
            ) if workload == "structured" else None,
            "thinking_only_requests": sum(
                bool(event.get("quality_details", {}).get("thinking_only"))
                for event in applicable_events
            ),
            "cap_truncated_requests": sum(
                bool(event.get("quality_details", {}).get("cap_truncated"))
                for event in applicable_events
            ),
            "quality_failures": [
                {"sequence_index": event.get("sequence_index"), "reasons": event.get("quality_failures", [])}
                for event in invalid_events
            ],
        })

    prefill_groups: dict[tuple[str, int, str, int, str], list[dict[str, Any]]] = {}
    for wave in record.get("waves", []):
        if wave.get("phase") != "prefill-measurement":
            continue
        key = (
            wave["experiment"],
            int(wave.get("requested_context_tokens") or 0),
            wave["workload"],
            wave["concurrency"],
            wave.get("request_policy", {}).get("comparison_key", "unknown"),
        )
        prefill_groups.setdefault(key, []).append(wave)
    prefill_summary: list[dict[str, Any]] = []
    for (experiment, context_tokens, workload, concurrency, policy_key), waves in sorted(prefill_groups.items()):
        events = [event for wave in waves for event in wave["events"]]
        rates = [
            event["prefill_measurement"]["tps"] for event in events
            if event.get("throughput_eligible")
            and isinstance(event.get("prefill_measurement", {}).get("tps"), (int, float))
        ]
        prefill_summary.append({
            "experiment": experiment,
            "requested_context_tokens": context_tokens,
            "target_prompt_tokens": context_tokens,
            "workload": workload,
            "concurrency": concurrency,
            "request_policy_comparison_key": policy_key,
            "throughput_eligible_requests": len(rates),
            "throughput_ineligible_requests": len(events) - len(rates),
            "per_request_prefill_tps_by_wave": [
                {
                    "run_index": wave.get("run_index"),
                    "values": [
                        event.get("prefill_measurement", {}).get("tps")
                        for event in wave.get("events", [])
                    ],
                }
                for wave in waves
            ],
            "per_request_ttft_ms_by_wave": [
                {
                    "run_index": wave.get("run_index"),
                    "values": [event.get("ttft_ms") for event in wave.get("events", [])],
                }
                for wave in waves
            ],
            "actual_usage_prompt_tokens_by_request": [
                event.get("actual_usage_prompt_tokens") for event in events
            ],
            "quality_applicable_requests": sum(event.get("quality_applicable") is True for event in events),
            "quality_not_applicable_requests": sum(event.get("quality_applicable") is False for event in events),
            "median_per_request_prefill_tps": statistics.median(rates) if rates else None,
            "mean_per_request_prefill_tps": statistics.fmean(rates) if rates else None,
            "sample_stddev_per_request_prefill_tps": statistics.stdev(rates) if len(rates) > 1 else None,
            "quality_does_not_gate_timing_summary": True,
        })
    return {"decode": decode_summary, "prefill": prefill_summary, "quality": quality_summary}


def _fast_pass_snapshot(
    record: dict[str, Any],
    args: argparse.Namespace,
    profile: dict[str, Any],
    experiments: list[dict[str, Any]],
    runs: int,
    actual_elapsed_ms: float,
) -> dict[str, Any]:
    measured = [
        event for event in record.get("events", [])
        if event.get("phase") == "measurement" and event.get("measurement_kind") == "decode"
    ]
    completed = [
        event for event in measured
        if event.get("sse_done_seen") is True and event.get("finish_reason") is not None
    ]
    expected = (
        len(experiments) * runs * len(args.workloads) * len(args.decode_context_sizes)
        * sum(args.concurrencies)
    )
    applicable = [event for event in measured if event.get("quality_applicable") is True]
    finish_reasons: dict[str, int] = {}
    for event in measured:
        reason = event.get("finish_reason") or "missing"
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    requested_force = profile["force_to_cap"]
    requested_ignore = profile["ignore_eos_requested"]
    requested_fields = {
        "max_tokens_by_workload": dict(profile["caps"]),
        "temperature_by_workload": dict(_DEFAULT_TEMPERATURE),
        "thinking_by_workload": dict(_DEFAULT_THINKING),
        "top_p": args.top_p,
        "top_k": args.top_k if args.top_k is not None else "omitted",
        "ignore_eos": requested_ignore if requested_ignore is not None else "omitted",
        "min_tokens": "max_tokens" if requested_force else "omitted",
        "stop": [] if requested_force else "omitted",
    }
    signal = record.get("interrupt_signal")
    return {
        "protocol_id": record.get("protocol_id"),
        "protocol_profile": record.get("protocol_profile"),
        "schema": "mia-resident-fast-pass-v1",
        "status": (
            "interrupted_partial" if signal is not None and len(completed) < expected else
            "complete" if len(completed) == expected else
            "in_progress" if record.get("status") == "running" else "partial"
        ),
        "target_window_seconds": [5, 10],
        "target_window_semantics": "observational only; actual duration is reported without pass/fail interpretation",
        "actual_duration_ms": round(actual_elapsed_ms, 6),
        "duration_scope": "resident client from client start; engine load is excluded",
        "requested_completion_count": expected,
        "returned_measurement_request_count": len(measured),
        "actual_completed_stream_count": len(completed),
        "successful_http_response_count": sum(
            isinstance(event.get("http_status"), int) and 200 <= event["http_status"] < 300
            for event in measured
        ),
        "throughput_eligible_request_count": sum(event.get("throughput_eligible") is True for event in measured),
        "completion_definition": "SSE [DONE] observed and a non-null finish_reason received",
        "requested_eos_cap_policy": {
            "mode": "forced_to_cap" if requested_force else "natural_eos",
            **requested_fields,
            "finish_reason_counts": finish_reasons,
        },
        "quality_policy_and_counts": {
            "quality_is_independent_of_throughput": True,
            "cap_truncation_is_reported_separately_and_does_not_change_content_validation": True,
            "applicable_requests": len(applicable),
            "valid_requests": sum(event.get("quality_valid") is True for event in applicable),
            "invalid_requests": sum(event.get("quality_valid") is False for event in applicable),
            "unassessed_requests": sum(event.get("quality_valid") is None for event in applicable),
            "cap_truncated_requests": sum(
                event.get("quality_details", {}).get("cap_truncated") is True for event in applicable
            ),
        },
        "decision_policy": {
            "automatic_timeout_or_kill": False,
            "early_speed_gate": False,
            "speed_or_quality_pass_fail_threshold": None,
            "partial_completion_is_not_a_pass": True,
        },
        "request_fields_by_measurement": [
            {
                "sequence_index": event.get("sequence_index"),
                "workload": event.get("workload"),
                "requested": {
                    **requested_fields,
                    "max_tokens": event.get("requested_output_cap"),
                    "thinking": _DEFAULT_THINKING.get(event.get("workload")),
                },
                "effective": event.get("parameters", {}),
                "effective_request_policy": event.get("effective_request_policy", {}),
                "completion": {
                    "sse_done_seen": event.get("sse_done_seen"),
                    "finish_reason": event.get("finish_reason"),
                    "completion_tokens": event.get("completion_tokens"),
                    "quality_status": event.get("quality_status"),
                    "quality_valid": event.get("quality_valid"),
                },
                "raw_sse_retained_in_event": isinstance(event.get("raw_sse_base64"), str),
                "timing": {
                    "ttft_ms": event.get("ttft_ms"),
                    "decode_window_ms": event.get("decode_window_ms"),
                    "e2e_ms": event.get("e2e_ms"),
                },
            }
            for event in measured
        ],
        "stop_signal": {
            "requested": signal is not None,
            "signal": signal,
            "received_utc": record.get("interrupt_received_utc"),
            "elapsed_ms_at_signal": record.get("interrupt_elapsed_ms"),
            "completed_stream_count_at_signal": record.get("interrupt_completed_stream_count"),
            "behavior": "SIGINT/SIGTERM request graceful stop; in-flight HTTP requests can finish before exit",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serve", "client"), default="client")
    parser.add_argument("--model")
    parser.add_argument("--draft")
    parser.add_argument("--mtp", type=int)
    parser.add_argument("--draft-sample-method", choices=("greedy", "probabilistic"), default="probabilistic")
    parser.add_argument("--rejection-sampling", choices=("standard", "block"), default="standard")
    parser.add_argument("--graphs", choices=("PIECEWISE", "NONE"), default="PIECEWISE")
    parser.add_argument("--out", required=True)
    parser.add_argument("--state-file", default=_DEFAULT_STATE_FILE)
    parser.add_argument("--protocol-profile", choices=tuple(mia_bench.RESIDENT_PROTOCOLS), default="screen")
    parser.add_argument("--experiments", help="JSON list/map of named worker runtime settings")
    parser.add_argument("--output-cap", action="append", default=[], metavar="WORKLOAD=N")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--workloads", choices=("code", "prose", "structured", "prose_pure"), nargs="+",
                        default=["code", "prose", "structured"])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1])
    parser.add_argument("--decode-context-sizes", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--exact-input-tokens",
        action="store_true",
        help="treat decode/prefill sizes as exact final chat input-token budgets, including chat-template tokens",
    )
    parser.add_argument("--prefill-sizes", type=int, nargs="*", default=[])
    parser.add_argument("--prefill-concurrencies", type=int, nargs="+", default=[1])
    parser.add_argument("--prefill-repeats", type=int)
    parser.add_argument("--prefill-only", action="store_true",
                        help="skip all decode warmups and measurements; requires --prefill-sizes")
    parser.add_argument("--ple-diagnostics", action="store_true",
                        help="take opt-in PLE mmap/process diagnostic snapshots around each experiment")
    parser.add_argument("--peer-diagnostics", action="store_true",
                        help="preserve opt-in peer_inventory snapshots around each experiment")
    parser.add_argument("--campaign-index")
    parser.add_argument("--campaign-id")
    parser.add_argument("--campaign-model-id")
    parser.add_argument("--campaign-stage")
    parser.add_argument("--campaign-run-id")
    parser.add_argument("--request-timeout", type=float, default=360,
                        help="urllib socket-idle timeout; timeout streams are retained as failed partial samples")
    parser.add_argument("--server-start-timeout", type=int, default=450)
    parser.add_argument("--single-gb10", action="store_true")
    parser.add_argument("--kv-mib", type=int, default=2048)
    parser.add_argument("--auto-kv", action="store_true",
                        help="omit the fixed byte budget and use the actual vLLM memory planner")
    parser.add_argument("--auto-kv-memory-utilization", type=float, default=0.92)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--served-model-name", default=_SERVE_MODEL_NAME)
    parser.add_argument("--port", type=int, default=18800)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-parser", choices=("qwen3",))
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode == "serve" and (not args.model or not args.draft or args.mtp is None):
        parser.error("--mode serve requires --model, --draft, and --mtp")
    if args.mode == "serve" and args.mtp < 0:
        parser.error("--mtp must be non-negative")
    if args.exact_input_tokens and args.mode != "client":
        parser.error("--exact-input-tokens applies only to --mode client")
    if args.prefill_only and args.mode != "client":
        parser.error("--prefill-only applies only to --mode client")
    if args.prefill_only and not args.prefill_sizes:
        parser.error("--prefill-only requires at least one --prefill-sizes value")
    if (args.ple_diagnostics or args.peer_diagnostics) and args.mode != "client":
        parser.error("worker diagnostics apply only to --mode client")
    if args.campaign_index is not None and args.mode != "client":
        parser.error("campaign indexing applies only to --mode client")
    campaign_values = (
        args.campaign_id, args.campaign_model_id, args.campaign_stage, args.campaign_run_id,
    )
    if args.campaign_index is not None:
        if any(value is None for value in campaign_values):
            parser.error("--campaign-index requires campaign id, model id, stage, and run id")
        if args.campaign_id != resident_campaign.CAMPAIGN_ID:
            parser.error("--campaign-id does not match the approved campaign")
        if args.campaign_model_id != resident_campaign.CAMPAIGN_MODEL_ID:
            parser.error("--campaign-model-id does not match the approved model")
        if not resident_campaign.STAGE_RE.fullmatch(args.campaign_stage):
            parser.error("--campaign-stage must be a short alphanumeric slug")
        if args.campaign_run_id != Path(args.out).stem:
            parser.error("--campaign-run-id must match the output artifact basename")
        if not resident_campaign.RUN_ID_RE.fullmatch(args.campaign_run_id):
            parser.error("--campaign-run-id is outside the supported basename contract")
        if Path(args.campaign_index).parent.resolve() != Path(args.out).parent.resolve():
            parser.error("--campaign-index and --out must be in the same directory")
        if Path(args.out).exists():
            parser.error(f"refusing to overwrite existing campaign artifact: {Path(args.out).name}")
    elif any(value is not None for value in campaign_values):
        parser.error("campaign metadata options require --campaign-index")
    if args.kv_mib < 1 or args.port < 1 or args.port > 65535:
        parser.error("--kv-mib must be positive and --port must be 1..65535")
    if not math.isfinite(args.auto_kv_memory_utilization) or not 0 < args.auto_kv_memory_utilization < 1:
        parser.error("--auto-kv-memory-utilization must be a finite fraction strictly between0 and1")
    if args.runs is not None and args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.prefill_repeats is not None and args.prefill_repeats < 1:
        parser.error("--prefill-repeats must be at least 1")
    if args.warmup_tokens < 0 or args.request_timeout <= 0 or args.server_start_timeout <= 0:
        parser.error("warmup tokens must be non-negative and all timeouts positive")
    if not 0.0 <= args.top_p <= 1.0:
        parser.error("--top-p must be between 0 and 1")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive when supplied")
    if any(concurrency not in (1, 2, 4) for concurrency in args.concurrencies):
        parser.error("decode --concurrencies are restricted to the specified 1, 2, and 4 stream sweep")
    if any(concurrency not in (1, 2, 4) for concurrency in args.prefill_concurrencies):
        parser.error("prefill --prefill-concurrencies are restricted to 1, 2, and 4")
    if any(size < 256 or size > 300000 for size in args.prefill_sizes):
        parser.error("prefill sizes must be between 256 and 300000 tokens")
    if any(size < 0 or size > 300000 for size in args.decode_context_sizes):
        parser.error("decode context sizes must be between 0 and 300000 tokens")
    if args.exact_input_tokens and not args.prefill_only and any(size == 0 for size in args.decode_context_sizes):
        parser.error("--exact-input-tokens requires positive --decode-context-sizes unless --prefill-only is selected")
    if len(set(args.decode_context_sizes)) != len(args.decode_context_sizes):
        parser.error("--decode-context-sizes must not contain duplicates")
    if len(set(args.workloads)) != len(args.workloads) or len(set(args.concurrencies)) != len(args.concurrencies):
        parser.error("workloads and decode concurrency values must not be duplicated")
    if args.max_model_len is not None and args.max_model_len < 256:
        parser.error("--max-model-len must be at least 256")
    if args.max_num_seqs is not None and args.max_num_seqs < max(args.concurrencies + args.prefill_concurrencies):
        parser.error("--max-num-seqs must cover all requested concurrent streams")

def _effective_request_policy(
    event: dict[str, Any], *, force_cap_requested: bool, thinking_requested: bool,
    ignore_eos_requested: bool | None = None,
) -> dict[str, Any]:
    parameters = event.get("parameters", {})
    maximum = parameters.get("max_tokens")
    forced = (
        isinstance(maximum, int) and not isinstance(maximum, bool)
        and parameters.get("min_tokens") == maximum
        and parameters.get("ignore_eos") is True and parameters.get("stop") == []
    )
    natural_fields_omitted = (
        "min_tokens" not in parameters and "ignore_eos" not in parameters and "stop" not in parameters
    )
    natural_fields_explicit = (
        "min_tokens" not in parameters and parameters.get("ignore_eos") is False
        and "stop" not in parameters
    )
    natural_policy_preserved = (
        natural_fields_explicit if ignore_eos_requested is False
        else natural_fields_omitted if ignore_eos_requested is None
        else False
    )
    effective_eos_policy = (
        "forced_to_cap" if forced else
        "natural_explicit_ignore_eos_false" if natural_fields_explicit else
        "natural_fields_omitted" if natural_fields_omitted else
        "inconsistent_or_fallback"
    )
    control = event.get("thinking_control_state", "unknown")
    template = parameters.get("chat_template_kwargs", {})
    explicit_thinking = (
        isinstance(template, dict)
        and template.get("enable_thinking") is thinking_requested
    )
    http_ok = isinstance(event.get("http_status"), int) and 200 <= event["http_status"] < 300
    eos_policy_preserved = forced if force_cap_requested else natural_policy_preserved
    return {
        "schema": "resident-effective-request-policy-v2",
        "force_cap_requested": force_cap_requested,
        "force_cap_fields_sent": forced,
        "force_cap_fallback_used": event.get("force_cap_fallback_used") is True,
        "ignore_eos_requested": (
            True if force_cap_requested else
            ignore_eos_requested if ignore_eos_requested is not None else "omitted"
        ),
        "min_tokens_requested": "max_tokens" if force_cap_requested else "omitted",
        "stop_requested": [] if force_cap_requested else "omitted",
        "effective_eos_policy": effective_eos_policy,
        "eos_policy_preserved": eos_policy_preserved,
        "thinking_requested": thinking_requested,
        "thinking_control_state": control,
        "requested_conditions_preserved": bool(
            http_ok and eos_policy_preserved and explicit_thinking
            and control in ("requested_flags_sent", "explicit_disable_fallback_sent")
        ),
        "comparison_key": (
            f"eos={effective_eos_policy};thinking={'on' if thinking_requested else 'off'};"
            f"control={control}"
        ),
    }


def _request_event(
    *,
    api: ResidentApi,
    url: str,
    model_id: str,
    workload: str,
    prompt: str,
    phase: str,
    max_tokens: int,
    temperature: float,
    thinking: bool,
    top_p: float,
    top_k: int | None,
    force_to_cap: bool,
    ignore_eos: bool | None = None,
    sequence_index: int,
    run_index: int | None,
    require_usage: bool,
    require_prompt_usage: bool,
    require_complete_stream: bool,
    request_timeout: float,
    quality_applicable: bool = True,
) -> dict[str, Any]:
    body = mia_bench._request_body(
        model_id, prompt, max_tokens, temperature, thinking=thinking,
        top_p=top_p, top_k=top_k, force_to_cap=force_to_cap, ignore_eos=ignore_eos,
    )

    def perform(request_body: dict[str, Any]) -> dict[str, Any]:
        return mia_bench._perform_request(
            url=url,
            body=request_body,
            api_key=api.api_key,
            workload=workload,
            prompt=prompt,
            phase=phase,
            sequence_index=sequence_index,
            run_index=run_index,
            require_usage=require_usage,
            require_prompt_usage=require_prompt_usage,
            require_complete_stream=require_complete_stream,
            capture_raw_sse=True,
            request_timeout=request_timeout,
            thinking=thinking,
            quality_applicable=quality_applicable,
        )

    def strip_thinking_flags(request_body: dict[str, Any]) -> dict[str, Any]:
        next_body = dict(request_body)
        template_kwargs = next_body.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict):
            remaining = {
                key: value for key, value in template_kwargs.items()
                if key not in ("enable_thinking", "thinking_mode", "thinking")
            }
            if remaining:
                next_body["chat_template_kwargs"] = remaining
            else:
                next_body.pop("chat_template_kwargs", None)
        for key in (
            "enable_thinking", "thinking", "thinking_mode", "reasoning_effort", "reasoning_budget"
        ):
            next_body.pop(key, None)
        return next_body

    def thinking_off_fallback(request_body: dict[str, Any], *, minimal: bool) -> dict[str, Any]:
        next_body = strip_thinking_flags(request_body)
        if minimal:
            next_body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            next_body["chat_template_kwargs"] = {
                "enable_thinking": False, "thinking": False, "thinking_mode": "disabled"
            }
            next_body["thinking"] = {"type": "disabled"}
            next_body["enable_thinking"] = False
        return next_body

    def snapshot(result: dict[str, Any], stage: str) -> dict[str, Any]:
        return {
            "stage": stage,
            "http_status": result.get("http_status"),
            "parameters": result.get("parameters"),
            "failures": result.get("failures", []),
            "http_error_body": result.get("http_error_body"),
            "raw_sse_base64": result.get("raw_sse_base64"),
        }

    def perform_with_thinking_fallback(request_body: dict[str, Any], stage: str) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        result = perform(request_body)
        attempts.append(snapshot(result, stage))
        fallback_modes: list[str] = []
        if result.get("http_status") == 400 and "chat_template_kwargs" in request_body:
            if thinking:
                result = perform(strip_thinking_flags(request_body))
                attempts.append(snapshot(result, f"{stage}_thinking_flags_stripped"))
                fallback_modes.append("thinking_fields_stripped_after_http_400; effective backend mode unknown")
            else:
                result = perform(thinking_off_fallback(request_body, minimal=False))
                attempts.append(snapshot(result, f"{stage}_thinking_off_fallback"))
                fallback_modes.append("thinking_off_explicit_fallback_after_http_400")
                if result.get("http_status") == 400:
                    result = perform(thinking_off_fallback(request_body, minimal=True))
                    attempts.append(snapshot(result, f"{stage}_thinking_off_minimal"))
                    fallback_modes.append("thinking_off_minimal_fallback_after_http_400")
        result["request_attempts"] = attempts
        result["thinking_requested"] = thinking
        result["thinking_payload_last_attempt"] = result.get("parameters", {}).get(
            "chat_template_kwargs", "omitted"
        )
        result["thinking_fallback_used"] = bool(fallback_modes)
        result["thinking_fallback_modes"] = fallback_modes
        result["thinking_control_state"] = (
            "backend_effective_mode_unknown_after_400_fallback"
            if any("effective backend mode unknown" in mode for mode in fallback_modes)
            else ("explicit_disable_fallback_sent" if fallback_modes else "requested_flags_sent")
        )
        return result

    first = perform_with_thinking_fallback(body, "primary")
    if not force_to_cap or first.get("http_status") != 400:
        first["force_cap_fallback_used"] = False
        first["effective_request_policy"] = _effective_request_policy(
            first, force_cap_requested=force_to_cap, thinking_requested=thinking,
            ignore_eos_requested=ignore_eos,
        )
        return first

    retry_body = {
        key: value for key, value in body.items()
        if key not in ("min_tokens", "ignore_eos", "stop")
    }
    retry = perform_with_thinking_fallback(retry_body, "force_cap_fields_stripped")
    retry["request_attempts"] = first["request_attempts"] + retry["request_attempts"]
    retry["thinking_fallback_modes"] = (
        first["thinking_fallback_modes"] + retry["thinking_fallback_modes"]
    )
    retry["thinking_fallback_used"] = bool(retry["thinking_fallback_modes"])
    retry["force_cap_fallback_used"] = True
    retry["fallback_reason"] = "HTTP 400; retried after removing min_tokens, ignore_eos, and stop"
    retry["effective_request_policy"] = _effective_request_policy(
        retry, force_cap_requested=force_to_cap, thinking_requested=thinking,
        ignore_eos_requested=ignore_eos,
    )
    return retry


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _extract_engine_ids(value: Any) -> list[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            value = item.get("engine_instance_id")
            if isinstance(value, str) and value:
                found.add(value)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(found)


def _engine_identity(api: ResidentApi, state: dict[str, Any]) -> dict[str, Any]:
    response = api.worker_rpc("resident_engine_identity")
    worker_pids = _extract_worker_pids(response)
    engine_ids = _extract_engine_ids(response)
    expected = state.get("engine_instance_id")
    if not worker_pids:
        raise RuntimeError("resident_engine_identity returned no worker PID; identity is not fabricated")
    if not expected or engine_ids != [expected]:
        raise RuntimeError(f"resident engine identity mismatch: expected {expected!r}, observed {engine_ids!r}")
    return {
        "engine_instance_id": expected,
        "api_server_pid": state["api_server_pid"],
        "worker_pids": worker_pids,
        "worker_identity_rpc_result": response,
        "identity_source": "resident_engine_identity RPC plus resident state file API PID",
    }


def _run_wave(
    *,
    record: dict[str, Any],
    save,
    api: ResidentApi,
    api_process: subprocess.Popen | None,
    engine_identity: dict[str, Any],
    workload: str,
    phase: str,
    prompts: list[str],
    experiment_name: str,
    run_index: int | None,
    max_tokens: int,
    temperature: float,
    thinking: bool,
    top_p: float,
    top_k: int | None,
    request_timeout: float,
    require_usage: bool,
    interrupt_event: threading.Event,
    require_prompt_usage: bool = False,
    require_complete_stream: bool = True,
    quality_applicable: bool = True,
    measurement_kind: str = "decode",
    requested_context_tokens: int | None = None,
    requested_prompt_tokens: int | None = None,
    target_prompt_tokens: int | None = None,
    prefill_overhead_ms: float | None | dict[int, float | None] = 0,
    prompt_tokenization: list[dict[str, Any]] | None = None,
    force_to_cap: bool = False,
    ignore_eos: bool | None = None,
) -> dict[str, Any]:
    if prompt_tokenization is not None and len(prompt_tokenization) != len(prompts):
        raise ValueError("prompt tokenization metadata must match the request stream count")
    if api_process is not None:
        if api_process.poll() is not None:
            raise RuntimeError("resident API server exited before the request wave")
    elif not _pid_is_alive(int(record["server"]["api_server_pid"])):
        raise RuntimeError("resident API server PID from the state file is no longer alive")
    observed_identity = api.worker_rpc("resident_engine_identity")
    observed_worker_pids = _extract_worker_pids(observed_identity)
    observed_engine_ids = _extract_engine_ids(observed_identity)
    if observed_worker_pids != engine_identity["worker_pids"] or observed_engine_ids != [engine_identity["engine_instance_id"]]:
        raise RuntimeError(
            f"resident worker identity changed: expected {engine_identity}, "
            f"observed pids={observed_worker_pids}, engine_ids={observed_engine_ids}"
        )
    before_metrics = api.speculative_metrics()
    sequence_indices = list(range(record["next_sequence_index"], record["next_sequence_index"] + len(prompts)))
    record["next_sequence_index"] += len(prompts)
    wave_started = time.perf_counter()
    events: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = {
            executor.submit(
                _request_event,
                api=api,
                url=f"{api.base_url}/v1/chat/completions",
                model_id=record["model_id"],
                workload=workload,
                prompt=prompt,
                phase=phase,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking=thinking,
                force_to_cap=force_to_cap,
                ignore_eos=ignore_eos,
                top_p=top_p,
                top_k=top_k,
                sequence_index=sequence_index,
                run_index=run_index,
                require_usage=require_usage,
                require_prompt_usage=require_prompt_usage,
                require_complete_stream=require_complete_stream,
                request_timeout=request_timeout,
                quality_applicable=quality_applicable,
            ): stream_index
            for stream_index, (prompt, sequence_index) in enumerate(zip(prompts, sequence_indices))
        }
        for future in as_completed(futures):
            stream_index = futures[future]
            try:
                event = future.result()
            except Exception as exc:
                event = {
                    "sequence_index": sequence_indices[stream_index], "phase": phase,
                    "workload": workload, "run_index": run_index, "prompt": prompts[stream_index],
                    "prompt_sha256": hashlib.sha256(prompts[stream_index].encode("utf-8")).hexdigest(),
                    "parameters": {
                        "model": record["model_id"],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k if top_k is not None else "omitted",
                        **({"ignore_eos": ignore_eos} if ignore_eos is not None else {}),
                        **({"min_tokens": max_tokens, "ignore_eos": True, "stop": []} if force_to_cap else {}),
                    },
                    "force_to_cap": force_to_cap,
                    "force_cap_fallback_used": False,
                    "thinking_control_state": "unknown",
                    "output_text": "",
                    "reasoning_text": "",
                    "raw_sse_base64": None,
                    "success": False,
                    "throughput_eligible": False,
                    "quality_applicable": quality_applicable,
                    "quality_valid": None,
                    "quality_status": "unassessed" if quality_applicable else "not_applicable",
                    "failures": [{"stage": "client", "reason": f"{type(exc).__name__}: {exc}"}],
                }
            event.setdefault("effective_request_policy", _effective_request_policy(
                event, force_cap_requested=force_to_cap, thinking_requested=thinking,
                ignore_eos_requested=ignore_eos,
            ))
            event["stream_index"] = stream_index
            event["experiment"] = experiment_name
            event["engine_identity"] = dict(engine_identity)
            tokenization = prompt_tokenization[stream_index] if prompt_tokenization is not None else None
            if tokenization is not None:
                event["input_tokenization"] = tokenization
                target = tokenization["target_input_tokens"]
                actual = event.get("prompt_tokens")
                verified = (
                    tokenization.get("verified_exact_before_generation") is True
                    and tokenization.get("final_tokenizer_prompt_tokens") == target
                    and actual == target
                )
                event["input_token_usage_verification"] = {
                    "target_input_tokens": target,
                    "tokenizer_prompt_tokens": tokenization.get("final_tokenizer_prompt_tokens"),
                    "actual_usage_prompt_tokens": actual,
                    "verified": verified,
                }
                if not verified:
                    event["success"] = False
                    event["throughput_eligible"] = False
                    event.setdefault("failures", []).append({
                        "stage": "input_token_verification",
                        "reason": (
                            f"exact input target {target} was not confirmed by generation "
                            f"usage.prompt_tokens={actual!r}"
                        ),
                    })
            if measurement_kind == "prefill" and phase == "prefill-measurement":
                overhead = (
                    prefill_overhead_ms.get(stream_index, 0.0)
                    if isinstance(prefill_overhead_ms, dict) else prefill_overhead_ms
                )
                event["prefill_measurement"] = _prefill_rate(event, overhead)
                if event["prefill_measurement"].get("tps") is None:
                    event["throughput_eligible"] = False
                    event["success"] = False
                    event.setdefault("failures", []).append({
                        "stage": "prefill_timing",
                        "reason": event["prefill_measurement"].get("reason", "no prefill rate"),
                    })
            event["measurement_kind"] = measurement_kind
            event["input_target_semantics"] = record.get("protocol_metadata", {}).get(
                "input_target_semantics"
            )
            event["base_profile_id"] = record.get(
                "protocol_metadata", {}
            ).get("base_profile_id", record["protocol_id"])
            event["protocol_id"] = record["protocol_id"]
            event["measurement_protocol_id"] = (
                record["protocol_metadata"]["prefill_protocol_id"]
                if measurement_kind == "prefill"
                else (
                    f"{record['protocol_id']}-ctx{requested_context_tokens}"
                    if requested_context_tokens not in (None, 0) else record["protocol_id"]
                )
            )
            event["requested_context_tokens"] = requested_context_tokens
            event["requested_prompt_tokens"] = requested_prompt_tokens
            event["requested_input_tokens"] = (
                tokenization.get("target_input_tokens") if tokenization is not None else None
            )
            event["target_prompt_tokens"] = (
                target_prompt_tokens if target_prompt_tokens is not None else requested_prompt_tokens
            )
            event["requested_output_cap"] = max_tokens
            event["requested_ignore_eos"] = True if force_to_cap else (
                ignore_eos if ignore_eos is not None else "omitted"
            )
            event["requested_min_tokens"] = "max_tokens" if force_to_cap else "omitted"
            event["requested_stop"] = [] if force_to_cap else "omitted"
            event["actual_usage_prompt_tokens"] = event.get("prompt_tokens")
            if event.get("quality_applicable", quality_applicable) is False:
                event["quality_applicable"] = False
                event["quality_valid"] = None
                event["quality_status"] = "not_applicable"
            else:
                event["quality_applicable"] = True
                event.setdefault("quality_status", "valid" if event.get("quality_valid") is True else "invalid")
            if workload == "code":
                event["code_task_name"] = _code_task_name(prompts[stream_index])
                event["task_validation_scope"] = (
                    "generic Python nonempty/preface checks and AST parse diagnostics only; no per-task function-name or behavior validation"
                )
            for failure in event.get("failures", []):
                record["failures"].append({
                    "sequence_index": event["sequence_index"],
                    "phase": phase,
                    "workload": workload,
                    "run_index": run_index,
                    **failure,
                })
            events.append(event)
            record["events"].append(event)
            save()
    events.sort(key=lambda event: event["stream_index"])
    after_metrics = api.speculative_metrics()
    policy_keys = sorted({
        event.get("effective_request_policy", {}).get("comparison_key", "unknown")
        for event in events
    })
    request_policy = {
        "schema": "resident-wave-request-policy-v2",
        "requested_force_to_cap": force_to_cap,
        "requested_ignore_eos": True if force_to_cap else (
            ignore_eos if ignore_eos is not None else "omitted"
        ),
        "requested_min_tokens": "max_tokens" if force_to_cap else "omitted",
        "requested_stop": [] if force_to_cap else "omitted",
        "conditions_preserved_all_streams": len(events) == len(prompts) and all(
            event.get("effective_request_policy", {}).get("requested_conditions_preserved") is True
            for event in events
        ),
        "effective_policy_keys": policy_keys,
        "comparison_key": "|".join(policy_keys),
    }
    wave_elapsed = (time.perf_counter() - wave_started) * 1000
    wave_metrics = _aggregate_wave(
        len(prompts), events, wave_elapsed, workload=workload, phase=phase
    )
    wave = {
        "experiment": experiment_name,
        "phase": phase,
        "workload": workload,
        "run_index": run_index,
        "base_profile_id": record.get("protocol_metadata", {}).get("base_profile_id", record["protocol_id"]),
        "protocol_id": record["protocol_id"],
        "measurement_protocol_id": (
            record["protocol_metadata"]["prefill_protocol_id"]
            if measurement_kind == "prefill"
            else (
                f"{record['protocol_id']}-ctx{requested_context_tokens}"
                if requested_context_tokens not in (None, 0) else record["protocol_id"]
            )
        ),
        "measurement_kind": measurement_kind,
        "request_policy": request_policy,
        "requested_context_tokens": requested_context_tokens,
        "requested_prompt_tokens": requested_prompt_tokens,
        "requested_input_tokens": (
            prompt_tokenization[0]["target_input_tokens"] if prompt_tokenization else None
        ),
        "input_target_semantics": record.get("protocol_metadata", {}).get("input_target_semantics"),
        "input_tokenization_by_stream": [event.get("input_tokenization") for event in events],
        "input_token_usage_verification_by_stream": [
            event.get("input_token_usage_verification") for event in events
        ],
        "target_prompt_tokens": target_prompt_tokens if target_prompt_tokens is not None else requested_prompt_tokens,
        "actual_prompt_tokens": (
            sum(event["actual_usage_prompt_tokens"] for event in events)
            if len(events) == len(prompts) and bool(events) and all(
                isinstance(event.get("actual_usage_prompt_tokens"), int)
                and not isinstance(event.get("actual_usage_prompt_tokens"), bool)
                for event in events
            ) else None
        ),
        "actual_prompt_tokens_by_stream": [event.get("actual_usage_prompt_tokens") for event in events],
        "requested_output_cap": max_tokens,
        "temperature": temperature,
        "thinking": thinking,
        "top_p": top_p,
        "top_k": top_k if top_k is not None else "omitted",
        "force_to_cap": force_to_cap,
        "ignore_eos_requested": True if force_to_cap else (
            ignore_eos if ignore_eos is not None else "omitted"
        ),
        "min_tokens_requested": "max_tokens" if force_to_cap else "omitted",
        "stop_requested": [] if force_to_cap else "omitted",
        "concurrency": len(prompts),
        "engine_identity": dict(engine_identity),
        "identity_rpc_worker_pids_before_wave": observed_worker_pids,
        "events": events,
        "metrics": wave_metrics,
        "speculative_counters": _metric_delta(before_metrics, after_metrics),
        "speculative_metrics_before": before_metrics,
        "speculative_metrics_after": after_metrics,
    }
    record["waves"].append(wave)
    save()
    if interrupt_event.is_set():
        record["status"] = "interrupted"
        save()
    return wave


def _protocol_metadata(args: argparse.Namespace, profile: dict[str, Any], runs: int) -> dict[str, Any]:
    return {
        **profile["metadata"],
        "runs_per_workload_concurrency": runs,
        "workloads": list(args.workloads),
        "concurrencies": list(args.concurrencies),
        "code_validation_scope": "generic nonempty/preface checks and AST parse diagnostics only; AST parse failure at a truncation cap is not an automatic quality failure, and no per-task function-name or behavior check runs",
        "structured_validation_scope": "numbers-only format, count from 1 to 200, order, missing/duplicate/out-of-range/non-integer values, thinking-only output, and cap truncation are reported separately from throughput",
        "prefill_quality_scope": "short prefill outputs are quality not_applicable and do not gate throughput",
        "exact_input_tokens": args.exact_input_tokens,
        "input_target_semantics": (
            _EXACT_INPUT_TARGET_SEMANTICS if args.exact_input_tokens
            else _LEGACY_INPUT_TARGET_SEMANTICS
        ),
        "input_tokenization_method": (
            "resident API POST /tokenize using the exact messages and chat_template_kwargs; "
            "usage.prompt_tokens must match before the sample is throughput-eligible"
            if args.exact_input_tokens else "not queried; actual usage.prompt_tokens is recorded"
        ),
        "decode_context_sizes": list(args.decode_context_sizes),
        "decode_context_prompt_builder": (
            "exact final chat input-token target including chat template and generation/thinking "
            "header; fresh salted LCG filler is adjusted using resident API /tokenize and the exact "
            "Cn task remains unchanged at the tail"
            if args.exact_input_tokens else
            "pinned contextFill.js buildContextPrompt wrapper; zero preserves the exact source prompt; positive sizes use fresh salted LCG filler and retain the exact Cn task at the tail; target is approximate and usage is separate"
        ),
        "prefill_protocol_id": (
            mia_bench.RESIDENT_PREFILL_PROTOCOL_ID + "-exact-input-tokens-v1"
            if args.exact_input_tokens else mia_bench.RESIDENT_PREFILL_PROTOCOL_ID
        ),
        "prefill_sizes": list(args.prefill_sizes),
        "prefill_concurrencies": list(args.prefill_concurrencies),
        "prefill_repeats_per_size": args.prefill_repeats,
        "prefill_only": args.prefill_only,
        "decode_measurements_skipped": args.prefill_only,
        "prefill_warmup": (
            "one fresh-salt generic 512-token warmup and one fresh-salt generic 4096-token warmup, "
            "then three unique nominal 16-token fixed-overhead calibration prompts per experiment"
        ),
        "prefill_prompt_builder": (
            "exact final chat input-token target including chat template and generation header; "
            "fresh UUID salt and filler-only adjustment preserve the actual workload's Cn task tail"
            if args.exact_input_tokens else
            "per-workload extension: fresh UUID salt, LCG word filler, and the actual workload's Cn task prompt as tail; target length is approximate and actual usage is stored separately"
        ),
        "prefill_calibration": (
            "three unique pinned generic nominal 16-token prompts per experiment remain unchanged and outside exact prefill-size targets"
            if args.exact_input_tokens else
            "three unique pinned generic nominal 16-token prompts per experiment, shared fixed overhead; absent calibration makes TTFT-derived rates ineligible"
        ),
        "prefill_rate_formula": "actual uncached prompt usage tokens / max(TTFT - median of three 16-token fixed-overhead TTFT samples, TTFT * 0.5); no prompt-token estimate substituted",
        "concurrent_prefill_note": "each code/prose/structured workload is measured independently; C=2/4 are extensions beyond sequential sparkDash PrefillBench",
        "force_to_cap": profile["force_to_cap"],
        "requested_eos_policy": {
            "mode": "forced_to_cap" if profile["force_to_cap"] else "natural_eos",
            "max_tokens_by_workload": dict(profile["caps"]),
            "ignore_eos": profile["ignore_eos_requested"] if profile["ignore_eos_requested"] is not None else "omitted",
            "min_tokens": "max_tokens" if profile["force_to_cap"] else "omitted",
            "stop": [] if profile["force_to_cap"] else "omitted",
        },
        "quality_policy": {
            "source_prompts_unchanged": True,
            "raw_sse_and_reasoning_answer_decode_window_preserved": True,
            "throughput_requires_complete_sse_usage_and_positive_visible_decode_window": True,
            "quality_is_reported_separately_from_throughput": True,
            "cap_truncation_is_reported_separately_and_does_not_change_content_validation": True,
            "code_validation_scope": "generic Python nonempty/preface checks and AST parse diagnostics only; no per-task behavior claim",
            "prose_validation_scope": "nonempty answer and reasoning-leak checks only; no semantic quality-equivalence claim",
        },
        "server_generation_config_top_k_override": 20,
        "effective_top_k": args.top_k if args.top_k is not None else 20,
        "server_top_k_override": "--override-generation-config {\"top_k\":20}",
        "request_top_k": args.top_k if args.top_k is not None else "omitted",
        "async_scheduling": False,
        "draft_sample_method": args.draft_sample_method,
        "fast_pass_policy": (
            {
                "target_window_seconds": [5, 10],
                "duration_scope": "actual resident-client elapsed time excluding engine load",
                "window_is_observational_only": True,
                "automatic_timeout_or_kill": False,
                "speed_or_quality_pass_fail_gate": False,
                "completion_requires_sse_done_and_finish_reason": True,
            }
            if args.protocol_profile == "screen" else None
        ),
        "rejection_sample_method": args.rejection_sampling,
    }


def _server_dimensions(args: argparse.Namespace, caps: dict[str, int]) -> tuple[int, int]:
    maximum_input = max(args.prefill_sizes + args.decode_context_sizes, default=0)
    max_model_len = args.max_model_len or max(
        1024, ((maximum_input + max(max(caps.values()), 8) + 255) // 256) * 256
    )
    max_num_seqs = args.max_num_seqs or max(args.concurrencies + args.prefill_concurrencies)
    required_len = maximum_input + max(max(caps.values()), 8)
    if max_model_len < required_len:
        raise ValueError(
            f"max_model_len={max_model_len} is below requested prompt-plus-reply capacity {required_len}"
        )
    if max_num_seqs < max(args.concurrencies + args.prefill_concurrencies):
        raise ValueError("max_num_seqs is below a requested decode/prefill concurrency")
    return max_model_len, max_num_seqs


def _resident_control_snapshots(value: Any) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if isinstance(item.get("engine"), dict) and isinstance(item.get("capabilities"), dict):
                snapshots.append(item)
            else:
                for nested in item.values():
                    visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return snapshots


def _peer_inventory_workers(value: Any) -> list[dict[str, Any]]:
    workers: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if "runtime_controls" in item:
                workers.append(item)
                return
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return workers


def _peer_control_snapshot(value: Any) -> dict[str, Any]:
    worker_statuses = _peer_inventory_workers(value)
    if not worker_statuses:
        raise RuntimeError("peer_inventory returned no worker runtime_controls")

    workers = []
    states = []
    for worker_index, worker in enumerate(worker_statuses):
        controls = worker.get("runtime_controls")
        if not isinstance(controls, dict):
            raise RuntimeError(f"peer_inventory worker {worker_index} runtime_controls is invalid")
        async_state = controls.get("peer_async_transfers")
        arenas = async_state.get("arena_states") if isinstance(async_state, dict) else None
        environment_enabled = async_state.get("environment_enabled") if isinstance(async_state, dict) else None
        consistent = async_state.get("consistent") if isinstance(async_state, dict) else None
        if type(environment_enabled) is not bool or not isinstance(arenas, list) or type(consistent) is not bool:
            raise RuntimeError(f"peer_inventory worker {worker_index} lacks async-transfer state")
        arena_values = []
        for arena in arenas:
            if not isinstance(arena, dict) or type(arena.get("async_transfers")) is not bool:
                raise RuntimeError(f"peer_inventory worker {worker_index} has invalid transfer-arena state")
            arena_values.append(arena["async_transfers"])
        if arenas:
            if consistent is not True or any(enabled != environment_enabled for enabled in arena_values):
                raise RuntimeError(
                    f"peer_async_transfers is inconsistent across environment and arenas on worker {worker_index}"
                )
        elif consistent is not False:
            raise RuntimeError(
                f"peer_inventory worker {worker_index} must mark an empty transfer-arena set inconsistent"
            )

        shared_state = controls.get("peer_shared_overlap")
        shared_enabled = shared_state.get("runtime_enabled") if isinstance(shared_state, dict) else None
        shared_environment_enabled = (
            shared_state.get("environment_enabled") if isinstance(shared_state, dict) else None
        )
        shared_consistent = shared_state.get("consistent") if isinstance(shared_state, dict) else None
        if (
            type(shared_enabled) is not bool
            or type(shared_environment_enabled) is not bool
            or type(shared_consistent) is not bool
            or shared_consistent is not True
            or shared_enabled != shared_environment_enabled
        ):
            raise RuntimeError(f"peer_shared_overlap is inconsistent on worker {worker_index}")

        state = {
            "peer_async_transfers": environment_enabled,
            "peer_shared_overlap": shared_enabled,
        }
        states.append(state)
        workers.append({
            "worker_index": worker_index,
            "peer_async_transfers": {
                "environment_enabled": environment_enabled,
                "arena_states": arenas,
                "consistent": consistent,
            },
            "peer_shared_overlap": {
                "runtime_enabled": shared_enabled,
                "environment_enabled": shared_environment_enabled,
                "consistent": shared_consistent,
            },
        })

    expected = states[0]
    if any(state != expected for state in states[1:]):
        raise RuntimeError("peer transport controls differ across worker RPC results")
    return {
        "state": expected,
        "workers": workers,
        "arena_count": sum(len(worker["peer_async_transfers"]["arena_states"]) for worker in workers),
    }


def _verify_peer_control_state(
    inventory: Any, expected: dict[str, bool], *, require_arena_state: bool = False,
) -> dict[str, Any]:
    if set(expected) != set(_PEER_CONTROL_METHODS) or any(type(value) is not bool for value in expected.values()):
        raise ValueError("peer transport target must contain exact boolean values for both controls")
    snapshot = _peer_control_snapshot(inventory)
    if snapshot["state"] != expected:
        raise RuntimeError(
            f"peer transport worker state mismatch: expected {expected!r}, observed {snapshot['state']!r}"
        )
    if require_arena_state and snapshot["arena_count"] == 0:
        raise RuntimeError("peer transport settings have no live transfer-arena state after warmup")
    return snapshot


def _require_peer_control_candidate_change(
    candidate: dict[str, bool], baseline: dict[str, bool],
) -> None:
    if (
        len(candidate) != 1
        or set(candidate) - set(_PEER_CONTROL_METHODS)
        or any(type(value) is not bool for value in candidate.values())
        or set(baseline) != set(_PEER_CONTROL_METHODS)
        or any(type(value) is not bool for value in baseline.values())
    ):
        raise ValueError("peer-control comparison requires one boolean candidate and a complete boolean baseline")
    for key, value in candidate.items():
        if baseline[key] == value:
            raise ValueError(
                f"peer-control candidate {key}={value!r} matches the fresh live baseline; "
                "no A/B change would be exercised"
            )



def _peer_rpc_responses(value: Any) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if "enabled" in item:
                responses.append(item)
            elif "results" in item:
                visit(item["results"])
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return responses


def _set_peer_control_state(
    api: ResidentApi,
    expected: dict[str, bool],
    *,
    response_sink: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if set(expected) != set(_PEER_CONTROL_METHODS) or any(type(value) is not bool for value in expected.values()):
        raise ValueError("peer transport target must contain exact boolean values for both controls")
    setter_responses = response_sink if response_sink is not None else []
    for method in _PEER_CONTROL_METHODS:
        response = api.worker_rpc(method, kwargs={"enabled": expected[method]})
        setter_record = {"method": method, "response": response}
        setter_responses.append(setter_record)
        rejections = _explicit_control_rejection(response)
        if response is None or rejections:
            raise RuntimeError("; ".join(rejections or [f"{method} returned no payload"]))
        workers = _peer_rpc_responses(response)
        if not workers or any(type(worker.get("enabled")) is not bool or worker["enabled"] != expected[method]
                              for worker in workers):
            raise RuntimeError(f"{method} setter response did not verify the requested worker state")
    return setter_responses


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _peer_counter_deltas(before: Any, after: Any) -> dict[str, Any]:
    before_workers = _peer_inventory_workers(before)
    after_workers = _peer_inventory_workers(after)
    if not before_workers or len(before_workers) != len(after_workers):
        return {
            "status": "unavailable",
            "reason": "peer_inventory_worker_count_missing_or_changed",
            "workers": [],
        }
    rows = []
    counter_names = ("shared_overlap_defers", "async_event_bridges")
    for worker_index, (before_worker, after_worker) in enumerate(zip(before_workers, after_workers)):
        before_counters = before_worker.get("counters")
        after_counters = after_worker.get("counters")
        row: dict[str, Any] = {"worker_index": worker_index, "deltas": {}, "reasons": {}}
        for name in counter_names:
            left = before_counters.get(name) if isinstance(before_counters, dict) else None
            right = after_counters.get(name) if isinstance(after_counters, dict) else None
            if not _number(left) or not _number(right):
                row["deltas"][name] = None
                row["reasons"][name] = "counter_missing_before_or_after"
            elif right < left:
                row["deltas"][name] = None
                row["reasons"][name] = "counter_reset_during_measurement"
            else:
                row["deltas"][name] = right - left
        rows.append(row)
    return {"status": "captured", "workers": rows}


def _restore_peer_control_state(
    api: ResidentApi, baseline: dict[str, bool],
) -> dict[str, Any]:
    restoration: dict[str, Any] = {
        "settings_requested": dict(baseline),
        "source": "fresh worker peer_inventory at client entry",
    }
    try:
        restoration["status_before"] = api.worker_rpc("peer_inventory")
        restoration["worker_state_before"] = _peer_control_snapshot(restoration["status_before"])
        restoration["setter_responses"] = _set_peer_control_state(api, baseline)
        restoration["status_after"] = api.worker_rpc("peer_inventory")
        restoration["worker_state_after"] = _verify_peer_control_state(
            restoration["status_after"], baseline,
        )
        restoration["status"] = "restored_and_verified"
    except Exception as exc:
        restoration["status"] = "restore_failed"
        restoration["failure"] = f"{type(exc).__name__}: {exc}"
    return restoration


def _same_control_value(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool) \
            and isinstance(right, (int, float)) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def _resident_control_value(snapshot: dict[str, Any], key: str) -> Any:
    if key == "code_draft_tokens":
        if (
            snapshot.get("capabilities", {}).get(key, {}).get("enabled") is not True
            or snapshot.get("execution", {}).get("code_draft_depth_patch_loaded") is not True
        ):
            raise RuntimeError("resident worker does not report the code-depth capability and runner patch")
        engine = snapshot.get("engine", {})
        value = engine.get(key)
        maximum = engine.get("max_speculative_tokens")
        if (
            type(maximum) is not int or maximum < 1
            or type(value) is not int or not 1 <= value <= maximum
        ):
            raise RuntimeError("resident_spec_status omitted exact integer code_draft_tokens within Kmax")
        return value
    if key == "compact_draft_head_enabled":
        enabled = snapshot.get("engine", {}).get(key)
        if type(enabled) is not bool:
            raise RuntimeError("resident_spec_status omitted the exact compact draft-head enabled state")
        return enabled
    if key == "qad_hot_cache_enabled":
        hot_cache = snapshot.get("qad_hot_cache", {})
        layers = hot_cache.get("layers")
        enabled = hot_cache.get("enabled")
        if (
            hot_cache.get("ready") is not True
            or type(layers) is not int
            or layers != 48
            or type(enabled) is not bool
        ):
            raise RuntimeError("resident_spec_status omitted the ready 48-layer QAD HotCache state")
        return enabled
    if key == "ple_gather_fast_rows":
        gather = snapshot.get("ple_gather", {})
        values = [entry.get("fast_rows") for entry in gather.get("tables", [])]
        if gather.get("ready") is not True or not values or any(type(value) is not int or not 0 <= value <= 512 for value in values):
            raise RuntimeError("resident_spec_status omitted loaded PLE gather thresholds")
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError("resident PLE tables report inconsistent gather thresholds")
        return values[0]
    if key != "ple_madvise":
        if key not in snapshot["engine"]:
            raise RuntimeError(f"resident_spec_status omitted active control value {key}")
        return snapshot["engine"][key]
    ple = snapshot.get("ple_mmap", {})
    tables = ple.get("tables", [])
    modes = [table.get("mode") for table in tables]
    if ple.get("ready") is not True or not modes or any(mode not in ("normal", "random") for mode in modes):
        raise RuntimeError("resident_spec_status omitted loaded PLE mapping hints")
    if any(mode != modes[0] for mode in modes[1:]):
        raise RuntimeError("resident PLE tables report inconsistent mapping hints")
    return modes[0]

def _resident_baseline_settings(status: Any) -> dict[str, Any]:
    snapshots = _resident_control_snapshots(status)
    if not snapshots:
        raise RuntimeError("resident_spec_status returned no worker control snapshots")
    if any(snapshot.get("status") != "ready" for snapshot in snapshots):
        raise RuntimeError("resident_spec_status did not report ready on every worker")
    settings: dict[str, Any] = {}
    for key in (
        "draft_scale",
        "prose_max_draft_tokens",
        "code_draft_tokens",
        "draft_scale_per_pos",
        "prose_confidence_threshold",
        "prose_draft_top_k",
        "prose_draft_head_mode",
        "compact_draft_head_enabled",
        "ple_madvise",
        "ple_gather_fast_rows",
        "target_only",
        "qad_hot_cache_enabled",
    ):
        enabled = [
            snapshot.get("capabilities", {}).get(key, {}).get("enabled") is True
            for snapshot in snapshots
        ]
        if any(enabled) and not all(enabled):
            raise RuntimeError(f"resident worker capability mismatch for {key}")
        if not all(enabled):
            continue
        values = []
        for snapshot in snapshots:
            values.append(_resident_control_value(snapshot, key))
        if any(not _same_control_value(values[0], value) for value in values[1:]):
            raise RuntimeError(f"resident workers report inconsistent baseline {key} values")
        settings[key] = values[0]
    return settings

def _add_supported_draft_head_metadata(
    profile: dict[str, Any], status: Any
) -> None:
    snapshots = _resident_control_snapshots(status)
    if not snapshots or any(
        snapshot.get("capabilities", {}).get("prose_draft_head_mode", {}).get("enabled") is not True
        for snapshot in snapshots
    ):
        return
    control = profile["metadata"].get("resident_spec_control")
    if not isinstance(control, dict):
        return
    supported = control.get("supported_live_settings")
    if not isinstance(supported, dict):
        return
    supported["prose_draft_head_mode"] = (
        "exact int4/bf16; defaults to int4; prepares BF16 once through the existing MTP builder "
        "from the private INT4 head's selected source rows, without target-head or target-weight changes"
    )
    control["draft_scale_application_evidence"] = (
        "resident status reports the configured logits_processor.scale value only; "
        "Q7 runtime output comparison is required to prove its effect on INT4 and BF16 logits"
    )
    status_fields = control.get("status_fields")
    if isinstance(status_fields, list) and "prose_draft_head_mode/bytes/scale/source scope" not in status_fields:
        status_fields.append("prose_draft_head_mode/bytes/scale/source scope")
    if isinstance(status_fields, list) and (
        "configured scale only; Q7 output comparison required to prove application" not in status_fields
    ):
        status_fields.append("configured scale only; Q7 output comparison required to prove application")
def _add_supported_compact_draft_head_metadata(
    profile: dict[str, Any], status: Any
) -> None:
    snapshots = _resident_control_snapshots(status)
    if not snapshots or any(
        snapshot.get("capabilities", {}).get("compact_draft_head_enabled", {}).get("enabled") is not True
        for snapshot in snapshots
    ):
        return
    control = profile["metadata"].get("resident_spec_control")
    if not isinstance(control, dict):
        return
    supported = control.get("supported_live_settings")
    if isinstance(supported, dict):
        supported["compact_draft_head_enabled"] = (
            "exact boolean, defaults false; true prepares exact dense BF16 MTP lm_head rows once; "
            "false restores the retained original full compute_logits path; no reload or INT4 requantization"
        )
    status_fields = control.get("status_fields")
    if isinstance(status_fields, list):
        for field in (
            "compact_draft_head prepared/enabled/mode/selected rows/bytes/source evidence",
            "compact draft_scale positive-temperature policy",
        ):
            if field not in status_fields:
                status_fields.append(field)
    control["compact_draft_head_comparison"] = {
        "disabled_mode": "full_bf16 via the retained original MTP compute_logits callable",
        "enabled_mode": "compact_bf16 selected rows from the actual dense BF16 MTP lm_head",
        "selection": "QWEN38_COMPACT_DRAFT_VOCAB; tokenizer.json SHA256 must equal QWEN38_COMPACT_DRAFT_TOKENIZER_SHA256",
        "all_workloads": True,
        "target_weights_or_logits_modified": False,
        "rejection_sampling": "standard",
        "draft_scale_policy": "configured MTP logits_processor.scale is applied to compact proposal logits only for all-positive-temperature batches; zero-temperature and mixed batches remain unscaled",
    }

def _draft_scale_per_pos_capability(status: Any) -> tuple[bool, int | None, str | None]:
    snapshots = _resident_control_snapshots(status)
    if not snapshots:
        return False, None, "resident_spec_status returned no worker control snapshots"
    capabilities = [
        snapshot.get("capabilities", {}).get("draft_scale_per_pos", {})
        for snapshot in snapshots
    ]
    enabled = [item.get("enabled") is True for item in capabilities if isinstance(item, dict)]
    if len(enabled) != len(snapshots):
        return False, None, "per-position scale capability was not reported by every worker"
    if any(enabled) and not all(enabled):
        raise RuntimeError("resident worker capability mismatch for draft_scale_per_pos")
    if not all(enabled):
        reason = next(
            (item.get("reason") for item in capabilities
             if isinstance(item, dict) and isinstance(item.get("reason"), str)),
            "per-position scale source patch is not enabled",
        )
        return False, None, reason
    maximums = [item.get("max_length") for item in capabilities]
    if any(type(value) is not int or value < 1 for value in maximums):
        raise RuntimeError("resident per-position scale capability omitted a positive max_length")
    if any(value != maximums[0] for value in maximums[1:]):
        raise RuntimeError("resident workers report inconsistent per-position scale limits")
    return True, maximums[0], None


def _add_supported_draft_scale_per_pos_metadata(
    profile: dict[str, Any], status: Any,
) -> None:
    enabled, maximum, reason = _draft_scale_per_pos_capability(status)
    control = profile["metadata"].get("resident_spec_control")
    if not isinstance(control, dict):
        return
    control["draft_scale_per_pos_capability"] = {
        "enabled": enabled,
        "reason": reason,
        "max_length": maximum,
    }
    supported = control.get("supported_live_settings")
    if not isinstance(supported, dict) or not enabled:
        return
    supported["draft_scale_per_pos"] = (
        f"null restores global draft_scale, or a positive finite relative-scale list "
        f"of at most {maximum} zero-based positions; all-positive-temperature batches only"
    )
    status_fields = control.get("status_fields")
    if isinstance(status_fields, list):
        for field in (
            "draft_scale_per_pos actual value (null or relative list)",
            "per-position capability and maximum supported list length",
        ):
            if field not in status_fields:
                status_fields.append(field)




def _verify_resident_control_settings(status: Any, settings: dict[str, Any]) -> None:
    snapshots = _resident_control_snapshots(status)
    if not snapshots:
        raise RuntimeError("resident_spec_status returned no worker control snapshots")
    if any(snapshot.get("status") != "ready" for snapshot in snapshots):
        raise RuntimeError("resident_spec_status did not report ready on every worker")
    for snapshot in snapshots:
        for key, expected in settings.items():
            if key == "code_draft_tokens":
                maximum = snapshot.get("engine", {}).get("max_speculative_tokens")
                if (
                    type(maximum) is not int or maximum < 1
                    or type(expected) is not int or not 1 <= expected <= maximum
                ):
                    raise RuntimeError("code_draft_tokens must be an exact integer within 1..Kmax")
            if key == "draft_scale_per_pos":
                capability = snapshot.get("capabilities", {}).get(key, {})
                maximum = capability.get("max_length")
                if capability.get("enabled") is not True or type(maximum) is not int or maximum < 1:
                    raise RuntimeError("resident worker does not report supported per-position scale state")
                if expected is not None and (
                    type(expected) not in (list, tuple)
                    or len(expected) > maximum
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) <= 0
                        for value in expected
                    )
                ):
                    raise RuntimeError("per-position scales must be null or finite positive values within Kmax")
            if key == "prose_draft_head_mode":
                capability = snapshot.get("capabilities", {}).get(key, {})
                head = snapshot.get("draft_head")
                if not isinstance(head, dict):
                    raise RuntimeError("resident draft-head status is missing its state object")
                head_scale = head.get("scale")
                if not isinstance(head_scale, dict):
                    raise RuntimeError("resident draft-head status is missing its scale binding")
                actual_scale = snapshot.get("engine", {}).get("draft_scale")
                active_bytes = head.get("active_bytes")
                if capability.get("enabled") is not True:
                    raise RuntimeError("resident worker does not report the draft-head mode capability")
                if type(expected) is not str or expected not in ("int4", "bf16"):
                    raise RuntimeError("resident draft-head mode must be exactly int4 or bf16")
                if (
                    head.get("mode") != expected
                    or type(active_bytes) is not int
                    or active_bytes <= 0
                    or head.get("implicit_prepare_or_fallback") is not False
                ):
                    raise RuntimeError("resident draft-head status omitted the actual mode/bytes or permits a hidden fallback")
                if expected == "bf16" and (
                    head.get("prepared") is not True
                    or head.get("bf16_bytes") != active_bytes
                ):
                    raise RuntimeError("resident BF16 draft-head mode is not backed by its prepared state and bytes")
                if expected == "int4" and (
                    head.get("int4_state_ready") is not True
                    or head.get("int4_bytes") != active_bytes
                ):
                    raise RuntimeError("resident INT4 draft-head mode is not backed by its existing state and bytes")
                bound_scale = head_scale.get("logits_processor_scale")
                if (
                    head_scale.get("application_verified") is not False
                    or head_scale.get("application_evidence_required")
                    != "Q7 runtime output comparison for both INT4 and BF16 head paths"
                    or isinstance(bound_scale, bool)
                    or not isinstance(bound_scale, (int, float))
                    or not math.isfinite(float(bound_scale))
                    or not _same_control_value(bound_scale, actual_scale)
                ):
                    raise RuntimeError("resident draft-head status omitted the configured scale or Q7 evidence caveat")
            if key == "compact_draft_head_enabled":
                capability = snapshot.get("capabilities", {}).get(key, {})
                head = snapshot.get("compact_draft_head")
                if type(expected) is not bool:
                    raise RuntimeError("compact draft-head setting must be an exact boolean")
                if capability.get("enabled") is not True or not isinstance(head, dict):
                    raise RuntimeError("resident worker does not report compact BF16 head capability and state")
                expected_mode = "compact_bf16" if expected else "full_bf16"
                if (
                    head.get("enabled") is not expected
                    or head.get("mode") != expected_mode
                    or type(head.get("prepared")) is not bool
                ):
                    raise RuntimeError("resident compact draft-head mode does not match the requested bool")
                prepared = head["prepared"]
                selected_count = head.get("selected_count")
                persistent_bytes = head.get("persistent_bytes")
                evidence = head.get("source_evidence")
                if prepared and isinstance(evidence, dict):
                    if type(selected_count) is not int or selected_count <= 0:
                        raise RuntimeError("prepared compact-head selected_count must be positive")
                    source_shape = evidence.get("source_weight_shape")
                    input_features = evidence.get("source_input_features")
                    source_vocab_size = evidence.get("source_vocab_size")
                    if (
                        not isinstance(source_shape, list) or len(source_shape) != 2
                        or any(type(size) is not int or size <= 0 for size in source_shape)
                        or type(input_features) is not int or input_features <= 0
                        or source_shape[1] != input_features
                        or type(source_vocab_size) is not int or source_vocab_size <= 0
                        or source_shape[0] < source_vocab_size
                        or selected_count > source_vocab_size
                        or evidence.get("source_weight_dtype") != "torch.bfloat16"
                    ):
                        raise RuntimeError("compact-head source evidence does not prove dense BF16 vocabulary geometry")
                if prepared:
                    if (
                        type(selected_count) is not int or selected_count <= 0
                        or type(persistent_bytes) is not int or persistent_bytes <= 0
                        or not isinstance(evidence, dict)
                        or evidence.get("source_kind") != "dense_bf16_mtp_lm_head"
                        or evidence.get("ids_validated") is not True
                        or evidence.get("actual_tokenizer_sha256")
                        != evidence.get("expected_tokenizer_sha256")
                        or evidence.get("source_tokenizer_sha256")
                        != evidence.get("actual_tokenizer_sha256")
                        or not isinstance(evidence.get("vocab_metadata_sha256"), str)
                        or len(evidence.get("vocab_metadata_sha256", "")) != 64
                        or evidence.get("vocab_sha256_verified") is not True
                        or evidence.get("vocab_file_sha256")
                        != evidence.get("vocab_output_sha256")
                        or not isinstance(evidence.get("vocab_output_sha256"), str)
                        or len(evidence.get("vocab_output_sha256", "")) != 64
                    ):
                        raise RuntimeError("prepared compact head is missing validated BF16 rows, bytes, or tokenizer evidence")
                    weight_bytes = head.get("selected_weight_bytes")
                    ids_bytes = head.get("selected_ids_bytes")
                    if (
                        type(weight_bytes) is not int or weight_bytes <= 0
                        or type(ids_bytes) is not int or ids_bytes <= 0
                        or persistent_bytes != weight_bytes + ids_bytes
                        or evidence.get("selected_count") != selected_count
                        or evidence.get("ids_unique") is not True
                        or weight_bytes != selected_count * input_features * 2
                        or ids_bytes != selected_count * 8
                    ):
                        raise RuntimeError("resident compact-head persistent byte/count evidence is inconsistent")
                elif (
                    expected
                    or selected_count is not None
                    or persistent_bytes != 0
                ):
                    raise RuntimeError("unprepared compact head must remain in full BF16 mode with no selected rows")
            if key == "prose_draft_top_k":
                capability = snapshot.get("capabilities", {}).get(key, {})
                maximum = capability.get("max_value")
                if capability.get("enabled") is not True or type(maximum) is not int:
                    raise RuntimeError("resident worker does not report the proposal top-k capability")
                if expected is not None and (type(expected) is not int or not 1 <= expected <= maximum):
                    raise RuntimeError("resident proposal top-k setting must be null or an exact integer within the draft vocabulary")
            if key == "qad_hot_cache_enabled":
                if type(expected) is not bool:
                    raise RuntimeError("resident hot-cache setting must be an exact boolean")
                capability = snapshot.get("capabilities", {}).get(key, {})
                if capability.get("enabled") is not True:
                    raise RuntimeError("resident worker does not report the ready QAD HotCache capability")
            actual = _resident_control_value(snapshot, key)
            if not _same_control_value(actual, expected):
                raise RuntimeError(f"resident worker control mismatch for {key}: expected {expected!r}, observed {actual!r}")


def _zw_prebake(api: "ResidentApi", model: str) -> list[dict[str, Any]]:
    """ZW-1/ZW-4/ZW-6 boot-time pre-bake.

    Runs before the benchmark timer starts: forces every verification row
    shape M=1..6, the temp=0.7 rejection sampler kernel, a 600-token KV/GDN
    pre-touch, and a prose hot-expert seed request.
    """
    results: list[dict[str, Any]] = []

    def _one(stage: str, prompt: str, max_tokens: int, temperature: float) -> None:
        started = time.perf_counter()
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "ignore_eos": True,
        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            with api._open("/v1/completions", method="POST", data=body, timeout=300) as response:
                response.read()
            results.append({"stage": stage, "seconds": round(time.perf_counter() - started, 3), "ok": True})
        except Exception as exc:  # pre-bake is best-effort; record and continue
            results.append({
                "stage": stage,
                "seconds": round(time.perf_counter() - started, 3),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    for m in range(1, 7):
        _one(f"zw1_m{m}", f"def zw1_{m}(x):\n    return x + {m}\n", m, 0.0)
    _one("zw1_temp07", "Write a short paragraph about morning light.", 6, 0.7)
    _one("zw6_kv600", "Write a short paragraph about morning light.", 600, 0.7)
    _one("zw4_prose_hot", "Think step by step, then write a calm essay about rivers.", 256, 0.7)
    return results


def _load_ready_server(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("status") != "ready":
        current = state.get("status") if isinstance(state, dict) else "invalid"
        raise RuntimeError(f"resident server state is not ready: {current}")
    base_url = state.get("base_url")
    if not isinstance(base_url, str) or not base_url.startswith("http://127.0.0.1:"):
        raise RuntimeError("resident server state must point to container loopback only")
    api_pid = state.get("api_server_pid")
    if not isinstance(api_pid, int) or not _pid_is_alive(api_pid):
        raise RuntimeError("resident API server PID from state file is not alive")
    return state


def _execute_experiments(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    save,
    api: ResidentApi,
    engine_identity: dict[str, Any],
    baseline_settings: dict[str, Any],
    experiments: list[dict[str, Any]],
    profile: dict[str, Any],
    runs: int,
    interrupt_event: threading.Event,
    baseline_peer_controls: dict[str, bool] | None,
    baseline_peer_inventory: Any,
) -> int:
    if args.exact_input_tokens and not args.prefill_only:
        preflight_samples = []
        for workload in args.workloads:
            prompt = _decode_prompts(workload, 1)[0]
            thinking_modes = [_DEFAULT_THINKING[workload]]
            if args.prefill_sizes and False not in thinking_modes:
                thinking_modes.append(False)
            for thinking in thinking_modes:
                preflight_samples.append({
                    "workload": workload,
                    "thinking": thinking,
                    "tokenizer_prompt_tokens": api.tokenize_chat_prompt(
                        record["model_id"], prompt, thinking=thinking,
                        timeout=args.request_timeout,
                    ),
                })
        record["input_tokenization_preflight"] = {
            "status": "passed",
            "endpoint": "resident API POST /tokenize",
            "samples": preflight_samples,
            "generation_not_started": True,
        }
        save()
    record["host_memory_before_measurement"] = host_memory()
    results: list[dict[str, Any]] = []
    peer_control_mode = baseline_peer_controls is not None
    current_peer_state = dict(baseline_peer_controls) if peer_control_mode else None
    current_peer_inventory = baseline_peer_inventory
    if peer_control_mode and current_peer_inventory is None:
        raise RuntimeError("peer-control experiment has no fresh client-entry peer_inventory")

    def capture_peer_inventory(
        experiment_name: str, boundary: str, *, required: bool = False,
    ) -> Any:
        snapshot: dict[str, Any] = {
            "experiment": experiment_name,
            "boundary": boundary,
            "kind": "peer_inventory",
            "rpc_method": "peer_inventory",
        }
        try:
            result = api.worker_rpc("peer_inventory", timeout=30)
            if required and result is None:
                raise RuntimeError("peer_inventory returned no payload")
            snapshot["result"] = result
        except Exception as exc:
            snapshot["available"] = False
            snapshot["error"] = f"{type(exc).__name__}: {exc}"
            record.setdefault("worker_diagnostics", []).append(snapshot)
            save()
            if required:
                raise
            return None
        record.setdefault("worker_diagnostics", []).append(snapshot)
        save()
        return result

    def capture_worker_diagnostics(experiment_name: str, boundary: str) -> dict[str, Any]:
        requests = []
        if args.ple_diagnostics:
            requests.append(("ple_mmap", "resident_ple_mmap_diagnostics", 900))
        if args.peer_diagnostics:
            requests.append(("peer_inventory", "peer_inventory", 30))
        captured: dict[str, Any] = {}
        for kind, method, timeout in requests:
            if kind == "peer_inventory":
                captured[kind] = capture_peer_inventory(experiment_name, boundary)
                continue
            snapshot: dict[str, Any] = {
                "experiment": experiment_name,
                "boundary": boundary,
                "kind": kind,
                "rpc_method": method,
            }
            try:
                snapshot["result"] = api.worker_rpc(method, timeout=timeout)
                captured[kind] = snapshot["result"]
            except Exception as exc:
                snapshot["available"] = False
                snapshot["error"] = f"{type(exc).__name__}: {exc}"
            record.setdefault("worker_diagnostics", []).append(snapshot)
        if requests:
            save()
        return captured

    try:
        for experiment in experiments:
            if interrupt_event.is_set():
                break
            experiment_start = time.perf_counter()
            control: dict[str, Any] = {
                "experiment": experiment["name"],
                "settings_requested": experiment["settings"],
                "engine_identity": dict(engine_identity),
            }
            try:
                effective_settings = dict(baseline_settings)
                effective_settings.update(experiment["settings"])
                control["settings_effective_request"] = effective_settings
                control["configure_response"] = api.worker_rpc(
                    "resident_spec_configure", kwargs={"settings": effective_settings}
                )
                rejections = _explicit_control_rejection(control["configure_response"])
                if control["configure_response"] is None:
                    rejections.append("configure RPC returned no payload")
                if rejections:
                    raise RuntimeError("; ".join(rejections))
                control["status_response"] = api.worker_rpc("resident_spec_status")
                status_rejections = _explicit_control_rejection(control["status_response"])
                if control["status_response"] is None:
                    status_rejections.append("status RPC returned no payload")
                if status_rejections:
                    raise RuntimeError("; ".join(status_rejections))
                _verify_resident_control_settings(control["status_response"], effective_settings)
                control["status"] = "applied_or_verified_unchanged"
            except Exception as exc:
                control["status"] = "rejected_or_unavailable"
                control["failure"] = f"{type(exc).__name__}: {exc}"
                record["control_events"].append(control)
                record["failures"].append({
                    "stage": "control",
                    "experiment": experiment["name"],
                    "reason": control["failure"],
                })
                save()
                results.append({
                    "name": experiment["name"],
                    "settings": experiment["settings"],
                    "status": "rejected_or_unavailable",
                    "failure": control["failure"],
                    "engine_identity": dict(engine_identity),
                })
                continue
            record["control_events"].append(control)
            save()
            peer_event = None
            peer_target = None
            if peer_control_mode:
                requested_peer_controls = experiment["peer_controls"]
                peer_target = dict(baseline_peer_controls)
                peer_target.update(requested_peer_controls)
                peer_event = {
                    "experiment": experiment["name"],
                    "peer_controls_requested": dict(requested_peer_controls),
                    "peer_controls_effective": dict(peer_target),
                    "worker_state_before_apply": _verify_peer_control_state(
                        current_peer_inventory, current_peer_state,
                    ),
                    "setter_responses": [],
                }
                record.setdefault("peer_control_events", []).append(peer_event)
                save()
                try:
                    _set_peer_control_state(api, peer_target, response_sink=peer_event["setter_responses"])
                    current_peer_inventory = capture_peer_inventory(
                        experiment["name"], "transport-after-apply", required=True,
                    )
                    peer_event["worker_state_after_apply"] = _verify_peer_control_state(
                        current_peer_inventory, peer_target,
                    )
                    peer_event["status"] = "applied_and_verified"
                    current_peer_state = dict(peer_target)
                except Exception as exc:
                    peer_event["status"] = "rejected_or_unavailable"
                    peer_event["failure"] = f"{type(exc).__name__}: {exc}"
                    record["failures"].append({
                        "stage": "peer_control",
                        "experiment": experiment["name"],
                        "reason": peer_event["failure"],
                    })
                    save()
                    raise
                save()
            worker_diagnostics_before = capture_worker_diagnostics(experiment["name"], "before")
            if peer_event is not None:
                before_inventory = worker_diagnostics_before.get("peer_inventory")
                peer_event["worker_state_before_warmup"] = _verify_peer_control_state(
                    before_inventory, peer_target,
                )
                current_peer_inventory = before_inventory
            first_wave = len(record["waves"])
            selection_request_ms: dict[str, list[dict[str, Any]]] = {}
            measurement_before_inventory = None
            for workload in ([] if args.prefill_only else args.workloads):
                if interrupt_event.is_set():
                    break
                temperature = _DEFAULT_TEMPERATURE[workload]
                thinking = _DEFAULT_THINKING[workload]
                if args.warmup_tokens > 0:
                    warmup_prompt = (
                        mia_bench.WARMUP_PROMPTS["code"] if workload == "code"
                        else (_STRUCTURED_PROMPT if workload == "structured" else mia_bench.PROMPTS["prose"])
                    )
                    warmup_started = time.perf_counter()
                    warmup = _run_wave(
                        record=record, save=save, api=api, api_process=None,
                        engine_identity=engine_identity, workload=workload, phase="warmup",
                        prompts=[warmup_prompt], experiment_name=experiment["name"], run_index=None,
                        max_tokens=args.warmup_tokens, temperature=temperature, thinking=thinking,
                        top_p=args.top_p, top_k=args.top_k, request_timeout=args.request_timeout,
                        require_usage=False, interrupt_event=interrupt_event,
                        require_complete_stream=False, measurement_kind="decode", quality_applicable=False,
                        force_to_cap=False,
                        ignore_eos=profile["ignore_eos_requested"] if not profile["force_to_cap"] else None,
                    )
                    warmup["warmup_elapsed_ms"] = round((time.perf_counter() - warmup_started) * 1000, 6)
                    save()
                if peer_event is not None and not interrupt_event.is_set():
                    measurement_before_inventory = capture_peer_inventory(
                        experiment["name"], "measurement-before", required=True,
                    )
                    peer_event["worker_state_before_measurement"] = _verify_peer_control_state(
                        measurement_before_inventory, peer_target, require_arena_state=True,
                    )
                    current_peer_inventory = measurement_before_inventory
                    save()
                for context_tokens in args.decode_context_sizes:
                    if interrupt_event.is_set():
                        break
                    for concurrency in args.concurrencies:
                        if interrupt_event.is_set():
                            break
                        base_prompts = _decode_prompts(workload, concurrency)
                        prompt_tokenization = None
                        if args.exact_input_tokens:
                            prepared_prompts = [
                                _build_exact_workload_context_prompt(
                                    api, record["model_id"], prompt, context_tokens, str(uuid.uuid4()),
                                    thinking=thinking, request_timeout=args.request_timeout,
                                )
                                for prompt in base_prompts
                            ]
                            prompts = [prepared[0] for prepared in prepared_prompts]
                            prompt_tokenization = [prepared[1] for prepared in prepared_prompts]
                        else:
                            prompts = [
                                _build_workload_context_prompt(prompt, context_tokens, str(uuid.uuid4()))
                                for prompt in base_prompts
                            ]
                        for run_index in range(1, runs + 1):
                            if interrupt_event.is_set():
                                break
                            requested_prompt_tokens = context_tokens if context_tokens > 0 else None
                            wave = _run_wave(
                                record=record, save=save, api=api, api_process=None,
                                engine_identity=engine_identity, workload=workload, phase="measurement",
                                prompts=prompts, experiment_name=experiment["name"], run_index=run_index,
                                max_tokens=profile["caps"][workload], temperature=temperature, thinking=thinking,
                                top_p=args.top_p, top_k=args.top_k, request_timeout=args.request_timeout,
                                require_usage=True, interrupt_event=interrupt_event,
                                require_complete_stream=True, quality_applicable=True,
                                measurement_kind="decode", requested_context_tokens=context_tokens,
                                requested_prompt_tokens=requested_prompt_tokens,
                                target_prompt_tokens=requested_prompt_tokens,
                                prompt_tokenization=prompt_tokenization,
                                force_to_cap=profile["force_to_cap"],
                                ignore_eos=profile["ignore_eos_requested"],
                            )
                            if args.protocol_profile == "screen":
                                selection_key = f"context={context_tokens};workload={workload};concurrency={concurrency}"
                                selection_request_ms.setdefault(selection_key, []).append({
                                    "run_index": run_index,
                                    "wave_elapsed_ms": wave["metrics"]["wave_elapsed_ms"],
                                    "per_request_e2e_ms": [event.get("e2e_ms") for event in wave["events"]],
                                    "per_request_ttft_ms": [event.get("ttft_ms") for event in wave["events"]],
                                    "throughput_eligible": wave["metrics"]["throughput_eligible"],
                                })
            if args.prefill_sizes and not interrupt_event.is_set():
                for warmup_tokens in (512, 4096):
                    if interrupt_event.is_set():
                        break
                    _run_wave(
                        record=record, save=save, api=api, api_process=None,
                        engine_identity=engine_identity, workload="prefill", phase="prefill-warmup",
                        prompts=[_build_prefill_prompt(warmup_tokens, str(uuid.uuid4()))],
                        experiment_name=experiment["name"], run_index=None,
                        max_tokens=8, temperature=0.0, thinking=False, top_p=1.0, top_k=None,
                        request_timeout=args.request_timeout, require_usage=False,
                        interrupt_event=interrupt_event, require_prompt_usage=True,
                        require_complete_stream=False, measurement_kind="prefill",
                        requested_context_tokens=warmup_tokens,
                        requested_prompt_tokens=warmup_tokens, target_prompt_tokens=warmup_tokens,
                        quality_applicable=False, force_to_cap=False,
                    )
                calibration_ttfts: list[float] = []
                calibration_prompt_tokens: list[int | None] = []
                for calibration_index in range(1, 4):
                    if interrupt_event.is_set():
                        break
                    calibration = _run_wave(
                        record=record, save=save, api=api, api_process=None,
                        engine_identity=engine_identity, workload="prefill", phase="prefill-calibration",
                        prompts=[_build_prefill_prompt(16, str(uuid.uuid4()))],
                        experiment_name=experiment["name"], run_index=calibration_index,
                        max_tokens=8, temperature=0.0, thinking=False, top_p=1.0, top_k=None,
                        request_timeout=args.request_timeout, require_usage=False,
                        interrupt_event=interrupt_event, require_prompt_usage=True,
                        require_complete_stream=True, measurement_kind="prefill",
                        requested_context_tokens=16, requested_prompt_tokens=16,
                        target_prompt_tokens=16, quality_applicable=False, force_to_cap=False,
                    )
                    for event in calibration["events"]:
                        calibration_prompt_tokens.append(event.get("actual_usage_prompt_tokens"))
                        ttft = event.get("ttft_ms")
                        if event.get("throughput_eligible") and isinstance(ttft, (int, float)) and ttft > 0:
                            calibration_ttfts.append(float(ttft))
                overhead_ms = statistics.median(calibration_ttfts) if calibration_ttfts else None
                record.setdefault("prefill_overhead_calibration", []).append({
                    "experiment": experiment["name"],
                    "workload_scope": "shared fixed calibration prompt; workload-specific measurement prompts retain their task tail",
                    "requested_context_tokens": 16,
                    "requested_samples": 3,
                    "successful_samples": len(calibration_ttfts),
                    "median_ttft_ms": overhead_ms,
                    "actual_usage_prompt_tokens_by_sample": calibration_prompt_tokens,
                    "missing_reason": None if calibration_ttfts else "no successful calibration sample",
                    "ttft_fallback_rate_requires_calibration": True,
                })
                save()
                for workload in args.workloads:
                    if interrupt_event.is_set():
                        break
                    for size in args.prefill_sizes:
                        repeats_for_size = args.prefill_repeats or _prefill_repeats(size)
                        for concurrency in args.prefill_concurrencies:
                            base_prompts = _decode_prompts(workload, concurrency)
                            for prefill_run in range(1, repeats_for_size + 1):
                                if interrupt_event.is_set():
                                    break
                                prompt_tokenization = None
                                if args.exact_input_tokens:
                                    prepared_prompts = [
                                        _build_exact_prefill_context_prompt(
                                            api, record["model_id"], prompt, size, str(uuid.uuid4()),
                                            request_timeout=args.request_timeout,
                                        )
                                        for prompt in base_prompts
                                    ]
                                    prompts = [prepared[0] for prepared in prepared_prompts]
                                    prompt_tokenization = [prepared[1] for prepared in prepared_prompts]
                                else:
                                    prompts = [
                                        _build_workload_context_prompt(prompt, size, str(uuid.uuid4()))
                                        for prompt in base_prompts
                                    ]
                                _run_wave(
                                    record=record, save=save, api=api, api_process=None,
                                    engine_identity=engine_identity, workload=workload,
                                    phase="prefill-measurement", prompts=prompts,
                                    experiment_name=experiment["name"], run_index=prefill_run,
                                    max_tokens=8, temperature=0.0, thinking=False, top_p=1.0, top_k=None,
                                    request_timeout=args.request_timeout, require_usage=False,
                                    interrupt_event=interrupt_event, require_prompt_usage=True,
                                    require_complete_stream=True, measurement_kind="prefill",
                                    requested_context_tokens=size, requested_prompt_tokens=size,
                                    prefill_overhead_ms=overhead_ms, target_prompt_tokens=size,
                                    prompt_tokenization=prompt_tokenization,
                                    quality_applicable=False, force_to_cap=False,
                                )
            worker_diagnostics_after = capture_worker_diagnostics(experiment["name"], "after")
            if peer_event is not None:
                after_inventory = worker_diagnostics_after.get("peer_inventory")
                if after_inventory is None:
                    raise RuntimeError("peer-control measurement has no post-request peer_inventory")
                peer_event["worker_state_after_measurement"] = _verify_peer_control_state(
                    after_inventory, peer_target, require_arena_state=True,
                )
                if measurement_before_inventory is not None:
                    peer_event["measurement_counter_delta"] = _peer_counter_deltas(
                        measurement_before_inventory, after_inventory,
                    )
                    peer_event["measurement_state_status"] = "verified"
                else:
                    peer_event["measurement_state_status"] = "no_measurement_boundary"
                current_peer_inventory = after_inventory
                current_peer_state = dict(peer_target)
                save()
            experiment_waves = record["waves"][first_wave:]
            elapsed_ms = round((time.perf_counter() - experiment_start) * 1000, 6)
            summary = {
                "name": experiment["name"],
                "settings": experiment["settings"],
                "status": "interrupted" if interrupt_event.is_set() else "completed",
                "experiment_elapsed_ms_including_warmup_and_measurements": elapsed_ms,
                "screen_selection_elapsed_ms_excluding_engine_load": elapsed_ms
                if args.protocol_profile == "screen" else None,
                "screen_selection_request_elapsed_ms_by_context_workload_and_concurrency": selection_request_ms,
                "decode_waves": sum(wave["phase"] == "measurement" for wave in experiment_waves),
                "prefill_waves": sum(wave["phase"] == "prefill-measurement" for wave in experiment_waves),
                "engine_identity": dict(engine_identity),
            }
            if peer_event is not None:
                summary.update({
                    "peer_controls_requested": peer_event["peer_controls_requested"],
                    "peer_controls_effective": peer_event["peer_controls_effective"],
                    "peer_control_status": peer_event.get("status"),
                    "peer_measurement_counter_delta": peer_event.get("measurement_counter_delta"),
                })
            results.append(summary)
            record["selection_elapsed_ms"][experiment["name"]] = summary[
                "screen_selection_elapsed_ms_excluding_engine_load"
            ]
            save()
            if interrupt_event.is_set():
                break
    except BaseException as exc:
        record["failures"].append({
            "stage": "experiment_execution",
            "reason": f"{type(exc).__name__}: {exc}",
        })
        raise
    finally:
        restoration: dict[str, Any] = {
            "experiment": "restore-client-entry-baseline",
            "settings_requested": dict(baseline_settings),
            "settings_effective_request": dict(baseline_settings),
            "source": "fresh live worker control state captured at client entry",
            "engine_identity": dict(engine_identity),
        }
        try:
            restoration["configure_response"] = api.worker_rpc(
                "resident_spec_configure", kwargs={"settings": baseline_settings}
            )
            rejection = _explicit_control_rejection(restoration["configure_response"])
            if restoration["configure_response"] is None or rejection:
                raise RuntimeError("; ".join(rejection or ["baseline restore configure RPC returned no payload"]))
            restoration["status_response"] = api.worker_rpc("resident_spec_status")
            rejection = _explicit_control_rejection(restoration["status_response"])
            if restoration["status_response"] is None or rejection:
                raise RuntimeError("; ".join(rejection or ["baseline restore status RPC returned no payload"]))
            _verify_resident_control_settings(restoration["status_response"], baseline_settings)
            restoration["status"] = "restored_and_verified"
        except Exception as exc:
            restoration["status"] = "restore_failed"
            restoration["failure"] = f"{type(exc).__name__}: {exc}"
            record["failures"].append({
                "stage": "control_restore",
                "reason": restoration["failure"],
            })
        record["control_events"].append(restoration)
        record["runtime_control_restoration"] = restoration
        save()
    record["experiments"] = results
    record["host_memory_after_measurement"] = host_memory()
    memory_before = record["host_memory_before_measurement"]
    memory_after = record["host_memory_after_measurement"]
    record["swap_out_pages_during_measurement"] = memory_after["pswpout"] - memory_before["pswpout"]
    record["swap_in_pages_during_measurement"] = memory_after["pswpin"] - memory_before["pswpin"]
    record["host_memory_window_scope"] = (
        "System-wide counters from after model load to after all experiments; not process-exclusive attribution"
    )
    if interrupt_event.is_set():
        record["status"] = "interrupted"
    else:
        has_failures = bool(record["failures"]) or any(not event.get("success") for event in record["events"])
        record["status"] = "completed_with_failures" if has_failures else "completed"
    record["finished_utc"] = _utc_now()
    save()
    return 130 if interrupt_event.is_set() else (1 if record["status"] == "completed_with_failures" else 0)


def _serve_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        profile = _profile(args)
        max_model_len, max_num_seqs = _server_dimensions(args, profile["caps"])
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    state_path = Path(args.state_file)
    out = Path(args.out)
    base_url = f"http://127.0.0.1:{args.port}"
    engine_id = str(uuid.uuid4())
    server_state: dict[str, Any] = {
        "schema": "mia-resident-server-v1",
        "status": "starting",
        "engine_instance_id": engine_id,
        "base_url": base_url,
        "model_id": args.served_model_name,
        "model_path": args.model,
        "draft_path": args.draft,
        "mtp": args.mtp,
        "draft_sample_method": args.draft_sample_method,
        "rejection_sample_method": args.rejection_sampling,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv_cache_memory_bytes": None if args.auto_kv else args.kv_mib * 1024**2,
        "kv_allocation": "actual planner" if args.auto_kv else "fixed bytes",
        "gpu_memory_utilization": args.auto_kv_memory_utilization if args.auto_kv else 0.01,
        "decode_context_sizes": list(args.decode_context_sizes),
        "prefill_sizes": list(args.prefill_sizes),
        "effective_top_k": 20,
        "override_generation_config": {"top_k": 20},
        "async_scheduling": False,
        "worker_extension_cls": _WORKER_EXTENSION,
        "network_scope": "127.0.0.1 inside the network-disabled benchmark container",
        "started_utc": _utc_now(),
    }
    _save_json(state_path, server_state)
    interrupt_event = threading.Event()
    previous_handlers: dict[int, Any] = {}
    process: subprocess.Popen | None = None

    def request_stop(signum, frame) -> None:
        interrupt_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    started = time.perf_counter()
    try:
        command, compile_config, speculative_config = build_api_server_command(
            model=args.model,
            draft=args.draft,
            mtp=args.mtp,
            graphs=args.graphs,
            single_gb10=args.single_gb10,
            kv_mib=None if args.auto_kv else args.kv_mib,
            gpu_memory_utilization=args.auto_kv_memory_utilization if args.auto_kv else 0.01,
            draft_sample_method=args.draft_sample_method,
            rejection_sample_method=args.rejection_sampling,
            block_size=args.block_size,
            worker_extension_cls=_WORKER_EXTENSION,
            served_model_name=args.served_model_name,
            host="127.0.0.1",
            port=args.port,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            override_generation_config={"top_k": 20},
            async_scheduling=False,
            reasoning_parser=args.reasoning_parser,
        )
        server_env = os.environ.copy()
        server_env["VLLM_SERVER_DEV_MODE"] = "1"
        server_env["QWEN38_RESIDENT_WORKER_PROCESS"] = "1"
        server_env["QWEN38_SINGLE_GB10"] = "1" if args.single_gb10 else "0"
        server_env["QWEN38_RESIDENT_ENGINE_ID"] = engine_id
        log_path = out.with_suffix(".server.log")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parent,
                env=server_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            server_state.update({
                "api_server_pid": process.pid,
                "command": command,
                "compile_config": compile_config,
                "speculative_config": speculative_config,
                "log_path": str(log_path),
            })
            _save_json(state_path, server_state)
            api = ResidentApi(base_url, os.environ.get("OPENAI_API_KEY") or None)
            deadline = time.monotonic() + args.server_start_timeout
            while True:
                if interrupt_event.is_set():
                    raise KeyboardInterrupt
                if process.poll() is not None:
                    raise RuntimeError(f"resident vLLM API server exited before readiness: {process.returncode}")
                try:
                    if api.health():
                        break
                except (HTTPError, URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("resident vLLM API readiness timed out")
                time.sleep(1)
            worker_identity = api.worker_rpc("resident_engine_identity")
            worker_pids = _extract_worker_pids(worker_identity)
            engine_ids = _extract_engine_ids(worker_identity)
            if not worker_pids or engine_ids != [engine_id]:
                raise RuntimeError(
                    f"resident worker identity mismatch: expected engine {engine_id}, got {engine_ids} with PIDs {worker_pids}"
                )
            status_response = api.worker_rpc("resident_spec_status")
            server_state["worker_identity_rpc_result"] = worker_identity
            server_state["worker_status_before_readiness_validation"] = status_response
            _save_json(state_path, server_state)
            if status_response is None:
                raise RuntimeError("resident_spec_status returned no payload")
            status_rejections = _explicit_control_rejection(status_response)
            if status_rejections:
                raise RuntimeError("; ".join(status_rejections))
            baseline_settings = _resident_baseline_settings(status_response)
            server_state.update({
                "status": "ready",
                "ready_utc": _utc_now(),
                "engine_load_seconds": round(time.perf_counter() - started, 6),
                "worker_pids": worker_pids,
                "worker_identity_rpc_result": worker_identity,
                "worker_status_at_ready": status_response,
                "initial_control_settings": baseline_settings,
            })
            _save_json(state_path, server_state)
            _save_json(out, server_state)
            if os.environ.get("QWEN38_ZW_PREBAKE") == "1":
                prebake_started = time.perf_counter()
                prebake_results = _zw_prebake(api, args.served_model_name)
                server_state["zw_prebake"] = {
                    "seconds": round(time.perf_counter() - prebake_started, 3),
                    "results": prebake_results,
                }
                _save_json(state_path, server_state)
                _save_json(out, server_state)
            print(json.dumps({
                "status": "serving",
                "state_file": str(state_path),
                "api_server_pid": process.pid,
                "engine_instance_id": engine_id,
                "worker_pids": worker_pids,
            }, ensure_ascii=False), flush=True)
            while not interrupt_event.wait(1):
                if process.poll() is not None:
                    raise RuntimeError(f"resident vLLM API server exited while serving: {process.returncode}")
            server_state["status"] = "stopping"
            _save_json(state_path, server_state)
    except KeyboardInterrupt:
        server_state["status"] = "stopping"
    except Exception as exc:
        server_state["status"] = "failed"
        server_state["failure"] = f"{type(exc).__name__}: {exc}"
        _save_json(out, server_state)
        return 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if process is not None:
            server_state["exit_status"] = process.returncode
        if server_state["status"] != "failed":
            server_state["status"] = "stopped"
        server_state["finished_utc"] = _utc_now()
        _save_json(state_path, server_state)
        _save_json(out, server_state)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 130 if interrupt_event.is_set() else 0


def _client_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    client_started_perf_counter = time.perf_counter()
    try:
        server_state = _load_ready_server(Path(args.state_file))
        experiments = _parse_experiments(args.experiments)
        profile = _profile(args)
        if args.campaign_index is not None:
            resident_campaign.validate_campaign_index(args.campaign_index)
    except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc:
        parser.error(str(exc))
    runs = args.runs if args.runs is not None else (
        3 if args.protocol_profile in ("final", "natural-goal", "official-headline") else 1
    )
    peer_control_mode = any("peer_controls" in item for item in experiments)
    peer_control_candidate: dict[str, bool] | None = None
    if peer_control_mode:
        candidates = [item["peer_controls"] for item in experiments if item["peer_controls"]]
        if len(candidates) == 0 or any(candidate != candidates[0] for candidate in candidates[1:]):
            parser.error("peer-control experiment set must contain one identical B candidate")
        peer_control_candidate = dict(candidates[0])
        if not args.peer_diagnostics:
            parser.error("peer-control experiments require --peer-diagnostics")
        if (
            args.protocol_profile not in ("screen", "natural-goal", "official-headline")
            or args.workloads not in (["code"], ["prose"])
            or runs != 1
            or args.warmup_tokens != 32
            or args.prefill_only
            or args.prefill_sizes
            or args.exact_input_tokens
            or args.decode_context_sizes != [0]
            or args.concurrencies != [1]
            or args.output_cap
            or args.top_p != 1.0
            or args.top_k is not None
        ):
            parser.error(
                "peer-control experiments require one code/prose workload, one run, "
                "32-token warmup, default request policy, and no context/prefill sweep"
            )
    server_max_model_len = int(server_state["max_model_len"])
    server_max_num_seqs = int(server_state["max_num_seqs"])
    maximum_requested_context = max(args.prefill_sizes + args.decode_context_sizes, default=0)
    needed_model_len = maximum_requested_context + max(max(profile["caps"].values()), 8)
    if server_max_model_len < needed_model_len:
        parser.error(
            f"resident server max_model_len={server_max_model_len} cannot cover requested prompt/reply capacity {needed_model_len}"
        )
    requested_concurrency = max(args.concurrencies + args.prefill_concurrencies)
    if server_max_num_seqs < requested_concurrency:
        parser.error(
            f"resident server max_num_seqs={server_max_num_seqs} is below requested concurrency {requested_concurrency}"
        )
    base_url = server_state["base_url"]
    api_key = os.environ.get("OPENAI_API_KEY") or None
    api = ResidentApi(base_url, api_key)
    peer_control_baseline_inventory = None
    peer_control_baseline_snapshot = None
    peer_control_baseline = None
    try:
        if not api.health():
            parser.error("resident API health check did not return HTTP 200")
        engine_identity = _engine_identity(api, server_state)
        baseline_status = api.worker_rpc("resident_spec_status")
        if not baseline_status:
            raise RuntimeError("resident worker returned no live control status")
        baseline_rejections = _explicit_control_rejection(baseline_status)
        if baseline_rejections:
            raise RuntimeError("; ".join(baseline_rejections))
        baseline_settings = _resident_baseline_settings(baseline_status)
        _verify_resident_control_settings(baseline_status, baseline_settings)
        _add_supported_draft_head_metadata(profile, baseline_status)
        _add_supported_compact_draft_head_metadata(profile, baseline_status)
        _add_supported_draft_scale_per_pos_metadata(profile, baseline_status)
        if "code_draft_tokens" in baseline_settings:
            profile["metadata"]["resident_spec_control"]["supported_live_settings"]["code_draft_tokens"] = (
                "exact integer 1..Kmax; initialized once from QWEN38_CODE_DRAFT_TOKENS "
                "or Kmax, then configurable between requests for nonempty all-zero-temperature "
                "batches only; engine buffers, prose, mixed and empty batches are unchanged"
            )
        if peer_control_mode:
            peer_control_baseline_inventory = api.worker_rpc("peer_inventory")
            peer_control_baseline_snapshot = _peer_control_snapshot(peer_control_baseline_inventory)
            peer_control_baseline = peer_control_baseline_snapshot["state"]
            _require_peer_control_candidate_change(
                peer_control_candidate,
                peer_control_baseline,
            )
        if any("draft_scale_per_pos" in item["settings"] for item in experiments):
            supported, _, reason = _draft_scale_per_pos_capability(baseline_status)
            if not supported:
                parser.error(
                    "draft_scale_per_pos was requested but the ready worker source does not support it: "
                    + str(reason or "capability unavailable")
                )
    except Exception as exc:
        parser.error(f"resident API/worker identity check failed: {type(exc).__name__}: {exc}")
    out = Path(args.out)
    record: dict[str, Any] = {
        "schema": "mia-resident-benchmark-v1",
        "protocol_id": profile["protocol_id"],
        "base_profile_id": profile["metadata"]["base_profile_id"],
        "protocol_profile": args.protocol_profile,
        "campaign_id": args.campaign_id,
        "campaign_model_id": args.campaign_model_id,
        "campaign_stage": args.campaign_stage,
        "campaign_run_id": args.campaign_run_id,
        "campaign_protocol_class": profile["metadata"].get("protocol_class"),
        "protocol_metadata": _protocol_metadata(args, profile, runs),
        "status": "running",
        "started_utc": _utc_now(),
        "model_id": server_state["model_id"],
        "base_url": base_url,
        "endpoint": f"{base_url}/v1/chat/completions",
        "server": dict(server_state),
        "engine_identity": engine_identity,
        "server_ready_control_settings": server_state.get("initial_control_settings"),
        "server_ready_control_status": server_state.get("worker_status_at_ready"),
        "resident_control_baseline": baseline_settings,
        "resident_control_baseline_status": baseline_status,
        "resident_control_baseline_source": (
            "fresh live worker status at client entry; may differ from server-ready settings"
        ),
        "engine_load_seconds": server_state.get("engine_load_seconds"),
        "engine_load_scope": "loaded once by resident serve process; excluded from client selection and measurement windows",
        "requested": {
            "runs_per_workload_concurrency": runs,
            "max_tokens_by_workload": profile["caps"],
            "warmup_tokens": args.warmup_tokens,
            "prefill_only": args.prefill_only,
            "workloads": list(args.workloads),
            "concurrencies": list(args.concurrencies),
            "decode_context_sizes": list(args.decode_context_sizes),
            "exact_input_tokens": args.exact_input_tokens,
            "input_target_semantics": (
                _EXACT_INPUT_TARGET_SEMANTICS if args.exact_input_tokens
                else _LEGACY_INPUT_TARGET_SEMANTICS
            ),
            "prefill_sizes": list(args.prefill_sizes),
            "prefill_concurrencies": list(args.prefill_concurrencies),
            "prefill_repeats": args.prefill_repeats,
            "ple_diagnostics": args.ple_diagnostics,
            "peer_diagnostics": args.peer_diagnostics,
            "top_p": args.top_p,
            "top_k": args.top_k if args.top_k is not None else "omitted",
            "experiments": experiments,
            "natural_eos": not profile["force_to_cap"],
            "ignore_eos": profile["ignore_eos_requested"] if profile["ignore_eos_requested"] is not None else "omitted",
            "min_tokens": "max_tokens" if profile["force_to_cap"] else "omitted",
            "stop": [] if profile["force_to_cap"] else "omitted",
            "temperature_by_workload": dict(_DEFAULT_TEMPERATURE),
            "thinking_by_workload": dict(_DEFAULT_THINKING),
            "force_to_cap": profile["force_to_cap"],
            "server_restarts_between_experiments": False,
            "api_development_rpc": "VLLM_SERVER_DEV_MODE=1, loopback-only, method allowlist",
            "authorization": "bearer_from_OPENAI_API_KEY" if api_key else "none",
        },
        "selection_elapsed_ms": {},
        "events": [],
        "worker_diagnostics": [],
        "waves": [],
        "control_events": [],
        "failures": [],
        "summaries": {"decode": [], "prefill": [], "quality": []},
        "next_sequence_index": 1,
    }
    if peer_control_mode:
        record["peer_control_baseline"] = {
            "source": "fresh peer_inventory at client entry",
            "state": peer_control_baseline,
            "worker_state": peer_control_baseline_snapshot,
        }
        record["peer_control_candidate"] = dict(peer_control_candidate)
        record["peer_control_events"] = []
        record["peer_control_restoration"] = None
        record["worker_diagnostics"].append({
            "experiment": "client-entry",
            "boundary": "before",
            "kind": "peer_inventory",
            "rpc_method": "peer_inventory",
            "result": peer_control_baseline_inventory,
        })

    def save() -> None:
        record["summaries"] = _summaries(record)
        if args.protocol_profile == "screen":
            record["fast_pass"] = _fast_pass_snapshot(
                record, args, profile, experiments, runs,
                (time.perf_counter() - client_started_perf_counter) * 1000,
            )
        _save_json(out, record)
        if args.campaign_index is not None:
            resident_campaign.update_campaign_index(
                index_path=args.campaign_index,
                artifact_path=out,
                record=record,
                campaign_id=args.campaign_id,
                stage=args.campaign_stage,
                run_id=args.campaign_run_id,
            )

    interrupt_event = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum, frame) -> None:
        interrupt_event.set()
        record["interrupt_signal"] = signum
        record["interrupt_received_utc"] = _utc_now()
        record["interrupt_elapsed_ms"] = round(
            (time.perf_counter() - client_started_perf_counter) * 1000, 6
        )
        record["interrupt_completed_stream_count"] = sum(
            event.get("phase") == "measurement"
            and event.get("measurement_kind") == "decode"
            and event.get("sse_done_seen") is True
            and event.get("finish_reason") is not None
            for event in record.get("events", [])
        )
        save()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    save()
    try:
        try:
            status = _execute_experiments(
                args=args,
                record=record,
                save=save,
                api=api,
                engine_identity=engine_identity,
                baseline_settings=baseline_settings,
                experiments=experiments,
                profile=profile,
                runs=runs,
                interrupt_event=interrupt_event,
                baseline_peer_controls=peer_control_baseline,
                baseline_peer_inventory=peer_control_baseline_inventory,
            )
        finally:
            if peer_control_mode:
                peer_restoration = _restore_peer_control_state(api, peer_control_baseline)
                record["peer_control_restoration"] = peer_restoration
                if peer_restoration.get("status") != "restored_and_verified":
                    record["failures"].append({
                        "stage": "peer_control_restore",
                        "reason": peer_restoration.get("failure", "peer-control restoration failed"),
                    })
                    if record.get("status") in ("completed", "completed_with_failures"):
                        record["status"] = "completed_with_failures"
                    record["finished_utc"] = _utc_now()
                save()
        if peer_control_mode and record["peer_control_restoration"].get("status") != "restored_and_verified":
            status = 1
        print(json.dumps({
            "status": record["status"],
            "out": str(out),
            "engine_instance_id": engine_identity["engine_instance_id"],
            "summaries": record["summaries"],
        }, ensure_ascii=False), flush=True)
        return status
    except Exception as exc:
        record["status"] = "failed"
        record["failures"].append({"stage": "execution", "reason": f"{type(exc).__name__}: {exc}"})
        record["finished_utc"] = _utc_now()
        save()
        print(json.dumps({"status": "failed", "out": str(out), "failure": record["failures"][-1]}, ensure_ascii=False), flush=True)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.mode == "serve":
        return _serve_main(args, parser)
    return _client_main(args, parser)




if __name__ == "__main__":
    raise SystemExit(main())

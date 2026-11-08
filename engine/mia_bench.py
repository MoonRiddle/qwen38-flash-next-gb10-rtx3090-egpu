#!/usr/bin/env python3
"""Single-request OpenAI-compatible SSE benchmark for the pinned sparkDash tasks."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_SPARKDASH_COMMIT = "693fa8016ea3a24b29fe36a98367669fd316393b"
_SPARKDASH_BASE = f"https://github.com/MiaAI-Lab/sparkDash/tree/{_SPARKDASH_COMMIT}"

_CODE_TAIL = (
    "Output only Python source. No comments, no docstrings, no markdown fences. "
    "Then add tests and the helpers this needs. Keep writing code."
)

# Exact C1 prompt returned by pickDecodeBenchPrompts(1, "code") at the pinned commit.
PROMPTS: dict[str, str] = {
    "code": (
        "binary_search\n"
        "def binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n"
        f"{_CODE_TAIL}"
    ),
    # Exact DECODE_PROSE_PROMPT returned by pickDecodeBenchPrompts(1, "prose").
    "prose": (
        "Write a detailed step-by-step explanation of how a hash map works, "
        "including collision handling, resizing, and time complexity. Be thorough."
    ),
    # myllmbox-style Slot 2: TypeScript module plus a JSON Schema, thinking OFF.
    "code_ts": (
        "Write a TypeScript module that implements binary search over a sorted number array, "
        "returning the index of the target or -1. Export the function and include a small test. "
        "Then output a JSON Schema for a configuration object with fields name (string), "
        "maxItems (integer), and enabled (boolean). Separate the TypeScript source and the "
        "JSON Schema with a line containing exactly ---. Output only the two parts, no "
        "markdown fences, no commentary."
    ),
    # myllmbox-style Slot 2 structured half: a JSON Schema document, thinking OFF.
    "json_schema": (
        "Output only a JSON Schema document that validates a configuration object with "
        "fields name (string, required), maxItems (integer, minimum 1, required), and "
        "enabled (boolean, required). Use JSON Schema draft 2020-12. No markdown fences, "
        "no commentary."
    ),
    # myllmbox-style Slot 4: pure narrative prose, thinking OFF.
    "prose_pure": (
        "Write a short story about a lighthouse keeper who finds a message in a bottle. "
        "Use vivid sensory detail and a clear narrative arc. Output only the story, no "
        "title, no commentary."
    ),
}
PROMPTS["prose_pure_greedy"] = PROMPTS["prose_pure"]

# DecodeBench uses this separate code JIT warmup, but reuses the selected prose
# prompt for prose warmup; both are sent with the configured short token cap.
WARMUP_PROMPTS: dict[str, str] = {
    "code": (
        "warmup_noop\n"
        "def warmup_noop(x): return x unchanged.\n"
        f"{_CODE_TAIL}"
    ),
    "prose": PROMPTS["prose"],
    "code_ts": (
        "warmup_noop\n"
        "export function warmupNoop(x: number): number { return x; }\n"
        "Output only TypeScript source. No commentary."
    ),
    "json_schema": (
        "Output only a JSON Schema document for an object with one string field named id. "
        "No markdown fences, no commentary."
    ),
    "prose_pure": PROMPTS["prose_pure"],
    "prose_pure_greedy": PROMPTS["prose_pure"],
}

# These are this offline profile's explicit temperatures, not the pinned
# DecodeBench defaults and not evidence of the settings used by prior results.
TEMPERATURES: dict[str, float] = {
    "code": 0.0,
    "prose": 0.7,
    "code_ts": 0.0,
    "json_schema": 0.0,
    "prose_pure": 0.7,
    "prose_pure_greedy": 0.0,
}
THINKING_BY_WORKLOAD: dict[str, bool] = {
    "code": False,
    "prose": True,
    "code_ts": False,
    "json_schema": False,
    "prose_pure": False,
    "prose_pure_greedy": False,
}
PROTOCOL_ID = "mia-lab-v2"

RESIDENT_PROTOCOL_ID = "mia-lab-resident-v1"
RESIDENT_PROTOCOL_METADATA: dict[str, Any] = {
    "schema": "mia-http-sse-resident-v1",
    "official_reference": {
        "sparkdash_commit": _SPARKDASH_COMMIT,
        "recipe_commit": "fd3f43e11238553685ebb68b6d8ab2b112818e3c",
        "decodebench_defaults": {
            "max_tokens": 400,
            "warmup_max_tokens": 32,
            "top_p": 1.0,
            "top_k": "omitted",
            "natural_eos": "HTTP 400 strips force-cap fields and retries; otherwise EOS is ignored through the cap",
            "forced_min_tokens": True,
            "ignore_eos": True,
            "stop": [],
            "fallback_on_http_400": "strip min_tokens, ignore_eos, and stop after thinking-field retries; retry once",
            "thinking_http_400_fallback": {
                "thinking_on": "strip thinking flags once and retry",
                "thinking_off": "explicit disable fallback, then minimal enable_thinking=false fallback",
                "source": "pinned LlmStreaming.runStreamingRequest",
            },
            "recipe_server_sampling_defaults": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "source": "pinned TensorFold recipe scripts/config.sh; request values override server defaults",
            },
            "pinned_request_top_k": "omitted; resident server override sets the effective value to 20 without sending top_k in the request",
            "vllm_sampling_default_note": "the resident server explicitly overrides generation-config top_k to 20; its underlying default is superseded",
        },
        "decode_tps": "(usage.completion_tokens - 1) / (last visible reasoning-or-answer timestamp - first visible reasoning-or-answer timestamp)",
        "prefill": {
            "request_max_tokens": 8,
            "temperature": 0.0,
            "thinking": False,
            "warmup_context_tokens": [512, 4096],
            "fixed_overhead_samples_at_context_tokens": [16, 16, 16],
            "default_context_tokens": [4096, 8192, 16384, 32768],
            "repeats_through_32768": 3,
        },
    },
    "requested_resident_profile": {
        "protocol_id": RESIDENT_PROTOCOL_ID,
        "temperature_by_workload": {"code": 0.0, "prose": 0.7, "structured": 0.7},
        "thinking_by_workload": {"code": False, "prose": True, "structured": True},
        "top_p": 1.0,
        "top_k": "omitted unless explicitly supplied",
        "warmup_max_tokens": 32,
        "formal_fixed400": {
            "protocol_id": "mia-lab-sparkdash-headline400-v1",
            "max_tokens_by_workload": {"code": 400, "prose": 400},
            "force_to_cap": True,
            "min_tokens": "max_tokens",
            "ignore_eos": True,
            "stop": [],
            "source": "Pinned MiaAI-Lab sparkDash DecodeBench.js at 693fa8016ea3a24b29fe36a98367669fd316393b; HTTP 400 fallback strips force-cap fields and retries once.",
        },
        "natural_goal": {
            "protocol_id": "mia-lab-resident-goal-code256-prose600-natural-v1",
            "max_tokens_by_workload": {"code": 256, "prose": 600},
            "ignore_eos": False,
            "min_tokens": "omitted",
            "stop": "omitted",
            "purpose": "Requested natural-termination goal series; never merged with fixed400 or diagnostic screen/profile records.",
        },
        "diagnostic_profiles": {
            "screen": "mia-lab-resident-screen256-natural-v2",
            "legacy_final": "mia-lab-resident-final600-natural-v2",
        },
        "note": "Mia pinned DecodeBench defaults and the requested natural-goal series have different output caps and EOS contracts; their records remain separate.",
    },
}
RESIDENT_PROTOCOLS: dict[str, dict[str, Any]] = {
    "screen": {
        "protocol_id": "mia-lab-resident-screen256-natural-v2",
        "max_tokens_by_workload": {"code": 256, "prose": 256, "structured": 256, "prose_pure": 256},
        "force_to_cap": False,
        "explicit_natural_ignore_eos": False,
        "protocol_class": "diagnostic",
        "source_condition": "Diagnostic 256-token cap with explicit ignore_eos=false and natural EOS; not a formal result.",
    },
    "final": {
        "protocol_id": "mia-lab-resident-final600-natural-v2",
        "max_tokens_by_workload": {"code": 600, "prose": 600, "structured": 600, "prose_pure": 600},
        "force_to_cap": False,
        "explicit_natural_ignore_eos": False,
        "protocol_class": "diagnostic",
        "source_condition": "Legacy natural-EOS 600-token profile retained only for the existing separately identified series; not the requested natural-goal or pinned fixed400 contract.",
    },
    "natural-goal": {
        "protocol_id": "mia-lab-resident-goal-code256-prose600-natural-v1",
        "max_tokens_by_workload": {"code": 256, "prose": 600, "structured": 600, "prose_pure": 600},
        "force_to_cap": False,
        "explicit_natural_ignore_eos": False,
        "protocol_class": "natural-goal",
        "source_condition": "Natural EOS with explicit ignore_eos=false, code cap 256 and prose cap 600; separate from formal fixed400 and all diagnostics.",
    },
    "official-headline": {
        "protocol_id": "mia-lab-sparkdash-headline400-v1",
        "max_tokens_by_workload": {"code": 400, "prose": 400, "structured": 400, "prose_pure": 400},
        "force_to_cap": True,
        "explicit_natural_ignore_eos": None,
        "protocol_class": "formal-fixed400",
        "source_condition": "Pinned measured body forces min_tokens=max_tokens, ignore_eos=true, stop=[]; a HTTP 400 strips those fields and retries once.",
    },
    "tensorfold-headline": {
        "protocol_id": "mia-lab-tensorfold-headline400-natural-v1",
        "max_tokens_by_workload": {"code": 400, "prose": 400, "structured": 400, "prose_pure": 400},
        "force_to_cap": False,
        "explicit_natural_ignore_eos": False,
        "protocol_class": "reference",
        "source_condition": "Matches MiaAI-Lab TensorFold README headline: sparkDash DecodeBench defaults 400 tokens with natural EOS, prose temperature 0.7 thinking on, code temperature 0 thinking off.",
    },
}


RESIDENT_PREFILL_PROTOCOL_ID = "sparkdash-prefill-resident-workload-context-v2"

PROTOCOL_METADATA: dict[str, Any] = {
    "schema": "mia-http-sse-v2",
    "reference": {
        "repository": "https://github.com/MiaAI-Lab/sparkDash",
        "commit": _SPARKDASH_COMMIT,
        "files": {
            "prompts": f"{_SPARKDASH_BASE}/src/shared/llmPrompts.js",
            "decode_bench": f"{_SPARKDASH_BASE}/server/collectors/DecodeBench.js",
            "streaming": f"{_SPARKDASH_BASE}/server/collectors/LlmStreaming.js",
        },
    },
    "prompt_selection": {
        "code": "DECODE_CODE_PROMPT = DECODE_CODE_TASKS[0]; pickDecodeBenchPrompts(1, 'code') returns it unchanged.",
        "prose": "DECODE_PROSE_PROMPT; pickDecodeBenchPrompts(1, 'prose') returns it unchanged.",
        "fill_to_max_suffix_defined": True,
        "fill_to_max_suffix_applied_by_decode_bench": False,
        "reason": "DecodeBench uses pickDecodeBenchPrompts/decodeBenchPromptForType directly; it does not call withFillToMaxInstruction.",
        "eos_policy": "Legacy mia-lab-v2 is a local variant with natural EOS, including at its default 400-token cap; it sends no ignore_eos/min_tokens/stop override. The current pinned official resident mia-lab-sparkdash-headline400-v1 profile instead sends min_tokens=max_tokens, ignore_eos=true, stop=[] and strips those force-cap fields before retrying after HTTP 400.",
    },
    "pinned_decodebench_defaults": {
        "temperature": 0.7,
        "top_p": 1.0,
        "thinking": True,
        "max_tokens": 400,
        "warmup_max_tokens": 32,
        "note": "These are source defaults only; they are not assumed to be historical measurement settings.",
    },
    "current_profile": {
        "temperature_by_workload": dict(TEMPERATURES),
        "top_p": 1.0,
        "thinking_by_workload": dict(THINKING_BY_WORKLOAD),
        "context_fill_tokens": 0,
        "reasoning": "Code uses thinking OFF; prose uses thinking ON. Both use the pinned Mia prompts and visible reasoning+answer streaming window.",
        "max_tokens": "--outputs (default 400)",
        "warmup_max_tokens": "--warmup-tokens (default 32)",
        "stream": True,
        "stream_options": {"include_usage": True},
        "retries": 0,
    },
    "historical_comparison": {
        "provenance": "Pinned Mia recipe README42-60 and CHANGELOG10 explicitly state these published sampling settings: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/fd3f43e11238553685ebb68b6d8ab2b112818e3c/README.md",
        "code": {"decode_tps": 110.5, "temperature": 0.0, "reasoning": "off"},
        "prose": {"decode_tps": 55.6, "temperature": 0.7, "reasoning": "on"},
        "current_comparison": "Code requests temperature0/thinking OFF; prose requests temperature0.7/thinking ON. Checkpoint, engine, hardware and context differences still prevent a direct published-speed attribution.",
    },
    "timing": {
        "clock": "time.perf_counter",
        "first_visible": "Timestamp of the first non-empty content, reasoning or reasoning_content fragment, matching Mia's tFirst.",
        "last_visible": "Timestamp of the last non-empty content or reasoning fragment, excluding usage/[DONE] teardown.",
        "decode_tps": "(usage.completion_tokens - 1) / (last_visible_perf_counter - first_visible_perf_counter); usage is mandatory and SSE chunk counts are never substituted.",
        "ttft_ms": "First visible reasoning-or-answer timestamp minus request-start perf_counter.",
        "e2e_ms": "SSE response completion timestamp minus request-start perf_counter.",
        "summary": "Median and sample standard deviation across successful measured runs; RSD percent is sample stddev / mean * 100, undefined for fewer than two runs.",
    },
    "request_body": {
        "endpoint_suffix": "/v1/chat/completions",
        "fields": [
            "model", "messages", "max_tokens", "temperature", "top_p", "top_k", "stream",
            "stream_options", "chat_template_kwargs", "min_tokens", "ignore_eos", "stop",
        ],
        "termination_fields": {
            "natural_screen_and_final": "ignore_eos=false is explicit; min_tokens and stop are omitted",
            "official_headline": "min_tokens=max_tokens, ignore_eos=true, stop=[]",
        },
        "thinking_fields": {
            "chat_template_kwargs.enable_thinking": "THINKING_BY_WORKLOAD[workload]",
            "chat_template_kwargs.thinking": "THINKING_BY_WORKLOAD[workload]",
            "chat_template_kwargs.thinking_mode": "'enabled' for prose; 'disabled' for code",
        },
    },
}

_EMPTY_THINK_RE = re.compile(r"<think\b[^>]*>\s*</think\s*>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>(.*)$", re.IGNORECASE | re.DOTALL)
_CODE_PREFACE_RE = re.compile(
    r"^(?:here(?:\s+(?:is|are)|\x27s)\b|sure\b|certainly\b|of\s+course\b|"
    r"the\s+(?:code|solution|implementation|function)\b|this\s+(?:code|solution|implementation|function)\b|"
    r"below\s+is\b|let(?:\s+me|\x27s)\b|i(?:\s+(?:will|can|should|need\s+to|think)|\x27ll)\b|"
    r"we\s+(?:need|can|should|will)\b|first,\s|to\s+solve\s+this\b)",
    re.IGNORECASE,
)


def _nonempty_reasoning(value: str, *, reasoning_field: bool = False) -> tuple[bool, bool]:
    """Return (has_nonempty_reasoning, saw_empty_think_tag)."""
    saw_empty_tag = bool(_EMPTY_THINK_RE.search(value))
    without_empty = _EMPTY_THINK_RE.sub("", value)
    for match in _THINK_BLOCK_RE.finditer(without_empty):
        if match.group(1).strip():
            return True, saw_empty_tag
    unclosed = _THINK_OPEN_RE.search(without_empty)
    if unclosed and unclosed.group(1).strip():
        return True, saw_empty_tag
    # Any remaining non-empty text in a dedicated reasoning field is leakage;
    # ordinary visible answer text is not reasoning unless wrapped in <think>.
    return reasoning_field and bool(without_empty.strip()), saw_empty_tag


def validate_output(text: str, workload: str, reasoning_text: str = "", *, thinking: bool = False) -> dict[str, Any]:
    """Separate the actual workload from prompt-format compliance.

    A markdown fence remains a format warning, not proof of reasoning output.
    Python parsing diagnoses the enclosed code; token counts and text stay raw.
    A syntax error at the output cap is not itself a quality failure.
    """
    reasons: list[str] = []
    format_warnings: list[str] = []
    workload_key = str(workload).strip().lower()
    output = text if isinstance(text, str) else str(text)
    reasoning = reasoning_text if isinstance(reasoning_text, str) else str(reasoning_text)
    has_reasoning_leak, empty_reasoning_tags = _nonempty_reasoning(reasoning, reasoning_field=True)
    content_leak, empty_content_tags = _nonempty_reasoning(output)
    has_reasoning_leak = has_reasoning_leak or content_leak
    empty_think_tags = empty_reasoning_tags or empty_content_tags

    visible_output = _EMPTY_THINK_RE.sub("", output).strip()
    reasoning_expected = thinking and workload_key in (
        "prose", "structured", "prose_pure", "prose_pure_greedy",
    )
    if not visible_output and not (reasoning_expected and reasoning.strip()):
        reasons.append("empty_output")
    if has_reasoning_leak and not reasoning_expected:
        reasons.append("reasoning_leak")

    first_line = next((line.strip() for line in visible_output.splitlines() if line.strip()), "")
    non_code_preface = False
    markdown_fence = False
    ast_parse_ok: bool | None = None
    ast_parse_error: str | None = None
    if workload_key == "code":
        markdown_fence = first_line.startswith("```")
        non_code_preface = bool(_CODE_PREFACE_RE.match(first_line))
        code_output = visible_output
        if markdown_fence:
            format_warnings.append("markdown_fence")
            code_lines = visible_output.splitlines()[1:]
            if code_lines and code_lines[-1].strip() == "```":
                code_lines.pop()
            code_output = "\n".join(code_lines)
        code_first_line = next((line.strip() for line in code_output.splitlines() if line.strip()), "")
        non_code_preface = bool(_CODE_PREFACE_RE.match(code_first_line))
        if non_code_preface:
            reasons.append("non_code_preface")
        if not code_output.strip():
            reasons.append("empty_code")
        if code_output:
            try:
                import ast

                parsed = ast.parse(code_output)
                if not parsed.body or all(
                    isinstance(node, ast.Expr) and isinstance(node.value, (ast.Name, ast.Constant))
                    for node in parsed.body
                ):
                    raise SyntaxError("No executable Python statement")
                ast_parse_ok = True
            except SyntaxError as exc:
                ast_parse_ok = False
                ast_parse_error = f"{exc.msg} (line {exc.lineno})"
                if code_first_line and not non_code_preface:
                    probe = code_first_line + ("\n    pass\n" if code_first_line.endswith(":") else "")
                    try:
                        if code_first_line.startswith("@"):
                            ast.parse(code_first_line[1:], mode="eval")
                        else:
                            prefix = ast.parse(probe)
                            if not prefix.body or any(
                                isinstance(node, ast.Expr) and isinstance(node.value, (ast.Name, ast.Constant))
                                for node in prefix.body
                            ):
                                raise SyntaxError("No executable Python statement")
                    except SyntaxError:
                        non_code_preface = True
                        reasons.append("non_python_start")
    elif workload_key == "structured":
        values = visible_output.split()
        if not values or any(not value.isdigit() for value in values):
            reasons.append("structured_output_not_integer_sequence")
        elif [int(value) for value in values] != list(range(1, len(values) + 1)):
            reasons.append("structured_output_not_consecutive_from_one")
        elif int(values[-1]) != 200:
            reasons.append("structured_count_did_not_reach_200")
    elif workload_key == "code_ts":
        if not visible_output.strip():
            reasons.append("empty_code")
    elif workload_key == "json_schema":
        if not visible_output.strip():
            reasons.append("empty_json_schema")
        else:
            try:
                parsed_json = json.loads(visible_output)
            except json.JSONDecodeError as exc:
                reasons.append(f"json_schema_parse_error: {exc.msg}")
            else:
                if not isinstance(parsed_json, (dict, list)):
                    reasons.append("json_schema_not_object_or_array")
    elif workload_key == "prefill":
        if visible_output.strip().lower() != "ok":
            reasons.append("prefill_reply_not_ok")
    elif workload_key not in ("prose", "prose_pure", "prose_pure_greedy"):
        reasons.append("unknown_workload")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "validation_version": 3,
        "format_compliant": not format_warnings and not reasons,
        "format_warnings": format_warnings,
        "workload": workload_key,
        "output_chars": len(output),
        "output_lines": len(output.splitlines()),
        "reasoning_chars": len(reasoning),
        "reasoning_leak": has_reasoning_leak and not reasoning_expected,
        "reasoning_enabled": thinking,
        "reasoning_expected": reasoning_expected,
        "reasoning_only": reasoning_expected and bool(reasoning.strip()) and not visible_output,
        "empty_think_tags": empty_think_tags,
        "non_code_preface": non_code_preface,
        "markdown_fence": markdown_fence,
        "ast_parse_ok": ast_parse_ok,
        "ast_parse_error": ast_parse_error,
        "first_nonempty_line": first_line[:240],
        "truncation_is_not_auto_failure": True,
    }


def _request_body(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    *,
    thinking: bool,
    top_p: float | int = 1,
    top_k: int | None = None,
    force_to_cap: bool = False,
    ignore_eos: bool | None = None,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            "thinking": thinking,
            "thinking_mode": "enabled" if thinking else "disabled",
        },
    }
    if top_k is not None:
        body["top_k"] = top_k
    if force_to_cap:
        body.update({"min_tokens": max_tokens, "ignore_eos": True, "stop": []})
    elif ignore_eos is not None:
        body["ignore_eos"] = ignore_eos
    return body


def _sse_data_events(response, raw_sink: bytearray | None = None):
    data_lines: list[str] = []
    while True:
        raw_line = response.readline()
        if not raw_line:
            if data_lines:
                yield "\n".join(data_lines)
            return
        if raw_sink is not None:
            raw_sink.extend(raw_line)
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and field == "data":
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)

def _safe_error(exc: BaseException, api_key: str | None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message.replace(api_key, "[REDACTED]") if api_key else message


def _perform_request(
    *,
    url: str,
    body: dict[str, Any],
    api_key: str | None,
    workload: str,
    prompt: str,
    phase: str,
    sequence_index: int,
    run_index: int | None,
    require_usage: bool,
    require_prompt_usage: bool = False,
    require_complete_stream: bool = False,
    capture_raw_sse: bool = False,
    request_timeout: float = 360,
    thinking: bool | None = None,
    quality_applicable: bool = True,
) -> dict[str, Any]:
    request_start = time.perf_counter()
    record: dict[str, Any] = {
        "sequence_index": sequence_index,
        "phase": phase,
        "workload": workload,
        "run_index": run_index,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "parameters": {
            key: body[key]
            for key in (
                "model", "max_tokens", "temperature", "top_p", "top_k", "min_tokens",
                "ignore_eos", "stop", "stream", "stream_options", "chat_template_kwargs",
                "thinking", "enable_thinking", "thinking_mode", "reasoning_effort", "reasoning_budget",
            )
            if key in body
        },
        "output_text": "",
        "reasoning_text": "",
        "raw_usage": None,
        "usage_events": [],
        "prompt_tokens": None,
        "cached_prompt_tokens": None,
        "prefill_tps": None,
        "server_prompt_ms": None,
        "server_prompt_tokens": None,
        "finish_reason": None,
        "http_status": None,
        "http_error_body": None,
        "sse_event_count": 0,
        "sse_done_seen": False,
        "first_content_perf_counter": None,
        "last_content_perf_counter": None,
        "first_reasoning_perf_counter": None,
        "last_reasoning_perf_counter": None,
        "content_chunks": 0,
        "reasoning_chunks": 0,
        "segment_events": [],
        "first_visible_perf_counter": None,
        "last_visible_perf_counter": None,
        "request_start_perf_counter": request_start,
        "request_end_perf_counter": None,
        "ttft_ms": None,
        "decode_window_ms": None,
        "e2e_ms": None,
        "completion_tokens": None,
        "decode_tokens": None,
        "decode_tps": None,
        "validation": None,
        "raw_sse_base64": None,
        "failures": [],
        "success": False,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    protocol_errors: list[str] = []
    raw_sse = bytearray() if capture_raw_sse else None
    try:
        with urlopen(request, timeout=request_timeout) as response:
            record["http_status"] = getattr(response, "status", None)
            for data in _sse_data_events(response, raw_sse):
                if data == "[DONE]":
                    record["sse_done_seen"] = True
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError as exc:
                    protocol_errors.append(f"invalid_sse_json: {exc.msg}; data={data[:200]!r}")
                    continue
                if not isinstance(payload, dict):
                    protocol_errors.append("sse_json_not_object")
                    continue
                record["sse_event_count"] += 1
                timings = payload.get("timings")
                if isinstance(timings, dict):
                    server_ms = timings.get("prompt_ms")
                    server_tokens = timings.get("prompt_n")
                    if isinstance(server_ms, (int, float)) and server_ms > 0:
                        record["server_prompt_ms"] = float(server_ms)
                    if isinstance(server_tokens, int) and not isinstance(server_tokens, bool) and server_tokens > 0:
                        record["server_prompt_tokens"] = server_tokens
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    record["raw_usage"] = usage
                    record["usage_events"].append(usage)
                    prompt_tokens = usage.get("prompt_tokens")
                    if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens >= 0:
                        record["prompt_tokens"] = prompt_tokens
                    prompt_details = usage.get("prompt_tokens_details")
                    cached_tokens = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
                    if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool) and cached_tokens >= 0:
                        record["cached_prompt_tokens"] = cached_tokens
                elif usage is not None:
                    protocol_errors.append("usage_not_object")
                choices = payload.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else None
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason") is not None:
                    record["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta")
                content = delta.get("content") if isinstance(delta, dict) else choice.get("text")
                reasoning = delta.get("reasoning") if isinstance(delta, dict) else None
                if not (isinstance(reasoning, str) and reasoning) and isinstance(delta, dict):
                    reasoning_content = delta.get("reasoning_content")
                    if isinstance(reasoning_content, str) and reasoning_content:
                        reasoning = reasoning_content
                has_content = isinstance(content, str) and bool(content)
                has_reasoning = isinstance(reasoning, str) and bool(reasoning)
                if has_content or has_reasoning:
                    now = time.perf_counter()
                    if record["first_visible_perf_counter"] is None:
                        record["first_visible_perf_counter"] = now
                    record["last_visible_perf_counter"] = now
                    if has_content:
                        if record["first_content_perf_counter"] is None:
                            record["first_content_perf_counter"] = now
                        record["last_content_perf_counter"] = now
                        content_parts.append(content)
                        record["content_chunks"] += 1
                    if has_reasoning:
                        if record["first_reasoning_perf_counter"] is None:
                            record["first_reasoning_perf_counter"] = now
                        record["last_reasoning_perf_counter"] = now
                        reasoning_parts.append(reasoning)
                        record["reasoning_chunks"] += 1
                    record["segment_events"].append({
                        "t": now,
                        "kind": "content" if has_content else "reasoning",
                        "chars": len(content) if has_content else len(reasoning),
                    })
    except HTTPError as exc:
        record["http_status"] = exc.code
        error_body = exc.read().decode("utf-8", errors="replace")
        record["http_error_body"] = (
            error_body.replace(api_key, "[REDACTED]") if api_key else error_body
        )
        record["failures"].append({"stage": "http", "reason": _safe_error(exc, api_key)})
    except (URLError, TimeoutError, OSError) as exc:
        record["failures"].append({"stage": "transport", "reason": _safe_error(exc, api_key)})
    except Exception as exc:
        record["failures"].append({"stage": "request", "reason": _safe_error(exc, api_key)})

    request_end = time.perf_counter()
    record["request_end_perf_counter"] = request_end
    record["e2e_ms"] = round((request_end - request_start) * 1000, 6)
    if raw_sse is not None:
        record["raw_sse_base64"] = base64.b64encode(raw_sse).decode("ascii")
    output_text = "".join(content_parts)
    reasoning_text = "".join(reasoning_parts)
    record["output_text"] = output_text
    record["reasoning_text"] = reasoning_text
    first_visible = record["first_visible_perf_counter"]
    last_visible = record["last_visible_perf_counter"]
    if first_visible is not None:
        record["ttft_ms"] = round((first_visible - request_start) * 1000, 6)
    if first_visible is not None and last_visible is not None:
        record["decode_window_ms"] = round((last_visible - first_visible) * 1000, 6)
    if record["prompt_tokens"] is not None and record["ttft_ms"] and record["ttft_ms"] > 0:
        record["prefill_tps"] = round(record["prompt_tokens"] / (record["ttft_ms"] / 1000), 6)

    for reason in protocol_errors:
        record["failures"].append({"stage": "protocol", "reason": reason})
    if not output_text and not reasoning_text:
        record["failures"].append({"stage": "protocol", "reason": "no_visible_content_or_reasoning"})
    if require_complete_stream:
        if not record["sse_done_seen"]:
            record["failures"].append({"stage": "protocol", "reason": "missing_sse_done"})
        if record["finish_reason"] is None:
            record["failures"].append({"stage": "protocol", "reason": "missing_finish_reason"})

    raw_usage = record["raw_usage"]
    completion_tokens = raw_usage.get("completion_tokens") if isinstance(raw_usage, dict) else None
    if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool) and completion_tokens >= 1:
        record["completion_tokens"] = completion_tokens
        record["decode_tokens"] = completion_tokens - 1
        if require_usage and record["decode_window_ms"] is not None and record["decode_window_ms"] > 0:
            record["decode_tps"] = round((completion_tokens - 1) / (record["decode_window_ms"] / 1000), 6)
    elif require_usage:
        record["failures"].append({
            "stage": "usage",
            "reason": "missing_or_invalid_usage_completion_tokens; SSE chunk count is not substituted",
        })

    if require_prompt_usage and not isinstance(record["prompt_tokens"], int):
        record["failures"].append({
            "stage": "usage",
            "reason": "missing_or_invalid_usage_prompt_tokens; no prompt estimate is substituted",
        })
    if require_usage and (record["decode_window_ms"] is None or record["decode_window_ms"] <= 0):
        record["failures"].append({"stage": "timing", "reason": "non_positive_visible_decode_window"})

    record["quality_applicable"] = quality_applicable
    if quality_applicable and isinstance(record["http_status"], int) and 200 <= record["http_status"] < 300:
        chat_template_kwargs = body.get("chat_template_kwargs")
        body_thinking = (
            chat_template_kwargs.get("enable_thinking")
            if isinstance(chat_template_kwargs, dict) else None
        )
        validation_thinking = thinking if isinstance(thinking, bool) else bool(body_thinking)
        validation = validate_output(
            output_text,
            workload,
            reasoning_text,
            thinking=validation_thinking,
        )
        record["validation"] = validation
        record["quality_valid"] = validation["valid"]
        record["quality_status"] = "valid" if validation["valid"] else "invalid"
        record["quality_failures"] = list(validation["reasons"])
        record["quality_details"] = {
            "reasoning_only": validation["reasoning_only"],
            "format_compliant": validation["format_compliant"],
            "format_warnings": list(validation["format_warnings"]),
            "cap_truncated": record["finish_reason"] == "length",
            "natural_eos_finish_reason": record["finish_reason"] == "stop",
        }
        if workload == "structured":
            values = _EMPTY_THINK_RE.sub("", output_text).split()
            numbers = [int(value) for value in values if value.isdigit()]
            seen_numbers: set[int] = set()
            duplicate_numbers: set[int] = set()
            for number in numbers:
                if number in seen_numbers:
                    duplicate_numbers.add(number)
                seen_numbers.add(number)
            missing_numbers = [number for number in range(1, 201) if number not in seen_numbers]
            out_of_range_numbers = [number for number in numbers if not 1 <= number <= 200]
            non_integer_tokens = [value for value in values if not value.isdigit()]
            sequence_ok = bool(values) and not non_integer_tokens and numbers == list(range(1, 201))
            record["quality_details"].update({
                "numbers_observed": len(numbers),
                "first_number": numbers[0] if numbers else None,
                "last_number": numbers[-1] if numbers else None,
                "ordered_consecutive_from_one": bool(values) and not non_integer_tokens
                and numbers == list(range(1, len(numbers) + 1)),
                "complete_count_1_to_200": sequence_ok,
                "missing_numbers": missing_numbers,
                "duplicate_numbers": sorted(duplicate_numbers),
                "out_of_range_numbers": out_of_range_numbers,
                "non_integer_tokens": non_integer_tokens,
                "thinking_only": not output_text.strip() and bool(reasoning_text.strip()),
            })
    elif quality_applicable:
        validation = {
            "status": "unassessed",
            "valid": None,
            "reasons": [],
            "reason": "no successful HTTP response; output quality was not evaluated",
        }
        record["validation"] = validation
        record["quality_valid"] = None
        record["quality_status"] = "unassessed"
        record["quality_failures"] = []
        record["quality_details"] = {
            "status": "unassessed",
            "reason": validation["reason"],
            "cap_truncated": record["finish_reason"] == "length",
        }
    else:
        validation = {
            "status": "not_applicable",
            "valid": None,
            "reasons": [],
            "reason": "short prefill reply; workload completion and format were not evaluated",
        }
        record["validation"] = validation
        record["quality_valid"] = None
        record["quality_status"] = "not_applicable"
        record["quality_failures"] = []
        record["quality_details"] = {
            "status": "not_applicable",
            "reason": validation["reason"],
            "cap_truncated": record["finish_reason"] == "length",
        }
    for reason in validation.get("reasons", []):
        record["failures"].append({"stage": "validation", "reason": reason})
    throughput_failures = [failure for failure in record["failures"] if failure.get("stage") != "validation"]
    record["throughput_eligible"] = not throughput_failures and (
        not require_usage or isinstance(record["decode_tps"], (int, float))
    )
    record["success"] = record["throughput_eligible"] and (
        not quality_applicable or bool(validation["valid"])
    )
    return record


def _segment_metrics(record: dict[str, Any]) -> dict[str, Any] | None:
    events = record.get("segment_events") or []
    if not events:
        return None
    reasoning_events = [event for event in events if event.get("kind") == "reasoning"]
    content_events = [event for event in events if event.get("kind") == "content"]
    completion_tokens = record.get("completion_tokens")
    reasoning_chars = sum(event.get("chars", 0) for event in reasoning_events)
    content_chars = sum(event.get("chars", 0) for event in content_events)
    total_chars = reasoning_chars + content_chars

    def _tokens_for(chars: int, chunks: int) -> float:
        if isinstance(completion_tokens, int) and completion_tokens >= 1 and total_chars > 0:
            return completion_tokens * chars / total_chars
        return float(chunks)

    def _window(subset: list[dict[str, Any]], chars: int) -> dict[str, Any] | None:
        if len(subset) < 2:
            return None
        window_ms = (subset[-1]["t"] - subset[0]["t"]) * 1000
        if window_ms <= 0:
            return None
        estimated_tokens = _tokens_for(chars, len(subset))
        return {
            "chunks": len(subset),
            "chars": chars,
            "estimated_tokens": round(estimated_tokens, 3),
            "window_ms": round(window_ms, 6),
            "tps": round(max(estimated_tokens - 1, 0) / (window_ms / 1000), 6),
        }

    peak_tps = None
    if len(events) >= 2:
        best = 0.0
        left = 0
        for right in range(len(events)):
            while events[right]["t"] - events[left]["t"] > 1.0:
                left += 1
            window_seconds = events[right]["t"] - events[left]["t"]
            if window_seconds >= 0.9:
                window_chars = sum(event.get("chars", 0) for event in events[left:right + 1])
                window_tokens = _tokens_for(window_chars, right - left + 1)
                best = max(best, window_tokens / window_seconds)
        peak_tps = round(best, 6) if best > 0 else None

    return {
        "thinking": _window(reasoning_events, reasoning_chars),
        "answer": _window(content_events, content_chars),
        "peak_tps": peak_tps,
        "token_estimate_note": (
            "segment tokens are apportioned from completion_tokens by visible character share; "
            "chunks are SSE fragments, not necessarily one token each"
        ),
    }


def _summary_for(workload: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        record["decode_tps"]
        for record in records
        if record["workload"] == workload and record["phase"] == "measurement"
        and record["success"] and isinstance(record["decode_tps"], (int, float))
        and math.isfinite(record["decode_tps"])
    ]
    if not successful:
        return {
            "workload": workload,
            "successful_runs": 0,
            "failed_runs": sum(
                1 for record in records
                if record["workload"] == workload and record["phase"] == "measurement"
                and not record["success"]
            ),
            "median_decode_tps": None,
            "sample_stddev_decode_tps": None,
            "rsd_percent": None,
        }
    average = statistics.fmean(successful)
    stddev = statistics.stdev(successful) if len(successful) > 1 else None
    segment_summaries = [
        metrics
        for record in records
        if record["workload"] == workload and record["phase"] == "measurement"
        and record["success"]
        for metrics in [_segment_metrics(record)]
        if metrics is not None
    ]
    return {
        "workload": workload,
        "successful_runs": len(successful),
        "failed_runs": sum(
            1 for record in records
            if record["workload"] == workload and record["phase"] == "measurement"
            and not record["success"]
        ),
        "median_decode_tps": statistics.median(successful),
        "sample_stddev_decode_tps": stddev,
        "rsd_percent": (stddev / average * 100) if stddev is not None and average else None,
        "mean_decode_tps": average,
        "segment_summaries": segment_summaries,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, help="Path for the JSON measurement record")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--outputs", type=int, default=400)
    parser.add_argument("--workloads", nargs="+", choices=tuple(PROMPTS), default=["code", "prose"])
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--warmup-measured-prompt", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.outputs < 1:
        parser.error("--outputs must be at least 1")
    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens must be non-negative")
    if args.warmup_rounds < 1:
        parser.error("--warmup-rounds must be at least 1")

    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/v1/chat/completions"
    api_key = os.environ.get("OPENAI_API_KEY") or None
    out_path = Path(args.out)
    record: dict[str, Any] = {
        "schema": PROTOCOL_METADATA["schema"],
        "protocol_id": PROTOCOL_ID,
        "status": "running",
        "base_url": base_url,
        "endpoint": url,
        "model": args.model,
        "source_commit": _SPARKDASH_COMMIT,
        "protocol_metadata": PROTOCOL_METADATA,
        "requested": {
            "runs_per_workload": args.runs,
            "outputs": args.outputs,
            "workloads": list(args.workloads),
            "warmup_rounds": args.warmup_rounds,
            "warmup_tokens": args.warmup_tokens,
            "concurrency": 1,
            "parallel_sum": False,
            "warmup_prompt": "measured" if args.warmup_measured_prompt else "Mia separate code warmup",
            "authorization": "bearer_from_OPENAI_API_KEY" if api_key else "none",
        },
        "prompt_sha256": {
            workload: hashlib.sha256(PROMPTS[workload].encode("utf-8")).hexdigest()
            for workload in args.workloads
        },
        "events": [],
        "failures": [],
        "summaries": [],
    }

    def save() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = out_path.with_name(out_path.name + ".tmp")
        temporary_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(out_path)

    try:
        for workload in args.workloads:
            warmup_prompt = PROMPTS[workload] if args.warmup_measured_prompt else WARMUP_PROMPTS[workload]
            if args.warmup_tokens > 0:
                for warmup_round in range(1, args.warmup_rounds + 1):
                    warmup_event = _perform_request(
                        url=url,
                        body=_request_body(
                            args.model, warmup_prompt, args.warmup_tokens, TEMPERATURES[workload],
                            thinking=THINKING_BY_WORKLOAD[workload],
                        ),
                        api_key=api_key,
                        workload=workload,
                        prompt=warmup_prompt,
                        phase="warmup",
                        sequence_index=len(record["events"]) + 1,
                        run_index=None,
                        require_usage=False,
                        quality_applicable=False,
                    )
                    warmup_event["warmup_round"] = warmup_round
                    record["events"].append(warmup_event)
                    for failure in warmup_event["failures"]:
                        record["failures"].append({
                            "sequence_index": warmup_event["sequence_index"],
                            "phase": "warmup",
                            "workload": workload,
                            "warmup_round": warmup_round,
                            **failure,
                        })
                    save()

            for run_index in range(1, args.runs + 1):
                event = _perform_request(
                    url=url,
                    body=_request_body(args.model, PROMPTS[workload], args.outputs, TEMPERATURES[workload],
                                       thinking=THINKING_BY_WORKLOAD[workload]),
                    api_key=api_key,
                    workload=workload,
                    prompt=PROMPTS[workload],
                    phase="measurement",
                    sequence_index=len(record["events"]) + 1,
                    run_index=run_index,
                    require_usage=True,
                )
                record["events"].append(event)
                for failure in event["failures"]:
                    record["failures"].append({
                        "sequence_index": event["sequence_index"],
                        "phase": "measurement",
                        "workload": workload,
                        "run_index": run_index,
                        **failure,
                    })
                record["summaries"] = [_summary_for(kind, record["events"]) for kind in args.workloads]
                save()
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["failures"].append({"stage": "execution", "reason": "KeyboardInterrupt"})
        record["summaries"] = [_summary_for(kind, record["events"]) for kind in args.workloads]
        save()
        print(json.dumps({"status": record["status"], "out": str(out_path)}, ensure_ascii=False))
        return 130

    record["summaries"] = [_summary_for(kind, record["events"]) for kind in args.workloads]
    record["status"] = "completed" if not record["failures"] else "completed_with_failures"
    save()
    print(json.dumps({"status": record["status"], "out": str(out_path), "summaries": record["summaries"]}, ensure_ascii=False))
    return 0 if not record["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Drive pinned sparkDash native jobs and prove each envelope honoured the request.

This is the small, readable front door used in this repository's docs. The campaign
runner (run_sparkdash_original_matrix.py) does the same thing with the full control
snapshot and restore steps; this script exists so a reader can reproduce one grid
without the campaign bookkeeping.

Usage:
  run_matrix.py --state-file STATE.json --node PATH --config k6-open \
      --case prose --concurrencies 1 2 4 --max-tokens 400 [--repeats 3] \
      [--out matrix.json]

  run_matrix.py --state-file STATE.json --node PATH --config k6-open \
      --case prefill --context-sizes 4096 8192 16384 [--repeats 3]

Every job gets its own options file and envelope under the state file's directory.
An existing envelope is never replaced: the job is skipped with an error, so a
number can never be silently re-rolled.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_CONCURRENCIES = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32}
MIN_MAX_TOKENS, MAX_MAX_TOKENS = 64, 2048
MIN_CONTEXT, MAX_CONTEXT = 256, 300000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--config", required=True, help="configuration id recorded in the envelope")
    parser.add_argument("--case", required=True, choices=("prose", "code", "prefill"))
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1])
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[4096])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prefix", default="matrix")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if any(n not in ALLOWED_CONCURRENCIES for n in args.concurrencies):
        parser.error(f"concurrencies must be drawn from {sorted(ALLOWED_CONCURRENCIES)}")
    if not MIN_MAX_TOKENS <= args.max_tokens <= MAX_MAX_TOKENS:
        parser.error(f"max-tokens must be {MIN_MAX_TOKENS}..{MAX_MAX_TOKENS}")
    if any(n < MIN_CONTEXT or n > MAX_CONTEXT for n in args.context_sizes):
        parser.error(f"context-sizes must be {MIN_CONTEXT}..{MAX_CONTEXT}")
    if args.repeats < 1:
        parser.error("repeats must be at least 1")
    if not args.state_file.is_file() or not args.node.is_file():
        parser.error("state file and node runtime must exist")

    state = json.loads(args.state_file.read_text(encoding="utf-8"))
    if state.get("status") != "ready" or not state.get("engine_instance_id"):
        parser.error("state file must describe a ready engine")
    root = args.state_file.resolve().parent
    runner = root / "bench" / "sparkdash-exact" / "runner.mjs"
    if not runner.is_file():
        runner = root / "sparkdash-exact" / "runner.mjs"
    if not runner.is_file():
        parser.error("pinned runner.mjs not found under bench/sparkdash-exact/")

    base_url = state["base_url"]
    from urllib.parse import urlsplit
    address = urlsplit(base_url)
    if address.scheme != "http" or address.hostname not in ("127.0.0.1", "localhost") or not address.port:
        parser.error("state file must describe a loopback engine")

    out_path = args.out or root / f"{args.prefix}-matrix.json"
    record = {
        "schema": "recipe-matrix-v1",
        "started_utc": utc_now(),
        "engine_instance_id": state["engine_instance_id"],
        "configuration_id": args.config,
        "case": args.case,
        "request": {
            "concurrencies": list(args.concurrencies),
            "max_tokens": args.max_tokens,
            "context_sizes": list(args.context_sizes),
            "repeats": args.repeats,
        },
        "jobs": [],
        "failures": [],
    }

    for repeat in range(1, args.repeats + 1):
        run_id = f"{args.prefix}-{args.config}-{args.case}-{repeat:02d}"
        envelope = root / f"{run_id}.json"
        options_path = root / f"{run_id}-options.json"
        if envelope.exists() or options_path.exists():
            record["failures"].append({"run_id": run_id, "reason": "invocation already exists"})
            continue
        options = {
            "case": args.case,
            "host": address.hostname,
            "port": address.port,
            "modelId": state["model_id"],
            "outputDir": str(root),
            "runId": run_id,
            "configurationId": args.config,
            "engineInstanceId": state["engine_instance_id"],
            "configuration": {},
            "freshSnapshot": {"observation_only_no_generation": True, "source": "run_matrix.py"},
            "metadata": {"matrix_id": args.prefix, "experiment_role": "optimization-ab"},
            "sparkId": "local",
            "tls": False,
            "apiKey": None,
        }
        if args.case == "prefill":
            options["contextSizes"] = list(args.context_sizes)
        else:
            options["concurrencies"] = list(args.concurrencies)
            options["maxTokens"] = args.max_tokens
        atomic_write(options_path, options)

        result = subprocess.run(
            [str(args.node), str(runner), "--options", str(options_path)],
            capture_output=True, text=True, timeout=1800,
        )
        job = {"run_id": run_id, "exit_code": result.returncode,
               "stdout": result.stdout.strip(), "stderr": result.stderr.strip()[-2000:]}
        if envelope.is_file():
            native = json.loads(envelope.read_text(encoding="utf-8"))
            config = (native.get("native_job") or {}).get("config") or {}
            if args.case == "prefill":
                got = config.get("contextSizes")
                want = list(args.context_sizes)
            else:
                got = config.get("concurrencies")
                want = list(args.concurrencies)
            job["effective"] = got
            job["requested"] = want
            job["contract_ok"] = got == want
            if got != want:
                record["failures"].append(
                    {"run_id": run_id, "reason": f"effective {got} != requested {want}"})
            rows = (native.get("native_job") or {}).get("results") or []
            job["rows"] = len(rows)
            job["streams_ok"] = [row.get("streamsOk") for row in rows]
            job["streams_failed"] = [row.get("streamsFailed") for row in rows]
        else:
            record["failures"].append({"run_id": run_id, "reason": "no envelope produced"})
        if result.returncode:
            record["failures"].append({"run_id": run_id, "reason": f"exit {result.returncode}"})
        record["jobs"].append(job)
        atomic_write(out_path, record)

    record["completed_utc"] = utc_now()
    record["status"] = "completed_with_failures" if record["failures"] else "completed"
    atomic_write(out_path, record)
    print(json.dumps({"out": str(out_path), "status": record["status"],
                      "jobs": len(record["jobs"]), "failures": record["failures"]}, indent=2))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare greedy replies across two engine builds (drafts on vs drafts off).

exact.py checks three lanes inside one server. This tool answers the question a
single server cannot: does a greedy reply change when the speculative-decoding
path is removed entirely? It compares the "serial" lane of two exact.py --dump
files produced by two engine builds.

Usage: verify/exactness_cross.py DRAFT_ON_DUMP.json DRAFT_OFF_DUMP.json

Exit 0 all comparable rows equal, 1 a row differs, 2 a dump could not be read,
3 no row is comparable (recorded as UNCHECKED, never as a pass).
"""
import json
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: exactness_cross.py DRAFT_ON_DUMP.json DRAFT_OFF_DUMP.json", file=sys.stderr)
        return 2
    try:
        left, right = load(sys.argv[1]), load(sys.argv[2])
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read dumps: {exc}", file=sys.stderr)
        return 2
    left_rows = {row["prompt_index"]: row for row in left.get("rows", [])}
    right_rows = {row["prompt_index"]: row for row in right.get("rows", [])}
    common = sorted(set(left_rows) & set(right_rows))
    if not common:
        print("UNCHECKED: no prompt index appears in both dumps")
        return 3
    differing = []
    for index in common:
        a, b = left_rows[index], right_rows[index]
        same_text = a["serial"] == b["serial"]
        same_finish = a["finish_reason"] == b["finish_reason"]
        same_tokens = (
            a.get("serial_token_sha") is None or b.get("serial_token_sha") is None
            or a["serial_token_sha"] == b["serial_token_sha"]
        )
        verdict = "same" if (same_text and same_finish and same_tokens) else "DIFFERS"
        print(
            f"#{index:>2}  draft-on {a['serial']:<12} draft-off {b['serial']:<12}  "
            f"finish {a['finish_reason']}/{b['finish_reason']}  {verdict}"
        )
        if verdict == "DIFFERS":
            differing.append(index)
    print(f"compared {len(common)} prompts across {sys.argv[1]} and {sys.argv[2]}")
    if differing:
        print(f"FAIL cross: {len(differing)} prompt(s) differ: {differing}")
        return 1
    print("PASS cross: greedy replies are identical with and without the draft path")
    return 0


if __name__ == "__main__":
    sys.exit(main())

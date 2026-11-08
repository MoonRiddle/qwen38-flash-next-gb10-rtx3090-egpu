#!/usr/bin/env python3
"""Verify the exact q9 source identity before applying experimental image patches."""
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("/tmp/autoround-source-identity.json").read_text())
expected = {item["path"]: item["sha256"] for item in manifest["files"]}
expected["/usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py"] = (
    "6ffc537effd05ec6515bd8578c28d292cb376d7eb4a87697466206dfd57748b5"
)
for name, wanted in expected.items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != wanted:
        raise RuntimeError(f"source hash mismatch: {name}: expected {wanted}, got {actual}")
    print(f"verified {actual} {name}")

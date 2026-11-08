#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-.}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$ROOT/vllm/v1/worker/gpu/model_runner.py"
EXPECTED="e36fea8fd26e9719b8b4cbd6e0db70bdeb448a61515a0b2a1e60ba3eecf63fee"
ACTUAL="$(sha256sum "$TARGET" | cut -d ' ' -f 1)"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    printf 'Refusing to patch: expected SHA-256 %s, got %s for %s\n' "$EXPECTED" "$ACTUAL" "$TARGET" >&2
    exit 1
fi
patch --dry-run --fuzz=0 -d "$ROOT" -p1 < "$PATCH_DIR/0001-resident-round-profile.patch"
patch --fuzz=0 -d "$ROOT" -p1 < "$PATCH_DIR/0001-resident-round-profile.patch"

#!/usr/bin/env bash
set -euo pipefail
SRC=/srv/vault/int4-models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN
DST=/home/user/OMP_Workspace/qwen38-gb10-3090-20261029/saren-target-no-mtp
mkdir -p "$DST"
cd "$DST"
for f in chat_template.jinja generation_config.json config.json config.json.autoround; do
    [[ -e "$f" ]] || cp "$SRC/$f" "$f"
done
python3 - <<'PY'
import json, os
src="/srv/vault/int4-models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN"
dst="/home/user/OMP_Workspace/qwen38-gb10-3090-20261029/saren-target-no-mtp"
idx=json.load(open(f"{src}/model.safetensors.index.json"))
wm=idx["weight_map"]
keep={k:v for k,v in wm.items() if "layers.48" not in k and "mtp" not in k.lower()}
json.dump({"metadata": idx.get("metadata", {}), "weight_map": keep},
          open(f"{dst}/model.safetensors.index.json", "w"), indent=2)
for shard in sorted(set(keep.values())):
    p=f"{dst}/{shard}"
    if not os.path.lexists(p):
        os.symlink(f"{src}/{shard}", p)
print("tensors", len(keep), "shards", len(set(keep.values())))
PY

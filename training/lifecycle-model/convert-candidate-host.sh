#!/bin/sh
# Convert a trained lifecycle student into the served Q4_K_M GGUF and publish
# it as an immutable candidate, so the qualification surface (constrained
# serving evaluation and the independent judge) can fetch exactly these bytes.
#
#   stado host run-helper <target> oko-lifecycle-convert.sh --uuid <JOB_UUID>
#
# Publishing here is deliberately NOT a release of a qualified model: the
# destination is .../candidates/..., and only training/lifecycle-model/
# publish-recovered-candidate-host.py may write .../models/... after the gate.
set -eu

job="${1:?job uuid required}"
stado="$HOME/.stado/bin/stado"
work="/mnt/wisent-training/stado/training/oko-lifecycle-$job"
llama_revision=b6180
llama_root=/mnt/wisent-staging/llama.cpp
model_name=oko-lifecycle-qwen3-4b-q4_k_m.gguf
part_bytes=134217728

[ -d "$work/student" ] || { echo "no trained student at $work/student" >&2; exit 1; }
cd "$work"

/usr/local/bin/pip3 install --quiet --break-system-packages gguf sentencepiece protobuf

if [ ! -d "$llama_root/.git" ]; then
    git clone --depth 1 --branch "$llama_revision" https://github.com/ggml-org/llama.cpp "$llama_root"
fi
if [ ! -x "$llama_root/build/bin/llama-quantize" ]; then
    cmake -S "$llama_root" -B "$llama_root/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF >cmake.log 2>&1
    cmake --build "$llama_root/build" --target llama-quantize -j "$(nproc)" >>cmake.log 2>&1
fi

if [ ! -s student-bf16.gguf ]; then
    /usr/bin/python3 "$llama_root/convert_hf_to_gguf.py" student \
        --outfile student-bf16.gguf --outtype bf16
fi
if [ ! -s "$model_name" ]; then
    "$llama_root/build/bin/llama-quantize" student-bf16.gguf "$model_name" Q4_K_M
fi

/usr/local/bin/pip3 freeze >python-requirements.lock

digest=$(sha256sum "$model_name" | cut -d' ' -f1)
destination="stado://releases/oko/candidates/lifecycle-qwen3-4b/$digest"

rm -rf candidate-parts
mkdir candidate-parts
split -b "$part_bytes" -d -a 3 "$model_name" "candidate-parts/$model_name.part-"

for part in candidate-parts/*; do
    "$stado" storage stat "$destination/large-output/$(basename "$part")" >/dev/null 2>&1 ||
        "$stado" storage put "$destination/large-output/$(basename "$part")" "$part" >/dev/null
done
for name in metrics.json predictions.jsonl python-requirements.lock; do
    "$stado" storage stat "$destination/$name" >/dev/null 2>&1 ||
        "$stado" storage put "$destination/$name" "$work/$name" >/dev/null
done

/usr/bin/python3 - "$model_name" "$digest" "$destination" <<'PY' >candidate-manifest.json
import json
import os
import sys
from pathlib import Path

name, digest, destination = sys.argv[1:4]
parts = sorted(p.name for p in Path("candidate-parts").iterdir())
manifest = {
    "contract": "oko-goal-lifecycle-v1",
    "qualified": False,
    "stage": "candidate",
    "base_model": os.environ.get("LIFECYCLE_STUDENT_MODEL", "Qwen/Qwen3-4B"),
    "filename": name,
    "sha256": digest,
    "bytes": Path(name).stat().st_size,
    "part_count": len(parts),
    "parts": parts,
    "destination": destination,
    "training_metrics": json.loads(Path("metrics.json").read_text(encoding="utf-8")).get(
        "action_accuracy"
    ),
}
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

"$stado" storage stat "$destination/candidate-manifest.json" >/dev/null 2>&1 ||
    "$stado" storage put "$destination/candidate-manifest.json" candidate-manifest.json >/dev/null

cat candidate-manifest.json

#!/bin/sh
# Convert a trained lifecycle student into the served Q4_K_M GGUF and record
# what it is, so the machine that will serve the model can fetch exactly these
# bytes through training/lifecycle-model/serve-candidate-host.sh.
#
#   stado host run-helper <target> oko-lifecycle-convert.sh --uuid <JOB_UUID>
#
# This host deliberately does not publish: `release_api.publishers` declares
# the `oko/` publisher only on the control-plane machine, and qualification —
# constrained serving evaluation plus the independent judge — happens there too.
set -eu

job="${1:?job uuid required}"
work="/mnt/wisent-training/stado/training/oko-lifecycle-$job"
llama_revision=b6180
llama_root=/mnt/wisent-staging/llama.cpp
model_name=oko-lifecycle-qwen3-4b-q4_k_m.gguf

[ -d "$work/student" ] || { echo "no trained student at $work/student" >&2; exit 1; }
cd "$work"

# mistral_common is an unconditional import of convert_hf_to_gguf.py at this
# llama.cpp revision, even for Qwen architectures.
/usr/local/bin/pip3 install --quiet --break-system-packages gguf sentencepiece protobuf mistral_common
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

/usr/bin/python3 - "$model_name" "$digest" <<'PY' >candidate-manifest.json
import json
import os
import sys
from pathlib import Path

name, digest = sys.argv[1:3]
metrics = json.loads(Path("metrics.json").read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "contract": "oko-goal-lifecycle-v1",
            "qualified": False,
            "stage": "candidate",
            "base_model": os.environ.get("LIFECYCLE_STUDENT_MODEL", "Qwen/Qwen3-4B"),
            "filename": name,
            "sha256": digest,
            "bytes": Path(name).stat().st_size,
            "training_action_accuracy": metrics.get("action_accuracy"),
            "training_joint_accuracy": metrics.get("joint_accuracy"),
            "training_finish_precision": metrics.get("finish_precision"),
        },
        indent=2,
        sort_keys=True,
    )
)
PY

cat candidate-manifest.json

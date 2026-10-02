#!/bin/sh
# Report whether the trained student's output projection survives GGUF
# conversion.
#
#   stado host run-helper <target> oko-lifecycle-inspect-student.sh --uuid <JOB_UUID>
#
# Check the served conversion against the trained output projection. A
# missing or incorrectly tied projection changes the measured function.
set -eu

job="${1:?job uuid required}"
work="/mnt/wisent-training/stado/training/oko-lifecycle-$job"
[ -d "$work/student" ] || { echo "no student directory in $work" >&2; exit 1; }

/usr/bin/python3 - "$work" <<'PY'
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
student = work / "student"
config = json.loads((student / "config.json").read_text(encoding="utf-8"))
print("tie_word_embeddings:", config.get("tie_word_embeddings"))
print("vocab_size:", config.get("vocab_size"), "hidden:", config.get("hidden_size"))

index = student / "model.safetensors.index.json"
names = []
if index.is_file():
    names = sorted(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
else:
    from safetensors import safe_open

    for shard in sorted(student.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            names.extend(handle.keys())
    names = sorted(names)
print("tensors:", len(names))
print("has lm_head.weight:", any(name == "lm_head.weight" for name in names))
print("has embed_tokens:", any(name.endswith("embed_tokens.weight") for name in names))

gguf = work / "student-bf16.gguf"
if gguf.is_file():
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf))
    tensor_names = {tensor.name for tensor in reader.tensors}
    print("gguf tensors:", len(tensor_names))
    print("gguf has output.weight:", "output.weight" in tensor_names)
    print("gguf has token_embd.weight:", "token_embd.weight" in tensor_names)
PY

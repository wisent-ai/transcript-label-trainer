#!/bin/sh
# Report whether this host can convert a trained student into a served GGUF:
# llama.cpp tooling, a C/C++ toolchain, git, and the Python modules the
# converter imports.
set -u

echo "== llama.cpp binaries =="
for binary in llama-server llama-quantize llama-cli; do
    path=$(command -v "$binary" 2>/dev/null || true)
    if [ -n "$path" ]; then
        echo "present $binary $path"
    else
        echo "absent  $binary"
    fi
done

echo
echo "== existing checkouts =="
for candidate in /mnt/wisent-staging/llama.cpp /mnt/wisent-training/llama.cpp "$HOME/llama.cpp"; do
    if [ -d "$candidate/.git" ]; then
        echo "present $candidate"
    else
        echo "absent  $candidate"
    fi
done

echo
echo "== toolchain =="
for tool in git cmake make cc c++ ninja pip3; do
    path=$(command -v "$tool" 2>/dev/null || true)
    if [ -n "$path" ]; then
        echo "present $tool $path"
    else
        echo "absent  $tool"
    fi
done

echo
echo "== converter imports =="
/usr/bin/python3 - <<'PY'
import importlib.util

for module in ("torch", "numpy", "gguf", "sentencepiece", "safetensors", "transformers"):
    print(module, "present" if importlib.util.find_spec(module) else "absent")
PY

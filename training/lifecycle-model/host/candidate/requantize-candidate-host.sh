#!/bin/sh
# Produce higher-precision quantizations of an already-converted candidate and
# serve them for measurement.
#
#   stado host run-helper <target> oko-lifecycle-requantize.sh --uuid <JOB_UUID>
#
# The served quantization must pass the quality gate; training precision
# alone does not qualify the released model.
set -eu

job="${1:?job uuid required}"
work="/mnt/wisent-training/stado/training/oko-lifecycle-$job"
llama_root=/mnt/wisent-staging/llama.cpp

[ -s "$work/student-bf16.gguf" ] || { echo "no converted bf16 GGUF in $work" >&2; exit 1; }
[ -x "$llama_root/build/bin/llama-quantize" ] || { echo "llama-quantize not built" >&2; exit 1; }
cd "$work"

for quant in Q6_K Q8_0; do
    lower=$(printf '%s' "$quant" | tr 'A-Z' 'a-z')
    target="oko-lifecycle-qwen3-4b-$lower.gguf"
    if [ ! -s "$target" ]; then
        "$llama_root/build/bin/llama-quantize" student-bf16.gguf "$target" "$quant" >/dev/null
    fi
    printf '%s %s %s\n' "$target" "$(stat -c %s "$target")" "$(sha256sum "$target" | cut -d' ' -f1)"
done

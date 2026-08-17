#!/bin/sh
# Report whether this host can actually run lifecycle label-model training:
# the declared models directory, real free space, the accelerator, and the
# Python training stack. Written because repository host scripts named
# /mnt/wd16tb/..., a 16 TB disk removed from this host, so their writes
# resolved onto the 100 GiB root volume while every reader believed them.
set -u

echo "== declared training root =="
for path in /mnt/wd16tb /mnt/wd16tb/stado/training /mnt/wd16tb/wisent-staging; do
    if [ -d "$path" ]; then
        echo "present $path"
    else
        echo "absent  $path"
    fi
done

echo
echo "== mounts =="
mount | grep -E ' / | /mnt| /var/lib/docker ' || true

echo
echo "== free space =="
df -BG / /tmp /var/lib/docker 2>/dev/null || true

echo
echo "== accelerator =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
else
    echo "nvidia-smi absent"
fi

echo
echo "== python training stack =="
for python in python3 /usr/bin/python3; do
    command -v "$python" >/dev/null 2>&1 || continue
    "$python" - <<'PY'
import importlib.util
import sys

print("python", sys.version.split()[0], sys.executable)
for module in ("torch", "transformers", "datasets", "peft", "accelerate"):
    spec = importlib.util.find_spec(module)
    print(module, "present" if spec else "absent")
if importlib.util.find_spec("torch"):
    import torch

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
PY
    break
done

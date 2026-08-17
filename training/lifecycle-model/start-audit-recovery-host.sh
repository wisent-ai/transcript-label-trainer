#!/bin/sh
set -eu

runner=/root/.stado/bin/oko-lifecycle-audit-runner

run_audit() {
  model="$1"
  label="$2"
  unit="oko-lifecycle-$label-audit-b5de55bd"
  systemctl reset-failed "$unit.service" 2>/dev/null || true
  systemd-run \
    --unit="$unit" \
    --collect \
    --wait \
    --pipe \
    --setenv="LIFECYCLE_AUDIT_MODEL=$model" \
    --setenv="LIFECYCLE_AUDIT_LABEL=$label" \
    --property=Type=exec \
    --property=TimeoutStartSec=infinity \
    /usr/bin/python3 "$runner"
}

if run_audit -best best; then
  exit 0
fi
run_audit wisent-backend/chat/primary local

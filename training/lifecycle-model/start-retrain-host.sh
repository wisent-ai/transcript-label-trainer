#!/bin/sh
set -eu

unit=oko-lifecycle-retrain-b5de55bd
systemctl reset-failed "$unit.service" 2>/dev/null || true
exec systemd-run \
  --unit="$unit" \
  --collect \
  --wait \
  --pipe \
  --property=Type=exec \
  --property=TimeoutStartSec=infinity \
  /usr/bin/python3 /root/.stado/bin/oko-lifecycle-retrain-runner

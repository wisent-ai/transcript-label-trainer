#!/bin/sh
set -eu

printf '%s\n' '=== worker state ==='
systemctl show wisent-agent.service \
  --property=ActiveState,SubState,MainPID,ExecMainStatus,ActiveEnterTimestamp \
  --no-pager

printf '%s\n' '=== target queue record ==='
environment="$(systemctl show wisent-agent.service --property=Environment --value)"
set -a
. /root/.stado/files/stado-agent-grant.env
set +a
/usr/bin/env -S "$environment" /root/.stado/bin/stado status 31645abc || true

printf '%s\n' '=== recent worker journal ==='
journalctl -u wisent-agent.service --no-pager -n 80 -o cat \
  | /usr/bin/cut -c 1-700

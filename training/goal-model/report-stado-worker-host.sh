#!/bin/sh
set -eu


printf '%s\n' '=== wisent-agent unit directives ==='
systemctl cat wisent-agent.service --no-pager \
  | while IFS= read -r line; do
      case "$line" in
        User=*|Group=*|WorkingDirectory=*|ExecStart=*|EnvironmentFile=*|Restart=*|RestartSec=*)
          printf '%s\n' "$line"
          ;;
      esac
    done

printf '%s\n' '=== stado version ==='
/root/.stado/bin/stado --version
printf '%s\n' '=== worker environment keys ==='
while IFS='=' read -r key _; do
  case "$key" in
    ''|'#'*) ;;
    *) printf '%s\n' "$key" ;;
  esac
done </root/.stado/files/stado-agent-grant.env

printf '%s\n' '=== recent worker journal ==='
journalctl -u wisent-agent.service --no-pager -n 30 -o cat | cut -c 1-500

printf '%s\n' '=== gpu state ==='
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

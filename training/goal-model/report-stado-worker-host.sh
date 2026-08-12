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

printf '%s\n' '=== gpu compute processes ==='
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits

printf '%s\n' '=== gpu process ancestry ==='
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do
  current=$pid
  depth=0
  while [ "$current" -gt 1 ] 2>/dev/null && [ "$depth" -lt 8 ]; do
    ps -p "$current" -o user= -o pid= -o ppid= -o etime= -o comm= -o args= \
      | cut -c 1-500
    next=$(ps -p "$current" -o ppid= | tr -d ' ')
    [ -n "$next" ] || break
    current=$next
    depth=$((depth + 1))
  done
  printf '%s\n' '-- cgroup --'
  cat "/proc/$pid/cgroup"
  scope=$(cat "/proc/$pid/cgroup")
  container=${scope#*docker-}
  container=${container%.scope}
  if [ "$container" != "$scope" ]; then
    printf '%s\n' '-- container --'
    docker inspect --format \
      'name={{.Name}} image={{.Config.Image}} created={{.Created}} restart={{.HostConfig.RestartPolicy.Name}} labels={{json .Config.Labels}}' \
      "$container"
  fi
done

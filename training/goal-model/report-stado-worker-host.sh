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
sha256sum /root/.stado/bin/stado
printf '%s\n' '=== worker environment keys ==='
while IFS='=' read -r key _; do
  case "$key" in
    ''|'#'*) ;;
    *) printf '%s\n' "$key" ;;
  esac
done </root/.stado/files/stado-agent-grant.env

printf '%s\n' '=== worker service environment ==='
environment=$(systemctl show wisent-agent.service --property=Environment --value)
for assignment in $environment; do
  case "${assignment%%=*}" in
    *TOKEN_FILE) printf '%s\n' "$assignment" ;;
    *TOKEN*|*SECRET*|*PASSWORD*) ;;
    *) printf '%s\n' "$assignment" ;;
  esac
done
token_file=$(printf '%s\n' "$environment" | tr ' ' '\n' | sed -n 's/^WC_STADO_STORAGE_TOKEN_FILE=//p')
if [ -n "$token_file" ] && [ -f "$token_file" ]; then
  sha256sum "$token_file"
fi

printf '%s\n' '=== inference reservation ==='
if [ -f /root/.stado/inference/reservation.json ]; then
  cat /root/.stado/inference/reservation.json
else
  printf '%s\n' 'missing'
fi

printf '%s\n' '=== worker-visible target job ==='
set -a
. /root/.stado/files/stado-agent-grant.env
set +a
/usr/bin/env -S "$environment" /root/.stado/bin/stado status 95947927 || true
/usr/bin/env -S "$environment" /root/.stado/bin/stado storage stat queue/95947927.json --json || true
/usr/bin/env -S "$environment" /root/.stado/bin/stado storage ls queue/ --json --limit 20 || true
job_file=/tmp/stado-report-95947927.json
/usr/bin/env -S "$environment" /root/.stado/bin/stado storage get queue/95947927.json "$job_file" || true
if [ -f "$job_file" ]; then
  jq '{job_id,state,gpu_mem_gb,gpu_type,provider,pin_to_provider,priority,pinned_host,assigned_to,exclusive,preemptible,max_cost_per_hour_usd,machine_type,created_at,command}' "$job_file"
  rm -f "$job_file"
fi

printf '%s\n' '=== configured Vast machine ==='
/usr/bin/env -S "$environment" /root/.stado/bin/stado vast status || true

printf '%s\n' '=== recent worker journal ==='
journalctl -u wisent-agent.service --no-pager -n 30 -o cat | cut -c 1-500

printf '%s\n' '=== worker process tree ==='
worker_pid=$(systemctl show wisent-agent.service --property=MainPID --value)
ps -p "$worker_pid" -o user= -o pid= -o ppid= -o etime= -o comm= -o args= | cut -c 1-500
ps --ppid "$worker_pid" -o user= -o pid= -o ppid= -o etime= -o comm= -o args= | cut -c 1-500

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

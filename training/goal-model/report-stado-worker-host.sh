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
printf '%s\n' '=== worker credential routes ==='
/usr/bin/sed -n \
  -e '/^WC_AGENT_SKARBIEC_URL=/p' \
  -e '/^WC_AGENT_SKARBIEC_CONSUMER=/p' \
  -e '/^WC_AGENT_SKARBIEC_ITEMS=/p' \
  -e '/^WC_AGENT_SKARBIEC_SECRET_FIELDS=/p' \
  /root/.stado/files/stado-agent-grant.env
agent_token_file=$(/usr/bin/sed -n 's/^WC_AGENT_SKARBIEC_TOKEN_FILE=//p' \
  /root/.stado/files/stado-agent-grant.env)
[ -n "$agent_token_file" ] && /usr/bin/sha256sum "$agent_token_file"
printf '%s\n' '=== worker credential reads ==='
/usr/bin/python3 - <<'PY'
import json
import urllib.error
import urllib.request

values = {}
with open("/root/.stado/files/stado-agent-grant.env", encoding="utf-8") as handle:
    for raw in handle:
        key, separator, value = raw.rstrip("\n").partition("=")
        if separator:
            values[key] = value

url = values["WC_AGENT_SKARBIEC_URL"].rstrip("/") + "/v1/items/read"
token = open(values["WC_AGENT_SKARBIEC_TOKEN_FILE"], encoding="utf-8").read().strip()
for item, field in (
    ("jeden-model-router", "token"),
    ("jeden-agent-auth", "agent_auth_secret"),
):
    request = urllib.request.Request(
        url,
        data=json.dumps({"id": item, "field": field}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Consumer": values["WC_AGENT_SKARBIEC_CONSUMER"],
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = json.load(response)
            value = body.get("value")
            status = response.status
    except urllib.error.HTTPError as error:
        value = None
        status = error.code
    print(json.dumps({
        "item": item,
        "field": field,
        "status": status,
        "value_type": type(value).__name__,
        "value_length": len(value) if isinstance(value, str) else None,
    }, separators=(",", ":")))
PY

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
/usr/bin/env -S "$environment" STADO_API_TOKEN= STADO_API_TOKEN_FILE="$token_file" \
  /root/.stado/bin/stado storage get queue/95947927.json "$job_file" || true
if [ -f "$job_file" ]; then
  jq '{job_id,state,gpu_mem_gb,gpu_type,provider,pin_to_provider,priority,pinned_host,assigned_to,exclusive,preemptible,max_cost_per_hour_usd,machine_type,created_at,command}' "$job_file"
  rm -f "$job_file"
fi
capacity_file=/tmp/stado-report-capacity.json
/usr/bin/env -S "$environment" STADO_API_TOKEN= STADO_API_TOKEN_FILE="$token_file" \
  /root/.stado/bin/stado storage get capacity/local-ubuntu-server.json "$capacity_file" || true
if [ -f "$capacity_file" ]; then
  jq '{consumer_id,published_at,free_slots,free_vram_gb,total_vram_gb,diag}' "$capacity_file"
  rm -f "$capacity_file"
fi

printf '%s\n' '=== configured Vast machine ==='
/usr/bin/env -S "$environment" /root/.stado/bin/stado vast status || true

printf '%s\n' '=== recent worker journal ==='
journalctl -u wisent-agent.service --no-pager -n 30 -o cat | cut -c 1-500

printf '%s\n' '=== worker process tree ==='
worker_pid=$(systemctl show wisent-agent.service --property=MainPID --value)
ps -p "$worker_pid" -o user= -o pid= -o ppid= -o etime= -o comm= -o args= | cut -c 1-500
ps --ppid "$worker_pid" -o user= -o pid= -o ppid= -o etime= -o comm= -o args= | cut -c 1-500

printf '%s\n' '=== root filesystem consumers ==='
du -x -k --max-depth=3 /root/.cache /root/.stado /root/.local /var /opt 2>/dev/null \
  | sort -nr \
  | sed -n '1,50p'

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

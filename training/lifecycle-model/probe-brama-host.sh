#!/bin/sh
# Prove this host can reach Brama before a training job spends hours only to
# fail at its audit step, which is the last thing run.sh does.
#
#   stado host run-helper <target> oko-lifecycle-probe-brama.sh
#
# Every address tried here is one the fleet declares for this machine: the
# forward marker Stado wrote, the directory endpoint, or a `service_resolver`
# adapter this target is told to dial. A public URL is never assumed — on
# 2026-08-17 the public hop was the thing that broke.
set -eu

stado="$HOME/.stado/bin/stado"
marker="$HOME/.stado/forwards/brama.url"

candidates=""
[ -n "${BRAMA_URL:-}" ] && candidates="$BRAMA_URL"
if [ -s "$marker" ]; then
    candidates="$candidates $(tr -d '\n' <"$marker")"
fi
if [ -x "$stado" ]; then
    declared=$("$stado" registry pull 2>/dev/null | /usr/bin/python3 -c '
import json, socket, sys

try:
    document = json.load(sys.stdin)
except Exception:
    sys.exit(0)
me = socket.gethostname().split(".")[0].casefold()
for target in document.get("targets", []):
    names = {str(target.get("name", "")).casefold()}
    names.update(str(name).casefold() for name in target.get("hostnames", []))
    if me not in {name.split(".")[0] for name in names}:
        continue
    resolver = target.get("service_resolver") or {}
    for adapter in resolver.get("adapters", []):
        if adapter.get("service") == "brama" and adapter.get("bind"):
            print("http://" + adapter["bind"])
    endpoint = ((target.get("services_endpoints") or {}).get("brama") or {}).get("url")
    if endpoint:
        print(endpoint)
' 2>/dev/null) || declared=""
    candidates="$candidates $declared"
fi

[ -n "$(printf '%s' "$candidates" | tr -d ' ')" ] || {
    echo "no declared Brama address for this host" >&2
    exit 1
}

for url in $candidates; do
    code=$(curl -sS -o /dev/null -m 15 -w '%{http_code}' "$url/healthz" 2>/dev/null || echo 000)
    printf 'tried %s -> %s\n' "$url" "$code"
    case "$code" in
        200 | 401 | 403)
            printf 'brama reachable at %s\n' "$url"
            exit 0
            ;;
    esac
done

echo "Brama is not reachable from this host through any declared address" >&2
exit 1

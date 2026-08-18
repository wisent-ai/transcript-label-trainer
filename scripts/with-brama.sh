#!/bin/sh
# Run the trainer with Brama resolved the way the fleet declares it, then hand
# every argument to the release binary.
#
#   scripts/with-brama.sh lifecycle-review input.jsonl --output out.jsonl --split train
#   scripts/with-brama.sh lifecycle-audit predictions.jsonl --output judge.json
#
# Brama is placed on one host and binds loopback only, so a caller elsewhere
# reaches it through a Stado forward whose address is written to
# ~/.stado/forwards/brama.url. Reading that marker is why this script exists:
# on 2026-08-17 the public hop (brama.wisent.com -> Vercel -> the placed host's
# Tailscale name) stopped terminating TLS, every model call in the fleet failed
# with ROUTER_EXTERNAL_TARGET_HANDSHAKE_ERROR, and a scratch script that had the
# public URL hardcoded could not be pointed anywhere else.
set -eu

# Skarbiec decrypts with gpg, so PATH must carry Homebrew's bin whatever
# environment invoked this script.
PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PATH

root=$(cd "$(dirname "$0")/.." && pwd)
stado="${STADO_BIN:-$HOME/.stado/bin/stado}"
skarbiec="${SKARBIEC_BIN:-$HOME/.stado/bin/skarbiec}"
marker="$HOME/.stado/forwards/brama.url"
binary="$root/target/release/transcript-label-trainer"

SKARBIEC_VAULT_FILE="${SKARBIEC_VAULT_FILE:-$HOME/.stado/skarbiec.vault.json}"
export SKARBIEC_VAULT_FILE

[ -x "$binary" ] || { echo "build first: cargo build --locked --release" >&2; exit 1; }
[ -x "$stado" ] || { echo "stado CLI absent at $stado" >&2; exit 1; }
[ -x "$skarbiec" ] || { echo "skarbiec CLI absent at $skarbiec" >&2; exit 1; }

field() {
    "$skarbiec" get "$1" 2>/dev/null | FIELD="$2" /usr/bin/python3 -c 'import json, os, sys
try:
    print(json.load(sys.stdin)["fields"][os.environ["FIELD"]])
except Exception:
    sys.exit(1)'
}

placed_host() {
    "$stado" service directory endpoint brama --json |
        /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["active_host"])'
}

open_forward() {
    "$stado" host forward-remote "$(placed_host)" brama \
        --remote-port 8080 --local-port 18080 >/dev/null
    printf 'http://127.0.0.1:18080'
}

answers() {
    code=$(/usr/bin/curl -sS -o /dev/null -m 20 -w '%{http_code}' "$1/healthz" 2>/dev/null || echo 000)
    case "$code" in
        200 | 401 | 403) return 0 ;;
        *) return 1 ;;
    esac
}

if [ -n "${BRAMA_URL:-}" ]; then
    url="$BRAMA_URL"
elif [ -s "$marker" ]; then
    url=$(tr -d '\n' <"$marker")
else
    url=$(open_forward)
fi

# The forward is an SSH channel and it does drop; a stale marker is not a
# reason to fail a run that can reopen it.
if ! answers "$url"; then
    url=$(open_forward)
    answers "$url" || {
        echo "Brama did not answer at $url after reopening the declared forward" >&2
        exit 1
    }
fi
BRAMA_URL="$url"
export BRAMA_URL
if [ -z "${BRAMA_TOKEN:-}" ]; then
    BRAMA_TOKEN=$(field jeden-model-router token) || {
        echo "could not read jeden-model-router.token from $SKARBIEC_VAULT_FILE" >&2
        exit 1
    }
fi
export BRAMA_TOKEN
if [ -z "${WISENT_APP_AGENT_AUTH_SECRET:-}" ]; then
    WISENT_APP_AGENT_AUTH_SECRET=$(field jeden-agent-auth agent_auth_secret) || {
        echo "could not read jeden-agent-auth.agent_auth_secret from $SKARBIEC_VAULT_FILE" >&2
        exit 1
    }
fi
export WISENT_APP_AGENT_AUTH_SECRET
WISENT_APP_AGENT_ID="${WISENT_APP_AGENT_ID:-wisent-app}"
export WISENT_APP_AGENT_ID
LIFECYCLE_REVIEW_WORKERS="${LIFECYCLE_REVIEW_WORKERS:-6}"
export LIFECYCLE_REVIEW_WORKERS

exec "$binary" "$@"

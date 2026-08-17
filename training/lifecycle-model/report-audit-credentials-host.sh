#!/bin/sh
set -eu

stado="$HOME/.stado/bin/stado"
check() {
  item="$1"
  field="$2"
  if value="$("$stado" credentials get "$item" --field "$field" 2>/dev/null)" && [ -n "$value" ]; then
    printf '%s/%s: ready\n' "$item" "$field"
  else
    printf '%s/%s: unavailable\n' "$item" "$field"
  fi
  value=''
}
check jeden-model-router token
check jeden-agent-auth agent_auth_secret

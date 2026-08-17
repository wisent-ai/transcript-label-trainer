#!/bin/sh
set -eu

environment="$(systemctl show wisent-agent.service --property=Environment --value)"
set -a
. /root/.stado/files/stado-agent-grant.env
set +a
exec /usr/bin/env -S "$environment" \
  /usr/bin/python3 /root/.stado/bin/oko-lifecycle-publish-recovered

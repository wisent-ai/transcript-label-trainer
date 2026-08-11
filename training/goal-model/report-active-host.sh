#!/bin/sh
set -eu

work=$(
  /usr/bin/find /tmp -maxdepth 1 -type d -name 'wc-*' -printf '%T@ %p\n' \
    | /usr/bin/sort -nr \
    | /usr/bin/awk 'NR == 1 { print $2 }'
)
[ -n "$work" ]
printf 'workdir=%s\n' "$work"
/usr/bin/find "$work" -maxdepth 3 -type f -printf '%p %s bytes %TY-%Tm-%TdT%TH:%TM:%TS\n' \
  | /usr/bin/sort
if [ -f "$work/output/command_output.log" ]; then
  printf '%s\n' '--- command output ---'
  /usr/bin/tail -n 80 "$work/output/command_output.log"
fi

#!/bin/sh
# Retire the lifecycle training unit and the superseded job directories once a
# candidate has been published. The unit is enabled at boot so an interrupted
# run resumes after power loss; after the release ships, a reboot would
# otherwise re-run a finished training for nothing.
#
#   stado host run-helper <target> oko-lifecycle-train-cleanup.sh --uuid <KEEP_JOB_UUID>
#
# The one argument is the job whose work directory is KEPT (the published
# candidate's provenance); every other oko-lifecycle-* work directory goes.
set -eu

keep="${1:?job uuid to keep required}"
root=/mnt/wisent-training/stado/training
unit=oko-lifecycle-train.service

systemctl disable --now "$unit" 2>/dev/null || true
rm -f "/etc/systemd/system/$unit"
systemctl daemon-reload
printf 'unit %s: disabled and removed\n' "$unit"

for dir in "$root"/oko-lifecycle-*; do
    [ -d "$dir" ] || continue
    case "$dir" in
        *"$keep"*) printf 'kept    %s (%s)\n' "$dir" "$(du -sh "$dir" | cut -f1)" ;;
        *)
            size=$(du -sh "$dir" | cut -f1)
            rm -rf "$dir"
            printf 'removed %s (%s)\n' "$dir" "$size"
            ;;
    esac
done
df -h "$root" | tail -1

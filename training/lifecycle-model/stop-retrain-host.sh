#!/bin/sh
set -eu
systemctl stop oko-lifecycle-retrain-b5de55bd.service
systemctl reset-failed oko-lifecycle-retrain-b5de55bd.service 2>/dev/null || true
printf '%s\n' 'stopped lifecycle retraining unit'

#!/bin/bash
# Deprecated wrapper — use scripts/submit_all.sh instead.
exec "$(dirname "$0")/submit_all.sh" "$@"

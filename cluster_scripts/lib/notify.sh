#!/bin/bash
# Shared ntfy.sh notification helper — source this file in PBS job scripts.
#
# The ntfy topic is kept OUT of version control. Provide it via either:
#   - environment variable NTFY_TOPIC, or
#   - an untracked file next to this one: cluster_scripts/lib/ntfy_topic.local
# If neither is set, ntfy_done is a no-op (notifications are simply skipped).

_ntfy_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NTFY_TOPIC="${NTFY_TOPIC:-$(cat "${_ntfy_dir}/ntfy_topic.local" 2>/dev/null)}"

ntfy_done() {
    local title="$1"
    local msg="${2:-$1}"
    [ -n "$NTFY_TOPIC" ] || return 0
    curl -s \
        -H "Title: $title" \
        -H "Priority: default" \
        -d "$msg" \
        "https://ntfy.sh/${NTFY_TOPIC}" || true
}

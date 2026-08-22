#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
unit_name="dlm-wonn-phase5-50k"
memory_high="${DLM_WONN_MEMORY_HIGH:-10G}"
memory_max="${DLM_WONN_MEMORY_MAX:-12G}"
pipeline="$repo_root/scripts/run_phase5_50k_pipeline.sh"

usage() {
    echo "usage: $0 --dry-run|--confirm" >&2
}

if [ "$#" -ne 1 ]; then
    usage
    exit 2
fi

command=(
    systemd-run --user
    "--unit=$unit_name"
    "--description=DLM-WONN WMT14 WONN pilot and gated 50K vs legacy ELF"
    "--property=MemoryHigh=$memory_high"
    "--property=MemoryMax=$memory_max"
    "--property=CPUWeight=50"
    "--property=IOWeight=50"
    "--property=Nice=5"
    "--property=TasksMax=64"
    "--property=OOMPolicy=stop"
    "--working-directory=$repo_root"
    /usr/bin/bash "$pipeline"
)

case "$1" in
    --dry-run)
        printf 'memory_high=%s\nmemory_max=%s\nunit=%s\ncommand=' \
            "$memory_high" "$memory_max" "$unit_name"
        printf '%q ' "${command[@]}"
        printf '\n'
        ;;
    --confirm)
        if systemctl --user is-active --quiet "$unit_name.service"; then
            echo "$unit_name.service is already active" >&2
            exit 3
        fi
        "${command[@]}"
        echo "started $unit_name.service"
        echo "status: systemctl --user status $unit_name.service"
        echo "logs:   journalctl --user -fu $unit_name.service"
        ;;
    *)
        usage
        exit 2
        ;;
esac

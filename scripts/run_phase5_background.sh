#!/usr/bin/env bash
set -u

if [ "$#" -ne 3 ]; then
    echo "usage: $0 CONFIG OUTPUT_DIR LABEL" >&2
    exit 2
fi

config_path="$1"
output_dir="$2"
label="$3"
repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
python_bin="${DLM_WONN_PYTHON:-$repo_root/.venv/bin/python}"
log_path="$repo_root/$output_dir/systemd.log"

cd "$repo_root" || exit 2
repo_status="$(git status --porcelain --untracked-files=all)"
if [ -n "$repo_status" ]; then
    echo "refusing to start Phase 5 training from a dirty worktree" >&2
    printf '%s\n' "$repo_status" >&2
    exit 3
fi

mkdir -p "$repo_root/$output_dir"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

started_at="$(date --iso-8601=seconds)"
printf '%s\n' "$started_at" > "$output_dir/systemd.started"
git rev-parse HEAD > "$output_dir/source_commit"
printf '%s\n' "$config_path" > "$output_dir/source_config"
printf '%q ' "$python_bin" src/train.py --config "$config_path" > "$output_dir/command.txt"
printf '\n' >> "$output_dir/command.txt"
"$python_bin" src/train.py --config "$config_path" >> "$log_path" 2>&1
status=$?
finished_at="$(date --iso-8601=seconds)"

if [ "$status" -eq 0 ]; then
    printf '%s\n' "$finished_at" > "$output_dir/systemd.complete"
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "DLM-WONN Phase 5" "$label completed successfully" || true
    fi
else
    printf '%s status=%s\n' "$finished_at" "$status" > "$output_dir/systemd.failed"
    if command -v notify-send >/dev/null 2>&1; then
        notify-send -u critical "DLM-WONN Phase 5" "$label failed with status $status" || true
    fi
fi

exit "$status"

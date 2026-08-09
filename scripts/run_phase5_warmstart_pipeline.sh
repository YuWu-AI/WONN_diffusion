#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
python_bin="/home/yuwu/Research/diffusion_lm/DLM_WONN/.venv/bin/python"
python_include="/home/yuwu/miniconda3/envs/WONN/include/python3.10"
config_path="src/configs/training_configs/train_de-en_ELF-WONN-B-phase5-warmstart-hwopt.yml"
baseline_config_path="src/configs/training_configs/train_de-en_ELF-B-phase5-pilot.yml"
baseline_checkpoint="outputs/phase5/pilot/elf_b/checkpoint_10000"
output_dir="outputs/phase5/warmstart_hwopt/wonn_l6t8_b12"
label="WONN warm-start 15k pipeline"
log_path="$repo_root/$output_dir/systemd.log"

cd "$repo_root" || exit 2
repo_status="$(git status --porcelain --untracked-files=all)"
if [ -n "$repo_status" ]; then
    echo "refusing to start Phase 5 training from a dirty worktree" >&2
    printf '%s\n' "$repo_status" >&2
    exit 3
fi
if [ -e "$output_dir/systemd.started" ] || [ -e "$output_dir/training_started.json" ]; then
    echo "refusing to overwrite an existing formal Phase 5 run" >&2
    exit 4
fi
if [ ! -s "$baseline_checkpoint" ]; then
    echo "missing official ELF-B baseline checkpoint: $baseline_checkpoint" >&2
    exit 5
fi

mkdir -p "$output_dir"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
if [ -n "${CPATH:-}" ]; then
    export CPATH="$python_include:$CPATH"
else
    export CPATH="$python_include"
fi

started_at="$(date --iso-8601=seconds)"
printf '%s\n' "$started_at" > "$output_dir/systemd.started"
git rev-parse HEAD > "$output_dir/source_commit"
printf '%s\n' "$config_path" > "$output_dir/source_config"
printf '%s\n' "$baseline_config_path" > "$output_dir/baseline_source_config"
printf '%s\n' "$baseline_checkpoint" > "$output_dir/baseline_source_checkpoint"
printf '%q ' "$0" > "$output_dir/command.txt"
printf '\n' >> "$output_dir/command.txt"

"$python_bin" src/train.py --config "$config_path" >> "$log_path" 2>&1
status=$?

if [ "$status" -eq 0 ] && [ ! -s "$output_dir/training_complete.json" ]; then
    echo "training exited successfully without training_complete.json" >> "$log_path"
    status=5
fi

for step in 2000 5000 10000 15000; do
    if [ "$status" -ne 0 ]; then
        break
    fi
    eval_dir="$output_dir/evaluations/checkpoint_$step"
    "$python_bin" src/eval.py \
        --config "$config_path" \
        --config_override "output_dir=$eval_dir" \
        --checkpoint_path "$output_dir/checkpoint_$step" \
        --seed 42 >> "$log_path" 2>&1
    status=$?
done

if [ "$status" -eq 0 ]; then
    baseline_eval_dir="$output_dir/comparisons/elf_b_checkpoint_10000"
    "$python_bin" src/eval.py \
        --config "$baseline_config_path" \
        --config_override "output_dir=$baseline_eval_dir" \
        --config_override batch_size=12 \
        --checkpoint_path "$baseline_checkpoint" \
        --seed 42 >> "$log_path" 2>&1
    status=$?
fi

if [ "$status" -eq 0 ]; then
    "$python_bin" scripts/summarize_phase5_run.py "$output_dir" \
        --expected-steps 2000,5000,10000,15000 \
        --batch-size 12 \
        --expected-samples 1000 \
        --warmstart-training-samples 40000 \
        --baseline-eval-dir "$output_dir/comparisons/elf_b_checkpoint_10000" \
        --baseline-checkpoint "$baseline_checkpoint" \
        --baseline-step 10000 \
        --baseline-training-samples 40000 \
        --verify-checkpoints >> "$log_path" 2>&1
    status=$?
fi

finished_at="$(date --iso-8601=seconds)"
if [ "$status" -eq 0 ]; then
    printf '%s\n' "$finished_at" > "$output_dir/systemd.complete"
    notify-send "DLM-WONN Phase 5" "$label completed successfully"
else
    printf '%s status=%s\n' "$finished_at" "$status" > "$output_dir/systemd.failed"
    notify-send -u critical "DLM-WONN Phase 5" "$label failed with status $status"
fi

exit "$status"

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
common_git_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir)"
main_checkout="$(dirname "$common_git_dir")"
python_bin="${DLM_WONN_PYTHON:-$main_checkout/.venv/bin/python}"
pipeline_checks=("$python_bin" scripts/phase5_pipeline_checks.py)
run_root="outputs/phase5/wonn_runs/mechanism_0_50k"
data_root="data/phase5_mechanism"
config_root="src/configs/training_configs/phase5_mechanism"
checkpoint_steps=(5000 10000 20000 30000 40000 50000)
full_eval_steps=(20000 40000 50000)

labels=(S-Base S-Token S-Token-Contrast)
configs=(
    "$config_root/train_de-en-WONN-S-base-50k.yml"
    "$config_root/train_de-en-WONN-S-token-50k.yml"
    "$config_root/train_de-en-WONN-S-token-contrast-50k.yml"
)
run_dirs=(
    "$run_root/s_base"
    "$run_root/s_token"
    "$run_root/s_token_contrast"
)

cd "$repo_root"

if [ "$(git branch --show-current)" != "phase5-wmt14" ]; then
    echo "refusing to run outside the phase5-wmt14 branch" >&2
    exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
    echo "refusing to start a mechanism run from a dirty worktree" >&2
    git status --short >&2
    exit 3
fi
if [ ! -x "$python_bin" ]; then
    echo "missing Python environment: $python_bin" >&2
    exit 4
fi
for config_path in "${configs[@]}"; do
    if [ ! -f "$config_path" ]; then
        echo "missing mechanism config: $config_path" >&2
        exit 5
    fi
done

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
export MAX_JOBS="${MAX_JOBS:-2}"

if [ ! -s "$data_root/manifest.json" ]; then
    "$python_bin" scripts/build_phase5_mechanism_subset.py --output-root "$data_root"
fi
if [ ! -d "$data_root/train" ] || [ ! -d "$data_root/heldout" ]; then
    echo "mechanism dataset is missing train or heldout split" >&2
    exit 6
fi
"$python_bin" -c '
import json,sys
payload=json.load(open(sys.argv[1]))
if payload.get("status") != "complete" or payload.get("train_size") != 50000 or payload.get("heldout_size") != 2000:
    raise SystemExit("invalid mechanism data manifest")
' "$data_root/manifest.json"

mkdir -p "$run_root"
exec 9>"$run_root/pipeline.lock"
if ! flock -n 9; then
    echo "another mechanism pipeline holds $run_root/pipeline.lock" >&2
    exit 7
fi

runner_commit="$(git rev-parse HEAD)"
manifest_path="$run_root/experiment_manifest.json"
if [ ! -s "$manifest_path" ]; then
    manifest_command=(
        "$python_bin" scripts/capture_phase5_manifest.py
        --repo-root "$repo_root"
        --output "$manifest_path"
    )
    for config_path in "${configs[@]}"; do
        manifest_command+=(--config "$config_path")
    done
    "${manifest_command[@]}"
fi
experiment_commit="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$manifest_path")"
if [ "$experiment_commit" != "$runner_commit" ]; then
    if ! git merge-base --is-ancestor "$experiment_commit" "$runner_commit"; then
        echo "experiment commit $experiment_commit is not an ancestor of runner $runner_commit" >&2
        exit 8
    fi
    while IFS= read -r changed_path; do
        case "$changed_path" in
            scripts/run_phase5_mechanism_pipeline.sh|scripts/phase5_pipeline_checks.py|tests/test_phase5_mechanism_pipeline.py)
                ;;
            *)
                echo "refusing to resume: non-pipeline file changed since experiment commit: $changed_path" >&2
                exit 8
                ;;
        esac
    done < <(git diff --name-only "$experiment_commit" "$runner_commit" --)
fi

if [ ! -s "$run_root/pipeline_started.json" ]; then
    "$python_bin" -c '
import json,sys
from datetime import datetime,timezone
json.dump({"status":"running","started_at":datetime.now(timezone.utc).isoformat(),"source_commit":sys.argv[2]},open(sys.argv[1],"w"),indent=2)
' "$run_root/pipeline_started.json" "$experiment_commit"
else
    recorded_commit="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$run_root/pipeline_started.json")"
    if [ "$recorded_commit" != "$experiment_commit" ]; then
        echo "existing run uses commit $recorded_commit, manifest uses $experiment_commit" >&2
        exit 9
    fi
fi

pipeline_status="$run_root/pipeline_status.json"
pipeline_stage="initializing"
pipeline_completed=0

write_pipeline_status() {
    local status="$1"
    local exit_code="${2:-}"
    local command=(
        "${pipeline_checks[@]}" status "$pipeline_status" "$status"
        "$experiment_commit" "$runner_commit" "$pipeline_stage" "$$"
    )
    if [ -n "$exit_code" ]; then
        command+=(--exit-code "$exit_code")
    fi
    "${command[@]}"
}

set_stage() {
    pipeline_stage="$1"
    write_pipeline_status running
}

record_pipeline_exit() {
    local status=$?
    trap - EXIT
    set +e
    if [ "$status" -ne 0 ] && [ "$pipeline_completed" -ne 1 ]; then
        write_pipeline_status failed "$status"
    fi
    exit "$status"
}
trap record_pipeline_exit EXIT
write_pipeline_status running

run_logged_with_retries() {
    local attempts="$1"
    local description="$2"
    local log_path="$3"
    shift 3

    local attempt status=1
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if "$@" >> "$log_path" 2>&1; then
            return 0
        else
            status=$?
        fi
        printf '%s failed (attempt %d/%d, status=%d)\n' \
            "$description" "$attempt" "$attempts" "$status" \
            | tee -a "$log_path" >&2
        if [ "$attempt" -lt "$attempts" ]; then
            sleep "$((attempt * 5))"
        fi
    done
    return "$status"
}

training_is_complete() {
    local completion="$1"
    [ -s "$completion" ] && "$python_bin" -c '
import json,sys
payload=json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("status") == "complete" and payload.get("completed_optimizer_step") == 50000 else 1)
' "$completion"
}

run_training_attempt() {
    local config_path="$1"
    local output_dir="$2"
    /usr/bin/time -v -o "$output_dir/process_resources.txt" \
        "$python_bin" src/train.py --config "$config_path"
    training_is_complete "$output_dir/training_complete.json"
}

run_training() {
    local label="$1"
    local config_path="$2"
    local output_dir="$3"
    local completion="$output_dir/training_complete.json"
    mkdir -p "$output_dir"
    if training_is_complete "$completion"; then
        echo "$label training already complete"
        return
    fi
    printf '%s\n' "$experiment_commit" > "$output_dir/source_commit"
    printf '%s\n' "$runner_commit" > "$output_dir/runner_commit"
    printf '%s\n' "$config_path" > "$output_dir/source_config"
    sha256sum "$config_path" > "$output_dir/source_config.sha256"
    echo "training $label to 50k"
    if ! run_logged_with_retries 3 "$label training" "$output_dir/pipeline.log" \
        run_training_attempt "$config_path" "$output_dir"; then
        echo "$label training failed or did not reach optimizer step 50000" >&2
        return 10
    fi
}

validate_checkpoints() {
    local label="$1"
    local output_dir="$2"
    local step checkpoint
    for step in "${checkpoint_steps[@]}"; do
        checkpoint="$output_dir/checkpoint_$step"
        if ! "${pipeline_checks[@]}" checkpoint "$checkpoint"; then
            echo "$label checkpoint is missing or invalid: $checkpoint" >&2
            return 11
        fi
    done
}

run_evaluation_attempt() {
    local label="$1"
    local config_path="$2"
    local checkpoint="$3"
    local temporary="$4"
    local step="$5"
    local samples="$6"
    "$python_bin" src/eval.py \
        --config "$config_path" \
        --config_override "output_dir=$temporary" \
        --config_override "num_samples=$samples" \
        --config_override "batch_size=16" \
        --config_override "online_eval=true" \
        --checkpoint_path "$checkpoint" \
        --seed 42
    "${pipeline_checks[@]}" finalize-evaluation \
        "$temporary" "$label" "$step" "$samples"
}

evaluate_checkpoint() {
    local label="$1"
    local config_path="$2"
    local output_dir="$3"
    local step="$4"
    local samples="$5"
    local checkpoint="$output_dir/checkpoint_$step"
    local eval_dir="$output_dir/evaluations/checkpoint_$step"
    local temporary="${eval_dir}.in_progress"
    if "${pipeline_checks[@]}" evaluation "$eval_dir" "$label" "$step" "$samples"; then
        echo "$label checkpoint $step full evaluation already complete"
        return
    fi
    if ! "${pipeline_checks[@]}" checkpoint "$checkpoint"; then
        echo "missing or invalid checkpoint: $checkpoint" >&2
        return 12
    fi
    if [ -e "$eval_dir" ]; then
        local preserved="${eval_dir}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
        echo "preserving incomplete evaluation as $preserved"
        mv "$eval_dir" "$preserved"
    fi
    if [ -e "$temporary" ]; then
        local preserved="${temporary}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
        echo "preserving interrupted evaluation as $preserved"
        mv "$temporary" "$preserved"
    fi
    mkdir -p "$temporary"
    if ! run_logged_with_retries 3 "$label checkpoint $step evaluation" \
        "$output_dir/pipeline.log" run_evaluation_attempt \
        "$label" "$config_path" "$checkpoint" "$temporary" "$step" "$samples"; then
        echo "$label checkpoint $step evaluation failed; artifacts retained in $temporary" >&2
        return 13
    fi
    mv "$temporary" "$eval_dir"
}

run_diagnostic_attempt() {
    local label="$1"
    local config_path="$2"
    local checkpoint="$3"
    local diagnostic_dir="$4"
    local samples="$5"
    local dataset_path="$6"
    "$python_bin" scripts/diagnose_phase5_checkpoint.py \
        --root . \
        --output-dir "$diagnostic_dir" \
        --num-samples "$samples" \
        --batch-size 16 \
        --seed 42 \
        --model-spec "$label|$config_path|$checkpoint" \
        --dataset-path "$dataset_path" \
        --dataset-revision "aee21115965ee2b9d3d6d02ec2a72f4f998476d9" \
        --t-values "0,0.05,0.15,0.3,0.5,0.75" \
        --rollout-starts "0,0.25,0.5,0.75" \
        --overwrite
    "${pipeline_checks[@]}" diagnostic "$diagnostic_dir/diagnostics.json" "$label"
}

diagnose_checkpoint() {
    local label="$1"
    local config_path="$2"
    local output_dir="$3"
    local step="$4"
    local split="$5"
    local samples=128
    if [ "$step" -eq 50000 ]; then
        samples=256
    fi
    local diagnostic_dir="$output_dir/diagnostics/checkpoint_$step/$split"
    if "${pipeline_checks[@]}" diagnostic "$diagnostic_dir/diagnostics.json" "$label"; then
        echo "$label checkpoint $step $split diagnostics already complete"
        return
    fi
    local dataset_path="$data_root/heldout"
    if [ "$split" = train ]; then
        dataset_path="$data_root/train"
    fi
    if ! run_logged_with_retries 3 "$label checkpoint $step $split diagnostics" \
        "$output_dir/pipeline.log" run_diagnostic_attempt \
        "$label" "$config_path" "$output_dir/checkpoint_$step" \
        "$diagnostic_dir" "$samples" "$dataset_path"; then
        echo "$label checkpoint $step $split diagnostics failed" >&2
        return 14
    fi
}

for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    config_path="${configs[$index]}"
    output_dir="${run_dirs[$index]}"
    set_stage "$label training"
    run_training "$label" "$config_path" "$output_dir"
    validate_checkpoints "$label" "$output_dir"
    for step in "${checkpoint_steps[@]}"; do
        set_stage "$label checkpoint $step train diagnostics"
        diagnose_checkpoint "$label" "$config_path" "$output_dir" "$step" train
        set_stage "$label checkpoint $step heldout diagnostics"
        diagnose_checkpoint "$label" "$config_path" "$output_dir" "$step" heldout
    done
    for step in "${full_eval_steps[@]}"; do
        samples=500
        if [ "$step" -eq 50000 ]; then
            samples=1000
        fi
        set_stage "$label checkpoint $step evaluation"
        evaluate_checkpoint "$label" "$config_path" "$output_dir" "$step" "$samples"
    done
done

set_stage "summarizing experiment"
"$python_bin" scripts/summarize_phase5_mechanism.py \
    --root "$run_root" \
    --output-dir "$run_root/analysis"
"$python_bin" scripts/render_phase5_mechanism_report.py \
    --summary "$run_root/analysis/mechanism_summary.json" \
    --output "$run_root/analysis/report.html"

"$python_bin" -c '
import json,sys
from datetime import datetime,timezone
json.dump({"status":"complete","completed_at":datetime.now(timezone.utc).isoformat(),"source_commit":sys.argv[2],"runner_commit":sys.argv[3],"analysis":sys.argv[4],"report":sys.argv[5]},open(sys.argv[1],"w"),indent=2)
' "$run_root/pipeline_complete.json" "$experiment_commit" "$runner_commit" "$run_root/analysis/mechanism_summary.json" "$run_root/analysis/report.html"

pipeline_stage="complete"
write_pipeline_status complete
pipeline_completed=1

if command -v notify-send >/dev/null 2>&1; then
    notify-send "DLM-WONN Phase 5" "Mechanism sandbox pipeline completed" || true
fi

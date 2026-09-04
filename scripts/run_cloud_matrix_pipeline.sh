#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
python_bin="${DLM_WONN_PYTHON:-$repo_root/.venv/bin/python}"
run_root="${DLM_WONN_RUN_ROOT:?set DLM_WONN_RUN_ROOT to a new persistent run directory}"
config_dir="src/configs/training_configs"
target_steps=100000
effective_batch=512
eval_batch="${DLM_WONN_EVAL_BATCH_SIZE:-16}"
eval_samples=1000
all_gpus="${DLM_WONN_ALL_GPUS:-0,1,2,3}"
checkpoint_steps=(5000 10000 25000 40000 60000 80000 100000)
model_keys=(E0 W0 W1 W2)
model_slugs=(e0 w0 w1 w2)
model_labels=(
    "E0 ELF-B"
    "W0 WONN-L12/K768/T3"
    "W1 WONN-L6/K768/T3"
    "W2 WONN-L9/K768/T3"
)
model_configs=(
    "$config_dir/train_de-en-ELF-B-E0.yml"
    "$config_dir/train_de-en-WONN-L12K768T3-W0.yml"
    "$config_dir/train_de-en-WONN-L6K768T3-W1.yml"
    "$config_dir/train_de-en-WONN-L9K768T3-W2.yml"
)
model_lrs=(0.002 0.001 0.001 0.001)
model_gpus=(0 1 2 3)
analysis_dir="$run_root/analysis"
status_path="$run_root/pipeline_status.json"
current_stage=preflight
pipeline_done=0

cd "$repo_root"

die() {
    printf '[cloud-matrix] ERROR: %s\n' "$*" >&2
    exit 1
}

[[ "$eval_batch" =~ ^[1-9][0-9]*$ ]] || die "DLM_WONN_EVAL_BATCH_SIZE must be positive"
[ -x "$python_bin" ] || die "missing Python environment: $python_bin"
for config in "${model_configs[@]}"; do
    [ -s "$config" ] || die "missing model config: $config"
done
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
    git status --short >&2
    die "formal cloud runs require a clean checkout"
fi
experiment_commit="$(git rev-parse HEAD)"
commit_short="${experiment_commit:0:7}"
[[ "$run_root" == *"$commit_short"* && "$run_root" == *"$target_steps"* ]] \
    || die "DLM_WONN_RUN_ROOT must contain commit $commit_short and target $target_steps"

visible_gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
[ "$visible_gpu_count" -eq 4 ] \
    || die "formal matrix requires exactly 4 visible GPUs, found $visible_gpu_count"
[ "$all_gpus" = "0,1,2,3" ] || die "DLM_WONN_ALL_GPUS must remain 0,1,2,3"

mkdir -p "$run_root" "$analysis_dir" "$run_root/provenance"
for slug in "${model_slugs[@]}"; do
    mkdir -p "$run_root/$slug"
done
exec 9>"$run_root/pipeline.lock"
flock -n 9 || die "another process holds $run_root/pipeline.lock"

existing_artifact="$(find "$run_root" -maxdepth 2 \( -name 'checkpoint_*' -o -name 'training_complete.json' \) -print -quit)"
if [ -n "$existing_artifact" ] && [ ! -s "$run_root/provenance/source_commit.txt" ]; then
    die "existing training artifacts lack matrix resume provenance"
fi
if [ -s "$run_root/provenance/source_commit.txt" ]; then
    recorded_commit="$(tr -d '\r\n' < "$run_root/provenance/source_commit.txt")"
    [ "$recorded_commit" = "$experiment_commit" ] \
        || die "run root belongs to commit $recorded_commit, not $experiment_commit"
fi
evaluation_contract="samples=$eval_samples;batch=$eval_batch;seed=42"
if [ -s "$run_root/provenance/evaluation_contract.txt" ]; then
    recorded_evaluation="$(tr -d '\r\n' < "$run_root/provenance/evaluation_contract.txt")"
    [ "$recorded_evaluation" = "$evaluation_contract" ] \
        || die "evaluation contract cannot change while resuming"
fi

git status --short --untracked-files=all > "$run_root/provenance/source_worktree_status.txt"
printf '%s\n' "$experiment_commit" > "$run_root/provenance/source_commit.txt"
printf '%s\n' "$evaluation_contract" > "$run_root/provenance/evaluation_contract.txt"
sha256sum "${model_configs[@]}" requirements-lock.txt \
    scripts/analyze_cloud_matrix.py scripts/run_cloud_matrix_pipeline.sh \
    > "$run_root/provenance/runtime_hashes.sha256"
nvidia-smi -q > "$run_root/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$run_root/provenance/nvidia-topology.txt"
uname -a > "$run_root/provenance/uname.txt"
lscpu > "$run_root/provenance/lscpu.txt"
if [ -s /opt/dlm-wonn-environment-manifest.json ]; then
    cp /opt/dlm-wonn-environment-manifest.json "$run_root/provenance/environment_manifest.json"
fi

requested_config="$run_root/provenance/matrix_config.requested.json"
"$python_bin" scripts/analyze_cloud_matrix.py validate-configs \
    --config-dir "$config_dir" --output "$requested_config"
if [ -s "$run_root/provenance/matrix_config.json" ] \
    && ! cmp -s "$run_root/provenance/matrix_config.json" "$requested_config"; then
    die "matrix config contract cannot change while resuming"
fi
mv "$requested_config" "$run_root/provenance/matrix_config.json"

write_status() {
    local value="$1"
    local exit_code="${2:-}"
    local command=(
        "$python_bin" scripts/cloud_pipeline_checks.py status
        "$status_path" "$value" "$experiment_commit" "$experiment_commit"
        "$current_stage" "$$"
    )
    if [ -n "$exit_code" ]; then command+=(--exit-code "$exit_code"); fi
    "${command[@]}"
}

on_exit() {
    local exit_code="$?"
    if [ "$pipeline_done" -ne 1 ]; then write_status failed "$exit_code" || true; fi
}
trap on_exit EXIT
write_status running

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

if [ "${DLM_WONN_PREFLIGHT_ONLY:-0}" = "1" ]; then
    current_stage="preflight validated"
    write_status complete 0
    pipeline_done=1
    printf '[cloud-matrix] preflight validated: 4 models, effective batch %s\n' "$effective_batch"
    exit 0
fi

if [ -s "$run_root/pipeline_complete.json" ]; then
    current_stage="existing completion validation"
    write_status running
    if ! "$python_bin" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); ok=(p.get("status")=="complete" and p.get("experiment_commit")==sys.argv[2] and p.get("terminal_optimizer_step")==100000 and p.get("evaluation_count")==28); raise SystemExit(0 if ok else 1)' \
        "$run_root/pipeline_complete.json" "$experiment_commit" \
        || ! "$python_bin" scripts/analyze_cloud_matrix.py analyze \
        --run-root "$run_root" --output-dir "$analysis_dir" \
        --expected-samples "$eval_samples" >> "$run_root/analysis.log" 2>&1; then
        rm -f "$run_root/pipeline_complete.json"
        die "existing completion marker failed artifact validation and was removed"
    fi
    current_stage=complete
    write_status complete 0
    pipeline_done=1
    printf '[cloud-matrix] existing completed run validated: %s\n' \
        "$run_root/pipeline_complete.json"
    exit 0
fi

pipeline_started_path="$run_root/provenance/pipeline_started_epoch.txt"
if [ ! -s "$pipeline_started_path" ]; then date +%s > "$pipeline_started_path"; fi
pipeline_started_epoch="$(tr -d '\r\n' < "$pipeline_started_path")"
[[ "$pipeline_started_epoch" =~ ^[0-9]+$ ]] || die "invalid pipeline start timestamp"

training_is_complete() {
    local index="$1"
    local run_dir="$run_root/${model_slugs[$index]}"
    "$python_bin" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); ok=(p.get("status")=="complete" and p.get("completed_optimizer_step")==100000 and p.get("effective_batch_size")==512 and p.get("batch_size_per_device")==16 and p.get("grad_accum_steps")==32 and p.get("world_size")==1 and p.get("learning_rate")==float(sys.argv[2])); raise SystemExit(0 if ok else 1)' \
        "$run_dir/training_complete.json" "${model_lrs[$index]}" 2>/dev/null
}

launch_training() {
    local index="$1"
    local key="${model_keys[$index]}"
    local slug="${model_slugs[$index]}"
    local run_dir="$run_root/$slug"
    if training_is_complete "$index"; then
        printf '[cloud-matrix] %s training already complete\n' "$key"
        return
    fi
    (
        export CUDA_VISIBLE_DEVICES="${model_gpus[$index]}"
        export TORCHINDUCTOR_CACHE_DIR="$run_root/cache/torchinductor-$slug"
        mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
        /usr/bin/time -v -o "$run_dir/process_resources_training.txt" \
            "$python_bin" src/train.py \
            --config "${model_configs[$index]}" \
            --config_override "output_dir=$run_dir"
    ) >> "$run_dir/pipeline.log" 2>&1 &
    training_pids[$index]="$!"
}

current_stage="four independent model trainings"
write_status running
training_pids=("" "" "" "")
for index in 0 1 2 3; do launch_training "$index"; done
training_failed=0
for index in 0 1 2 3; do
    if [ -n "${training_pids[$index]}" ]; then
        wait "${training_pids[$index]}" || training_failed=1
    fi
done
[ "$training_failed" -eq 0 ] || die "one or more model trainings failed"
for index in 0 1 2 3; do
    training_is_complete "$index" \
        || die "${model_keys[$index]} lacks a valid 100K completion marker"
done

task_models=()
task_steps=()
for index in 0 1 2 3; do
    for step in "${checkpoint_steps[@]}"; do
        task_models+=("$index")
        task_steps+=("$step")
    done
done

evaluate_one() {
    local model_index="$1"
    local step="$2"
    local gpu="$3"
    local slug="${model_slugs[$model_index]}"
    local label="${model_labels[$model_index]}"
    local run_dir="$run_root/$slug"
    local checkpoint="$run_dir/checkpoint_$step"
    local eval_dir="$run_dir/evaluations/checkpoint_$step"
    local temporary_dir="${eval_dir}.in_progress"
    "$python_bin" scripts/cloud_pipeline_checks.py checkpoint "$checkpoint" \
        || die "missing checkpoint: $checkpoint"
    if "$python_bin" scripts/cloud_pipeline_checks.py evaluation \
        "$eval_dir" "$label" "$step" "$eval_samples"; then
        printf '[cloud-matrix] %s checkpoint %s evaluation already complete\n' "$label" "$step"
        return
    fi
    local timestamp
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    if [ -e "$eval_dir" ]; then mv "$eval_dir" "${eval_dir}.archived_$timestamp"; fi
    if [ -e "$temporary_dir" ]; then mv "$temporary_dir" "${temporary_dir}.archived_$timestamp"; fi
    mkdir -p "$temporary_dir" "$run_root/cache/torchinductor-eval-gpu$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" \
    TORCHINDUCTOR_CACHE_DIR="$run_root/cache/torchinductor-eval-gpu$gpu" \
        "$python_bin" src/eval.py \
        --config "${model_configs[$model_index]}" \
        --config_override "global_batch_size=None" \
        --config_override "batch_size=$eval_batch" \
        --config_override "num_samples=$eval_samples" \
        --config_override "output_dir=$temporary_dir" \
        --checkpoint_path "$checkpoint" --seed 42 \
        >> "$run_dir/evaluation_gpu_${gpu}.log" 2>&1
    "$python_bin" scripts/cloud_pipeline_checks.py finalize-evaluation \
        "$temporary_dir" "$label" "$step" "$eval_samples" \
        || die "incomplete evaluation artifacts for $label checkpoint $step"
    mv "$temporary_dir" "$eval_dir"
}

eval_worker() {
    local gpu="$1"
    local start_index="$2"
    local index
    for ((index=start_index; index<${#task_models[@]}; index+=4)); do
        evaluate_one "${task_models[$index]}" "${task_steps[$index]}" "$gpu"
    done
}

current_stage="four-GPU checkpoint evaluation queue"
write_status running
evaluation_started_epoch="$(date +%s)"
worker_pids=()
for index in 0 1 2 3; do
    eval_worker "$index" "$index" &
    worker_pids+=("$!")
done
evaluation_failed=0
for pid in "${worker_pids[@]}"; do wait "$pid" || evaluation_failed=1; done
[ "$evaluation_failed" -eq 0 ] || die "one or more checkpoint evaluations failed"
evaluation_elapsed_seconds="$(( $(date +%s) - evaluation_started_epoch ))"

current_stage="matrix analysis and quality curves"
write_status running
analysis_started_epoch="$(date +%s)"
"$python_bin" scripts/analyze_cloud_matrix.py analyze \
    --run-root "$run_root" --output-dir "$analysis_dir" \
    --expected-samples "$eval_samples" >> "$run_root/analysis.log" 2>&1
analysis_elapsed_seconds="$(( $(date +%s) - analysis_started_epoch ))"

"$python_bin" - "$run_root/pipeline_complete.json" "$experiment_commit" \
    "$analysis_dir/comparison.json" "$evaluation_elapsed_seconds" \
    "$analysis_elapsed_seconds" "$(( $(date +%s) - pipeline_started_epoch ))" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = {
    "status": "complete",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_commit": sys.argv[2],
    "gpu_layout": "one model per GPU",
    "models": ["E0", "W0", "W1", "W2"],
    "effective_batch_size": 512,
    "terminal_optimizer_step": 100000,
    "checkpoint_steps": [5000, 10000, 25000, 40000, 60000, 80000, 100000],
    "evaluation_count": 28,
    "evaluation_samples_per_checkpoint": 1000,
    "evaluation_elapsed_seconds": int(sys.argv[4]),
    "analysis_elapsed_seconds": int(sys.argv[5]),
    "pipeline_elapsed_seconds": int(sys.argv[6]),
    "analysis": sys.argv[3],
    "quality_chart": str(Path(sys.argv[3]).with_name("quality_curves.svg")),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

current_stage=complete
write_status complete 0
pipeline_done=1
printf '[cloud-matrix] complete: %s\n' "$run_root/pipeline_complete.json"

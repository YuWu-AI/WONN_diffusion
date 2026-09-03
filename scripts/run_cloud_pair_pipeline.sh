#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
python_bin="${DLM_WONN_PYTHON:-$repo_root/.venv/bin/python}"
elf_config="${DLM_WONN_ELF_CONFIG:-src/configs/training_configs/train_de-en-ELF-B-cloud-60k.yml}"
wonn_config="${DLM_WONN_WONN_CONFIG:-src/configs/training_configs/train_de-en-WONN-L12K768T3-cloud-60k.yml}"
run_root="${DLM_WONN_PAIR_RUN_ROOT:?set DLM_WONN_PAIR_RUN_ROOT to a new persistent run directory}"
layout="${DLM_WONN_GPU_LAYOUT:-2+2}"
effective_batch="${DLM_WONN_GLOBAL_BATCH_SIZE:-24}"
eval_batch="${DLM_WONN_EVAL_BATCH_SIZE:-16}"
eval_samples="${DLM_WONN_EVAL_SAMPLES:-500}"
target_steps="${DLM_WONN_TARGET_STEPS:-20000}"
learning_rate="0.0005"
warmup_steps="3000"
elf_gpus="${DLM_WONN_ELF_GPUS:-0,1}"
wonn_gpus="${DLM_WONN_WONN_GPUS:-2,3}"
all_gpus="${DLM_WONN_ALL_GPUS:-0,1,2,3}"
elf_port="${DLM_WONN_ELF_MASTER_PORT:-29501}"
wonn_port="${DLM_WONN_WONN_MASTER_PORT:-29502}"
elf_run="$run_root/elf"
wonn_run="$run_root/wonn"
analysis_dir="$run_root/analysis"
status_path="$run_root/pipeline_status.json"
case "$target_steps" in
    20000)
        elf_steps=(10000 15000 20000)
        wonn_steps=(5000 10000 15000 20000)
        ;;
    *) printf '[cloud-pair] ERROR: DLM_WONN_TARGET_STEPS must be 20000\n' >&2; exit 1 ;;
esac
elf_checkpoint_csv="$(IFS=,; printf '%s' "${elf_steps[*]}")"
wonn_checkpoint_csv="$(IFS=,; printf '%s' "${wonn_steps[*]}")"
ELF_LABEL="Transformer ELF-B"
WONN_LABEL="WONN-L12K768T3"
current_stage="preflight"
pipeline_done=0

cd "$repo_root"

die() {
    printf '[cloud-pair] ERROR: %s\n' "$*" >&2
    exit 1
}

[[ "$effective_batch" =~ ^[1-9][0-9]*$ ]] || die "DLM_WONN_GLOBAL_BATCH_SIZE must be positive"
[ "$effective_batch" -eq 24 ] || die "DLM_WONN_GLOBAL_BATCH_SIZE must remain fixed at 24"
[[ "$eval_batch" =~ ^[1-9][0-9]*$ ]] || die "DLM_WONN_EVAL_BATCH_SIZE must be positive"
[[ "$eval_samples" =~ ^[1-9][0-9]*$ ]] || die "DLM_WONN_EVAL_SAMPLES must be positive"
[ "$eval_samples" -eq 500 ] || die "DLM_WONN_EVAL_SAMPLES must remain fixed at 500"
[[ "$elf_port" =~ ^[1-9][0-9]*$ ]] || die "ELF master port must be positive"
[[ "$wonn_port" =~ ^[1-9][0-9]*$ ]] || die "WONN master port must be positive"
[ "$elf_port" != "$wonn_port" ] || die "ELF and WONN must use different master ports"
[ -x "$python_bin" ] || die "missing Python environment: $python_bin"
[ -s "$elf_config" ] || die "missing ELF config: $elf_config"
[ -s "$wonn_config" ] || die "missing WONN config: $wonn_config"
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
    git status --short >&2
    die "formal cloud runs require a clean checkout"
fi
experiment_commit="$(git rev-parse HEAD)"
commit_short="${experiment_commit:0:7}"
[[ "$run_root" == *"$commit_short"* && "$run_root" == *"$target_steps"* ]] \
    || die "DLM_WONN_PAIR_RUN_ROOT must contain commit $commit_short and target $target_steps"

validate_gpu_list() {
    local value="$1"
    local expected="$2"
    local name="$3"
    local entries=()
    IFS=',' read -r -a entries <<< "$value"
    [ "${#entries[@]}" -eq "$expected" ] \
        || die "$name must contain exactly $expected comma-separated GPU indices"
    local seen="," entry
    for entry in "${entries[@]}"; do
        [[ "$entry" =~ ^[0-9]+$ ]] || die "$name contains a non-numeric GPU index: $entry"
        [ "$entry" -lt "$visible_gpu_count" ] \
            || die "$name contains unavailable GPU index $entry (visible count: $visible_gpu_count)"
        [[ "$seen" != *",$entry,"* ]] || die "$name contains duplicate GPU index $entry"
        seen+="$entry,"
    done
}

visible_gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
[ "$visible_gpu_count" -eq 4 ] \
    || die "formal four-GPU gate expected 4 visible GPUs, found $visible_gpu_count"
validate_gpu_list "$all_gpus" 4 DLM_WONN_ALL_GPUS
[ "$all_gpus" = "0,1,2,3" ] || die "DLM_WONN_ALL_GPUS must remain fixed at 0,1,2,3"

case "$layout" in
    2+2)
        world_size=2
        validate_gpu_list "$elf_gpus" 2 DLM_WONN_ELF_GPUS
        validate_gpu_list "$wonn_gpus" 2 DLM_WONN_WONN_GPUS
        [ "$elf_gpus" = "0,1" ] || die "DLM_WONN_ELF_GPUS must remain fixed at 0,1"
        [ "$wonn_gpus" = "2,3" ] || die "DLM_WONN_WONN_GPUS must remain fixed at 2,3"
        elf_gpu_array=()
        wonn_gpu_array=()
        IFS=',' read -r -a elf_gpu_array <<< "$elf_gpus"
        IFS=',' read -r -a wonn_gpu_array <<< "$wonn_gpus"
        for gpu in "${elf_gpu_array[@]}"; do
            [[ ",${wonn_gpus}," != *",$gpu,"* ]] \
                || die "2+2 GPU groups overlap at GPU $gpu"
        done
        for gpu in "${elf_gpu_array[@]}" "${wonn_gpu_array[@]}"; do
            [[ ",${all_gpus}," == *",$gpu,"* ]] \
                || die "training GPU $gpu is absent from DLM_WONN_ALL_GPUS"
        done
        all_gpu_array=()
        IFS=',' read -r -a all_gpu_array <<< "$all_gpus"
        for gpu in "${all_gpu_array[@]}"; do
            [[ ",${elf_gpus},${wonn_gpus}," == *",$gpu,"* ]] \
                || die "evaluation GPU $gpu is absent from the 2+2 training groups"
        done
        ;;
    *) die "DLM_WONN_GPU_LAYOUT must remain fixed at 2+2" ;;
esac

mkdir -p "$run_root" "$elf_run" "$wonn_run" "$analysis_dir" "$run_root/provenance"
exec 9>"$run_root/pipeline.lock"
flock -n 9 || die "another process holds $run_root/pipeline.lock"
exec 7>"$elf_run/training.lock"
flock -n 7 || die "another process holds $elf_run/training.lock"
exec 8>"$wonn_run/training.lock"
flock -n 8 || die "another process holds $wonn_run/training.lock"

existing_training_artifact="$(
    find "$elf_run" "$wonn_run" -maxdepth 1 \
        \( -name 'checkpoint_*' -o -name 'training_complete.json' \) \
        -print -quit
)"
if [ -n "$existing_training_artifact" ] \
    && { [ ! -s "$run_root/provenance/source_commit.txt" ] \
        || [ ! -s "$run_root/provenance/pair_config.json" ]; }; then
    die "existing training artifacts lack the cloud-pair resume provenance contract"
fi
if [ -s "$run_root/provenance/source_commit.txt" ]; then
    recorded_commit="$(tr -d '\r\n' < "$run_root/provenance/source_commit.txt")"
    [ "$recorded_commit" = "$experiment_commit" ] \
        || die "run root belongs to commit $recorded_commit, not $experiment_commit"
fi
assignment="layout=$layout;elf=$elf_gpus;wonn=$wonn_gpus;all=$all_gpus"
if [ -s "$run_root/provenance/gpu_assignment.txt" ]; then
    recorded_assignment="$(tr -d '\r\n' < "$run_root/provenance/gpu_assignment.txt")"
    [ "$recorded_assignment" = "$assignment" ] \
        || die "GPU layout cannot change while resuming this run"
fi
evaluation_contract="samples=$eval_samples;batch=$eval_batch;seed=42"
if [ -s "$run_root/provenance/evaluation_contract.txt" ]; then
    recorded_evaluation_contract="$(tr -d '\r\n' < "$run_root/provenance/evaluation_contract.txt")"
    [ "$recorded_evaluation_contract" = "$evaluation_contract" ] \
        || die "evaluation contract cannot change while resuming this run"
fi
training_contract="target=$target_steps;elf_checkpoints=$elf_checkpoint_csv;wonn_checkpoints=$wonn_checkpoint_csv;batch=$effective_batch;lr=$learning_rate;warmup=$warmup_steps;layout=$layout"
if [ -s "$run_root/provenance/training_contract.txt" ]; then
    recorded_training_contract="$(tr -d '\r\n' < "$run_root/provenance/training_contract.txt")"
    [ "$recorded_training_contract" = "$training_contract" ] \
        || die "training target/checkpoints/batch/LR/warmup/layout cannot change while resuming"
fi
git status --short --untracked-files=all > "$run_root/provenance/source_worktree_status.txt"
printf '%s\n' "$experiment_commit" > "$run_root/provenance/source_commit.txt"
printf '%s\n' "$assignment" > "$run_root/provenance/gpu_assignment.txt"
printf '%s\n' "$evaluation_contract" > "$run_root/provenance/evaluation_contract.txt"
printf '%s\n' "$training_contract" > "$run_root/provenance/training_contract.txt"
sha256sum "$elf_config" "$wonn_config" requirements-lock.txt \
    scripts/analyze_cloud_pair.py \
    scripts/run_cloud_pair_pipeline.sh \
    > "$run_root/provenance/runtime_hashes.sha256"
nvidia-smi -q > "$run_root/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$run_root/provenance/nvidia-topology.txt"
uname -a > "$run_root/provenance/uname.txt"
lscpu > "$run_root/provenance/lscpu.txt"
if [ -s /opt/dlm-wonn-environment-manifest.json ]; then
    cp /opt/dlm-wonn-environment-manifest.json "$run_root/provenance/environment_manifest.json"
fi

requested_pair_config="$run_root/provenance/pair_config.requested.json"
"$python_bin" scripts/analyze_cloud_pair.py validate-configs \
    --elf-config "$elf_config" \
    --wonn-config "$wonn_config" \
    --effective-batch "$effective_batch" \
    --world-size "$world_size" \
    --target-steps "$target_steps" \
    --output "$requested_pair_config"
if [ -s "$run_root/provenance/pair_config.json" ] \
    && ! cmp -s "$run_root/provenance/pair_config.json" "$requested_pair_config"; then
    die "batch/LR/world-size contract cannot change while resuming this run"
fi
mv "$requested_pair_config" "$run_root/provenance/pair_config.json"

write_status() {
    local value="$1"
    local exit_code="${2:-}"
    local command=(
        "$python_bin" scripts/cloud_pipeline_checks.py status
        "$status_path" "$value" "$experiment_commit" "$experiment_commit"
        "$current_stage" "$$"
    )
    if [ -n "$exit_code" ]; then
        command+=(--exit-code "$exit_code")
    fi
    "${command[@]}"
}

on_exit() {
    local exit_code="$?"
    if [ "$pipeline_done" -ne 1 ]; then
        write_status failed "$exit_code" || true
    fi
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
    write_status running
    pipeline_done=1
    printf '[cloud-pair] preflight validated: layout=%s world=%s effective_batch=%s\n' \
        "$layout" "$world_size" "$effective_batch"
    exit 0
fi

pipeline_started_path="$run_root/provenance/pipeline_started_epoch.txt"
if [ ! -s "$pipeline_started_path" ]; then
    date +%s > "$pipeline_started_path"
fi
pipeline_started_epoch="$(tr -d '\r\n' < "$pipeline_started_path")"
[[ "$pipeline_started_epoch" =~ ^[0-9]+$ ]] || die "invalid pipeline start timestamp"

training_is_complete() {
    local run_dir="$1"
    "$python_bin" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); ok=(p.get("status")=="complete" and p.get("completed_optimizer_step")==int(sys.argv[2]) and p.get("effective_batch_size")==int(sys.argv[3]) and p.get("learning_rate")==float(sys.argv[4]) and p.get("world_size")==2); raise SystemExit(0 if ok else 1)' \
        "$run_dir/training_complete.json" "$target_steps" "$effective_batch" "$learning_rate" 2>/dev/null
}

TRAIN_PID=""
launch_training() {
    local label="$1"
    local config="$2"
    local run_dir="$3"
    local gpu_list="$4"
    local master_port="$5"
    local save_steps="$6"
    if training_is_complete "$run_dir"; then
        printf '[cloud-pair] %s training already complete\n' "$label"
        TRAIN_PID=""
        return
    fi
    current_stage="$label training to $target_steps"
    write_status running
    (
        export CUDA_VISIBLE_DEVICES="$gpu_list"
        export TORCHINDUCTOR_CACHE_DIR="$run_root/cache/torchinductor-$label"
        mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
        /usr/bin/time -v -o "$run_dir/process_resources_training.txt" \
            "$python_bin" -m torch.distributed.run \
            --nproc_per_node="$world_size" \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=127.0.0.1 \
            --master_port="$master_port" \
            src/train.py \
            --config "$config" \
            --config_override "global_batch_size=$effective_batch" \
            --config_override "lr=$learning_rate" \
            --config_override "warmup_steps=$warmup_steps" \
            --config_override "max_optimizer_steps=$target_steps" \
            --config_override "save_optimizer_steps=$save_steps" \
            --config_override "num_samples=$eval_samples" \
            --config_override "output_dir=$run_dir"
    ) >> "$run_dir/pipeline.log" 2>&1 &
    TRAIN_PID="$!"
}

wait_for_training() {
    local elf_status=0
    local wonn_status=0
    launch_training ELF "$elf_config" "$elf_run" "$elf_gpus" "$elf_port" "$elf_checkpoint_csv"
    local elf_pid="$TRAIN_PID"
    launch_training WONN "$wonn_config" "$wonn_run" "$wonn_gpus" "$wonn_port" "$wonn_checkpoint_csv"
    local wonn_pid="$TRAIN_PID"
    if [ -n "$elf_pid" ]; then wait "$elf_pid" || elf_status="$?"; fi
    if [ -n "$wonn_pid" ]; then wait "$wonn_pid" || wonn_status="$?"; fi
    if [ "$elf_status" -ne 0 ] || [ "$wonn_status" -ne 0 ]; then
        die "training failed: ELF exit=$elf_status, WONN exit=$wonn_status"
    fi
    training_is_complete "$elf_run" || die "ELF lacks a valid $target_steps completion marker"
    training_is_complete "$wonn_run" || die "WONN lacks a valid $target_steps completion marker"
}

wait_for_training

task_models=()
task_steps=()
for step in "${elf_steps[@]}"; do
    task_models+=(elf)
    task_steps+=("$step")
done
for step in "${wonn_steps[@]}"; do
    task_models+=(wonn)
    task_steps+=("$step")
done

evaluate_one() {
    local model="$1"
    local step="$2"
    local gpu="$3"
    local label config run_dir
    if [ "$model" = "elf" ]; then
        label="$ELF_LABEL"
        config="$elf_config"
        run_dir="$elf_run"
    else
        label="$WONN_LABEL"
        config="$wonn_config"
        run_dir="$wonn_run"
    fi
    local checkpoint="$run_dir/checkpoint_$step"
    local eval_dir="$run_dir/evaluations/checkpoint_$step"
    local temporary_dir="${eval_dir}.in_progress"
    "$python_bin" scripts/cloud_pipeline_checks.py checkpoint "$checkpoint" \
        || die "missing checkpoint: $checkpoint"
    if "$python_bin" scripts/cloud_pipeline_checks.py evaluation \
        "$eval_dir" "$label" "$step" "$eval_samples"; then
        printf '[cloud-pair] %s checkpoint %s evaluation already complete\n' "$label" "$step"
        return
    fi
    local timestamp
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    if [ -e "$eval_dir" ]; then mv "$eval_dir" "${eval_dir}.archived_$timestamp"; fi
    if [ -e "$temporary_dir" ]; then
        mv "$temporary_dir" "${temporary_dir}.archived_$timestamp"
    fi
    mkdir -p "$temporary_dir" "$run_root/cache/torchinductor-eval-gpu$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" \
    TORCHINDUCTOR_CACHE_DIR="$run_root/cache/torchinductor-eval-gpu$gpu" \
        "$python_bin" src/eval.py \
        --config "$config" \
        --config_override "global_batch_size=None" \
        --config_override "batch_size=$eval_batch" \
        --config_override "num_samples=$eval_samples" \
        --config_override "output_dir=$temporary_dir" \
        --checkpoint_path "$checkpoint" \
        --seed 42 >> "$run_dir/evaluation_gpu_${gpu}.log" 2>&1
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
eval_gpu_array=()
IFS=',' read -r -a eval_gpu_array <<< "$all_gpus"
worker_pids=()
for index in 0 1 2 3; do
    eval_worker "${eval_gpu_array[$index]}" "$index" &
    worker_pids+=("$!")
done
evaluation_status=0
for worker_pid in "${worker_pids[@]}"; do
    wait "$worker_pid" || evaluation_status=1
done
[ "$evaluation_status" -eq 0 ] || die "one or more checkpoint evaluation workers failed"
evaluation_elapsed_seconds="$(( $(date +%s) - evaluation_started_epoch ))"

current_stage="paired analysis and quality curves"
write_status running
analysis_started_epoch="$(date +%s)"
"$python_bin" scripts/analyze_cloud_pair.py analyze \
    --elf-run-dir "$elf_run" \
    --wonn-run-dir "$wonn_run" \
    --output-dir "$analysis_dir" \
    --expected-samples "$eval_samples" \
    >> "$run_root/analysis.log" 2>&1
analysis_elapsed_seconds="$(( $(date +%s) - analysis_started_epoch ))"

"$python_bin" - "$run_root/pipeline_complete.json" "$experiment_commit" \
    "$layout" "$effective_batch" "$analysis_dir/comparison.json" \
    "$target_steps" "${#elf_steps[@]}" "${#wonn_steps[@]}" "${#task_models[@]}" \
    "$elf_checkpoint_csv" "$wonn_checkpoint_csv" "$learning_rate" "$warmup_steps" \
    "$evaluation_elapsed_seconds" "$analysis_elapsed_seconds" \
    "$(( $(date +%s) - pipeline_started_epoch ))" <<'PY'
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
    "gpu_layout": sys.argv[3],
    "effective_batch_size": int(sys.argv[4]),
    "terminal_optimizer_step": int(sys.argv[6]),
    "checkpoint_counts": {"elf": int(sys.argv[7]), "wonn": int(sys.argv[8])},
    "evaluation_count": int(sys.argv[9]),
    "checkpoint_steps": {
        "elf": [int(value) for value in sys.argv[10].split(",")],
        "wonn": [int(value) for value in sys.argv[11].split(",")],
    },
    "learning_rate": float(sys.argv[12]),
    "warmup_steps": int(sys.argv[13]),
    "evaluation_elapsed_seconds": int(sys.argv[14]),
    "analysis_elapsed_seconds": int(sys.argv[15]),
    "pipeline_elapsed_seconds": int(sys.argv[16]),
    "analysis": sys.argv[5],
    "quality_chart": str(Path(sys.argv[5]).with_name("quality_curves.svg")),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

current_stage="complete"
write_status complete 0
pipeline_done=1
printf '[cloud-pair] complete: %s\n' "$run_root/pipeline_complete.json"

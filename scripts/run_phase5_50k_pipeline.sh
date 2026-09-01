#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
common_git_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir)"
main_checkout="$(dirname "$common_git_dir")"
python_bin="${DLM_WONN_PYTHON:-$main_checkout/.venv/bin/python}"
run_root="outputs/phase5/wonn_runs/l6t3_seed42_b12/formal_0_50k"
elf_baseline="${DLM_WONN_ELF_BASELINE:-outputs/phase5/elf_b_seed42_b12_0_90k}"
wonn_config="src/configs/training_configs/train_de-en-WONN-L6T3-phase5-50k.yml"
wonn_dir="$run_root/wonn_l6t3_seed42_b12"
evaluation_steps=(10000 20000 30000 40000 50000)

cd "$repo_root"

if [ "$(git branch --show-current)" != "phase5-wmt14" ]; then
    echo "refusing to run outside the phase5-wmt14 branch" >&2
    exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
    echo "refusing to start a formal run from a dirty worktree" >&2
    git status --short >&2
    exit 3
fi
if [ ! -x "$python_bin" ]; then
    echo "missing Python environment: $python_bin" >&2
    exit 4
fi
if [ ! -f "$wonn_config" ]; then
    echo "missing formal 50K training configuration" >&2
    exit 5
fi
for required in \
    "$elf_baseline/provenance/experiment_manifest_00000_50000.json" \
    "$elf_baseline/provenance/training_complete_50000.json" \
    "$elf_baseline/provenance/config_00000_50000.yml" \
    "$elf_baseline/provenance/source_commit_00000_50000.txt" \
    "$elf_baseline/training/train_metrics_00000_50000.jsonl" \
    "$elf_baseline/checkpoints/checkpoint_50000" \
    "$elf_baseline/evaluations/checkpoint_50000/evaluation_complete.json"; do
    if [ ! -s "$required" ]; then
        echo "missing consolidated ELF baseline artifact: $required" >&2
        exit 14
    fi
done

mkdir -p "$run_root"
exec 9>"$run_root/pipeline.lock"
if ! flock -n 9; then
    echo "another Phase 5 50K pipeline already holds $run_root/pipeline.lock" >&2
    exit 6
fi

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

python_version="$($python_bin -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python_include="${DLM_WONN_PYTHON_INCLUDE:-$($python_bin -c 'import sysconfig; print(sysconfig.get_path("include"))')}"
if [ ! -f "$python_include/Python.h" ]; then
    user_home="$(getent passwd "$(id -u)" | cut -d: -f6)"
    python_header=""
    if [ -d "$user_home/miniconda3/envs" ]; then
        python_header="$(find "$user_home/miniconda3/envs" \
            -path "*/include/python$python_version/Python.h" -print -quit)"
    fi
    if [ -n "$python_header" ]; then
        python_include="$(dirname "$python_header")"
    fi
fi
if [ ! -f "$python_include/Python.h" ]; then
    echo "missing Python.h; set DLM_WONN_PYTHON_INCLUDE to a Python $python_version include directory" >&2
    exit 13
fi
export CPATH="$python_include${CPATH:+:$CPATH}"

orchestrator_commit="$(git rev-parse HEAD)"
source_commit="$orchestrator_commit"
started_at="$(date --iso-8601=seconds)"
if [ ! -e "$run_root/pipeline_started.json" ]; then
    printf '{\n  "status": "running",\n  "started_at": "%s",\n  "source_commit": "%s"\n}\n' \
        "$started_at" "$source_commit" > "$run_root/pipeline_started.json"
else
    recorded_commit="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$run_root/pipeline_started.json")"
    if [ "$recorded_commit" != "$orchestrator_commit" ]; then
        if ! git diff --quiet "$recorded_commit" "$orchestrator_commit" -- src; then
            echo "refusing to resume after model or training code changed since $recorded_commit" >&2
            exit 7
        fi
        source_commit="$recorded_commit"
        printf '{\n  "status": "resuming_to_50000",\n  "resumed_at": "%s",\n  "training_source_commit": "%s",\n  "orchestrator_commit": "%s",\n  "intermediate_gates": "advisory"\n}\n' \
            "$started_at" "$source_commit" "$orchestrator_commit" \
            > "$run_root/pipeline_resumed_to_50000.json"
    fi
fi

manifest_path="$run_root/experiment_manifest.json"
if [ ! -s "$manifest_path" ]; then
    "$python_bin" scripts/capture_phase5_manifest.py \
        --repo-root "$repo_root" \
        --output "$manifest_path" \
        --config "$wonn_config"
fi

run_training() {
    local label="$1"
    local config_path="$2"
    local output_dir="$3"
    local target_step="$4"
    local log_path="$output_dir/pipeline.log"
    local completion="$output_dir/training_complete.json"
    local resource_path="$output_dir/process_resources_to_${target_step}.txt"
    mkdir -p "$output_dir"
    if [ -s "$completion" ]; then
        completed_step="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_optimizer_step"))' "$completion")"
        if [ "$completed_step" -ge "$target_step" ]; then
            echo "$label training already reached step $completed_step; skipping target $target_step"
            return
        fi
    fi
    printf '%s\n' "$source_commit" > "$output_dir/source_commit"
    printf '%s\n' "$config_path" > "$output_dir/source_config"
    sha256sum "$config_path" > "$output_dir/source_config.sha256"
    command=("$python_bin" src/train.py --config "$config_path")
    if [ "$target_step" -eq 10000 ]; then
        command+=(
            --config_override "stop_optimizer_steps=10000"
            --config_override "save_optimizer_steps=5000,10000"
        )
    elif [ "$target_step" -eq 20000 ]; then
        command+=(
            --config_override "stop_optimizer_steps=20000"
            --config_override "save_optimizer_steps=5000,10000,15000,20000"
        )
    fi
    echo "starting or resuming $label training to optimizer step $target_step"
    /usr/bin/time -v -o "$resource_path" \
        "${command[@]}" >> "$log_path" 2>&1
    if [ ! -s "$completion" ]; then
        echo "$label exited without training_complete.json" >&2
        exit 8
    fi
    completed_step="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_optimizer_step"))' "$completion")"
    if [ "$completed_step" -ne "$target_step" ]; then
        echo "$label completed step $completed_step, expected $target_step" >&2
        exit 9
    fi
}

evaluate_checkpoint() {
    local label="$1"
    local config_path="$2"
    local output_dir="$3"
    local step="$4"
    local expected_samples="$5"
    local checkpoint="$output_dir/checkpoint_$step"
    local eval_dir="$output_dir/evaluations/checkpoint_$step"
    local temporary_dir="${eval_dir}.in_progress"
    local log_path="$output_dir/pipeline.log"

    if [ ! -s "$checkpoint" ]; then
        echo "missing $label checkpoint: $checkpoint" >&2
        exit 10
    fi
    if [ -s "$eval_dir/evaluation_complete.json" ]; then
        echo "$label checkpoint $step evaluation already complete; skipping"
        return
    fi
    if [ -e "$eval_dir" ]; then
        mv "$eval_dir" "${eval_dir}.incomplete.$(date +%Y%m%dT%H%M%S)"
    fi
    if [ -e "$temporary_dir" ]; then
        mv "$temporary_dir" "${temporary_dir}.$(date +%Y%m%dT%H%M%S)"
    fi
    mkdir -p "$temporary_dir"
    echo "evaluating $label checkpoint $step on $expected_samples validation examples"
    "$python_bin" src/eval.py \
        --config "$config_path" \
        --config_override "output_dir=$temporary_dir" \
        --config_override "num_samples=$expected_samples" \
        --config_override "batch_size=16" \
        --checkpoint_path "$checkpoint" \
        --seed 42 >> "$log_path" 2>&1

    generated_path="$(find "$temporary_dir" -type f -name "all_generated_*_${step}.jsonl" -print -quit)"
    metrics_path="$(find "$temporary_dir" -type f -name "metrics.jsonl" -print -quit)"
    if [ -z "$generated_path" ] || [ -z "$metrics_path" ]; then
        echo "$label checkpoint $step evaluation artifacts are incomplete" >&2
        exit 11
    fi
    generated_count="$(wc -l < "$generated_path")"
    if [ "$generated_count" -ne "$expected_samples" ]; then
        echo "$generated_path has $generated_count rows, expected $expected_samples" >&2
        exit 12
    fi
    printf '{\n  "status": "complete",\n  "model": "%s",\n  "step": %s,\n  "num_samples": %s\n}\n' \
        "$label" "$step" "$expected_samples" > "$temporary_dir/evaluation_complete.json"
    mv "$temporary_dir" "$eval_dir"
}

run_conditioning_diagnostic() {
    local step="$1"
    local output_dir="$wonn_dir/diagnostics/checkpoint_$step"
    local report="$output_dir/diagnostics.json"
    local log_path="$wonn_dir/pipeline.log"
    if [ -s "$report" ]; then
        diagnostic_status="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$report")"
        if [ "$diagnostic_status" = "complete" ]; then
            echo "WONN-L6T3 checkpoint $step conditioning diagnostic already complete; skipping"
            return
        fi
    fi
    echo "running WONN-L6T3 checkpoint $step source-conditioning diagnostic"
    "$python_bin" scripts/diagnose_phase5_checkpoint.py \
        --root "$run_root" \
        --output-dir "$output_dir" \
        --num-samples 256 \
        --batch-size 16 \
        --seed 42 \
        --model-spec "WONN-L6T3|$wonn_config|wonn_l6t3_seed42_b12/checkpoint_$step" \
        --t-values 0.5 \
        --rollout-starts 0.0 \
        --overwrite >> "$log_path" 2>&1
    if [ ! -s "$report" ]; then
        echo "WONN-L6T3 checkpoint $step diagnostic did not produce $report" >&2
        exit 15
    fi
}

run_gate() {
    local stage="$1"
    local step="$2"
    local gate_dir="$run_root/analysis/$stage"
    local gate_report="$gate_dir/comparison.json"
    local diagnostics="$wonn_dir/diagnostics/checkpoint_$step/diagnostics.json"
    local gate_passed="false"
    if [ -s "$gate_report" ]; then
        gate_passed="$($python_bin -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["passed"]).lower())' "$gate_report")"
    fi
    if [ "$gate_passed" != "true" ]; then
        "$python_bin" scripts/evaluate_phase5_wonn_gate.py \
            --root "$run_root" \
            --output-dir "$gate_dir" \
            --stage "$stage" \
            --diagnostics "$diagnostics" \
            --verify-checkpoints
        gate_passed="$($python_bin -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["passed"]).lower())' "$gate_report")"
    else
        echo "reusing passed $stage report at $gate_report"
    fi
    if [ "$gate_passed" != "true" ]; then
        observed_at="$(date --iso-8601=seconds)"
        printf '{\n  "status": "observed_failed_continuing",\n  "observed_at": "%s",\n  "source_commit": "%s",\n  "gate_report": "%s"\n}\n' \
            "$observed_at" "$source_commit" "$gate_report" \
            > "$run_root/pipeline_${stage}_observed.json"
        echo "$stage did not pass; recording the observation and continuing to 50K" >&2
        return
    fi
    printf '{\n  "status": "passed",\n  "source_commit": "%s",\n  "gate_report": "%s"\n}\n' \
        "$source_commit" "$gate_report" > "$run_root/pipeline_${stage}_passed.json"
}

pilot_report="$run_root/analysis/pilot10k/comparison.json"
pilot_passed="false"
if [ -s "$pilot_report" ]; then
    pilot_passed="$($python_bin -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["passed"]).lower())' "$pilot_report")"
fi
if [ "$pilot_passed" != "true" ]; then
    run_training "WONN-L6T3" "$wonn_config" "$wonn_dir" 10000
    for step in 5000 10000; do
        evaluate_checkpoint "WONN-L6T3" "$wonn_config" "$wonn_dir" "$step" 1000
    done
    run_conditioning_diagnostic 10000
fi
run_gate "pilot10k" 10000

gate_report="$run_root/analysis/gate20k/comparison.json"
gate_passed="false"
if [ -s "$gate_report" ]; then
    gate_passed="$($python_bin -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["gate"]["passed"]).lower())' "$gate_report")"
fi
if [ "$gate_passed" != "true" ]; then
    run_training "WONN-L6T3" "$wonn_config" "$wonn_dir" 20000

    for step in 10000 20000; do
        evaluate_checkpoint "WONN-L6T3" "$wonn_config" "$wonn_dir" "$step" 1000
    done
    run_conditioning_diagnostic 20000
fi
run_gate "gate20k" 20000

run_training "WONN-L6T3" "$wonn_config" "$wonn_dir" 50000

for step in "${evaluation_steps[@]}"; do
    samples=1000
    if [ "$step" -eq 50000 ]; then
        samples=3000
    fi
    evaluate_checkpoint "WONN-L6T3" "$wonn_config" "$wonn_dir" "$step" "$samples"
done

final_dir="$run_root/analysis/final50k"
"$python_bin" scripts/analyze_phase5_50k.py \
    --root "$run_root" \
    --elf-run-dir "$elf_baseline" \
    --output-dir "$final_dir" \
    --stage final50k \
    --verify-checkpoints \
    --bootstrap-resamples 1000

completed_at="$(date --iso-8601=seconds)"
printf '{\n  "status": "complete",\n  "completed_at": "%s",\n  "source_commit": "%s",\n  "orchestrator_commit": "%s",\n  "analysis": "%s"\n}\n' \
    "$completed_at" "$source_commit" "$orchestrator_commit" "$final_dir/comparison.json" \
    > "$run_root/pipeline_complete.json"

if command -v notify-send >/dev/null 2>&1; then
    notify-send "DLM-WONN Phase 5" "WMT14 WONN 50K vs consolidated ELF completed"
fi

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
common_git_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir)"
main_checkout="$(dirname "$common_git_dir")"
python_bin="${DLM_WONN_PYTHON:-$main_checkout/.venv/bin/python}"
run_root="outputs/phase5/mechanism50k"
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

source_commit="$(git rev-parse HEAD)"
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
manifest_commit="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$manifest_path")"
if [ "$manifest_commit" != "$source_commit" ]; then
    echo "experiment manifest uses commit $manifest_commit, current commit is $source_commit" >&2
    exit 8
fi

if [ ! -s "$run_root/pipeline_started.json" ]; then
    "$python_bin" -c '
import json,sys
from datetime import datetime,timezone
json.dump({"status":"running","started_at":datetime.now(timezone.utc).isoformat(),"source_commit":sys.argv[2]},open(sys.argv[1],"w"),indent=2)
' "$run_root/pipeline_started.json" "$source_commit"
else
    recorded_commit="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$run_root/pipeline_started.json")"
    if [ "$recorded_commit" != "$source_commit" ]; then
        echo "existing run uses commit $recorded_commit, current commit is $source_commit" >&2
        exit 9
    fi
fi

run_training() {
    local label="$1"
    local config_path="$2"
    local output_dir="$3"
    local completion="$output_dir/training_complete.json"
    mkdir -p "$output_dir"
    if [ -s "$completion" ]; then
        local completed_step
        completed_step="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_optimizer_step",-1))' "$completion")"
        if [ "$completed_step" -eq 50000 ]; then
            echo "$label training already complete"
            return
        fi
    fi
    printf '%s\n' "$source_commit" > "$output_dir/source_commit"
    printf '%s\n' "$config_path" > "$output_dir/source_config"
    sha256sum "$config_path" > "$output_dir/source_config.sha256"
    echo "training $label to 50k"
    /usr/bin/time -v -o "$output_dir/process_resources.txt" \
        "$python_bin" src/train.py --config "$config_path" \
        >> "$output_dir/pipeline.log" 2>&1
    if [ ! -s "$completion" ]; then
        echo "$label did not produce training_complete.json" >&2
        exit 10
    fi
    completed_step="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_optimizer_step",-1))' "$completion")"
    if [ "$completed_step" -ne 50000 ]; then
        echo "$label stopped at optimizer step $completed_step instead of 50000" >&2
        exit 11
    fi
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
    if [ -s "$eval_dir/evaluation_complete.json" ]; then
        echo "$label checkpoint $step full evaluation already complete"
        return
    fi
    if [ ! -d "$checkpoint" ]; then
        echo "missing checkpoint: $checkpoint" >&2
        exit 12
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
    "$python_bin" src/eval.py \
        --config "$config_path" \
        --config_override "output_dir=$temporary" \
        --config_override "num_samples=$samples" \
        --config_override "batch_size=16" \
        --checkpoint_path "$checkpoint" \
        --seed 42 >> "$output_dir/pipeline.log" 2>&1
    local generated
    generated="$(find "$temporary" -type f -name "all_generated_*_${step}.jsonl" -print -quit)"
    if [ -z "$generated" ] || [ "$(wc -l < "$generated")" -ne "$samples" ]; then
        echo "$label checkpoint $step evaluation is incomplete" >&2
        exit 13
    fi
    "$python_bin" -c '
import json,sys
json.dump({"status":"complete","model":sys.argv[2],"step":int(sys.argv[3]),"num_samples":int(sys.argv[4])},open(sys.argv[1],"w"),indent=2)
' "$temporary/evaluation_complete.json" "$label" "$step" "$samples"
    mv "$temporary" "$eval_dir"
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
    if [ -s "$diagnostic_dir/diagnostics.json" ]; then
        local status
        status="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$diagnostic_dir/diagnostics.json")"
        if [ "$status" = complete ]; then
            echo "$label checkpoint $step $split diagnostics already complete"
            return
        fi
    fi
    local dataset_path="$data_root/heldout"
    if [ "$split" = train ]; then
        dataset_path="$data_root/train"
    fi
    "$python_bin" scripts/diagnose_phase5_checkpoint.py \
        --root . \
        --output-dir "$diagnostic_dir" \
        --num-samples "$samples" \
        --batch-size 16 \
        --seed 42 \
        --model-spec "$label|$config_path|$output_dir/checkpoint_$step" \
        --dataset-path "$dataset_path" \
        --dataset-revision "aee21115965ee2b9d3d6d02ec2a72f4f998476d9" \
        --t-values "0,0.05,0.15,0.3,0.5,0.75" \
        --rollout-starts "0,0.25,0.5,0.75" \
        --overwrite \
        >> "$output_dir/pipeline.log" 2>&1
}

for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    config_path="${configs[$index]}"
    output_dir="${run_dirs[$index]}"
    run_training "$label" "$config_path" "$output_dir"
    for step in "${checkpoint_steps[@]}"; do
        diagnose_checkpoint "$label" "$config_path" "$output_dir" "$step" train
        diagnose_checkpoint "$label" "$config_path" "$output_dir" "$step" heldout
    done
    for step in "${full_eval_steps[@]}"; do
        samples=500
        if [ "$step" -eq 50000 ]; then
            samples=1000
        fi
        evaluate_checkpoint "$label" "$config_path" "$output_dir" "$step" "$samples"
    done
done

"$python_bin" scripts/summarize_phase5_mechanism.py \
    --root "$run_root" \
    --output-dir "$run_root/analysis"
"$python_bin" scripts/render_phase5_mechanism_report.py \
    --summary "$run_root/analysis/mechanism_summary.json" \
    --output "$run_root/analysis/report.html"

"$python_bin" -c '
import json,sys
from datetime import datetime,timezone
json.dump({"status":"complete","completed_at":datetime.now(timezone.utc).isoformat(),"source_commit":sys.argv[2],"analysis":sys.argv[3],"report":sys.argv[4]},open(sys.argv[1],"w"),indent=2)
' "$run_root/pipeline_complete.json" "$source_commit" "$run_root/analysis/mechanism_summary.json" "$run_root/analysis/report.html"

if command -v notify-send >/dev/null 2>&1; then
    notify-send "DLM-WONN Phase 5" "Mechanism sandbox pipeline completed"
fi

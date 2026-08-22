#!/usr/bin/env python
"""Validate and compare the gated WMT14 ELF-B/WONN-L6T3 runs."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import sacrebleu
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_phase5_run import _audit_checkpoint


MODEL_SPECS = {
    "ELF-B": {
        "directory": "elf_b_seed42_b12",
        "config_model": "ELF-B",
        "wonn_inner_steps": None,
    },
    "WONN-L6T3": {
        "directory": "wonn_l6t3_seed42_b12",
        "config_model": "ELF-WONN-B",
        "wonn_inner_steps": 3,
    },
}
STAGES = {
    "gate20k": {
        "target_step": 20000,
        "checkpoint_steps": (5000, 10000, 15000, 20000),
        "evaluation_steps": (10000, 20000),
        "expected_samples": {10000: 1000, 20000: 1000},
    },
    "final50k": {
        "target_step": 50000,
        "checkpoint_steps": tuple(range(5000, 50001, 5000)),
        "evaluation_steps": (10000, 20000, 30000, 40000, 50000),
        "expected_samples": {
            10000: 1000, 20000: 1000, 30000: 1000,
            40000: 1000, 50000: 3000,
        },
    },
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _metric_bundle(hypotheses: list, references: list) -> dict:
    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("hypotheses and references must be non-empty and aligned")
    generated_words = np.asarray(
        [len(text.split()) for text in hypotheses], dtype=np.float64,
    )
    reference_words = np.asarray(
        [len(text.split()) for text in references], dtype=np.float64,
    )
    return {
        "bleu": sacrebleu.corpus_bleu(
            hypotheses, [references], lowercase=True, use_effective_order=True,
        ).score,
        "chrf2": sacrebleu.corpus_chrf(
            hypotheses, [references], word_order=2,
        ).score,
        "ter": sacrebleu.corpus_ter(
            hypotheses, [references], normalized=True, case_sensitive=False,
        ).score,
        "empty_rate_pct": (
            100.0 * sum(not text.strip() for text in hypotheses) / len(hypotheses)
        ),
        "unique_rate_pct": 100.0 * len(set(hypotheses)) / len(hypotheses),
        "mean_generated_words": float(generated_words.mean()),
        "mean_reference_words": float(reference_words.mean()),
        "length_ratio": float(
            generated_words.sum() / max(reference_words.sum(), 1.0)
        ),
    }


def _load_evaluation(run_dir: Path, step: int, expected_count: int) -> dict:
    eval_dir = run_dir / "evaluations" / f"checkpoint_{step}"
    generated_paths = sorted(eval_dir.glob("*/all_generated_*_*.jsonl"))
    metric_paths = sorted(eval_dir.glob("*/metrics.jsonl"))
    if len(generated_paths) != 1 or len(metric_paths) != 1:
        raise ValueError(
            f"{eval_dir} must contain exactly one generated file and one metrics file"
        )
    rows = _read_jsonl(generated_paths[0])
    if len(rows) != expected_count:
        raise ValueError(
            f"{generated_paths[0]} has {len(rows)} rows, expected {expected_count}"
        )
    if [row.get("id") for row in rows] != list(range(expected_count)):
        raise ValueError(f"{generated_paths[0]} IDs are not consecutive from zero")
    required_text = ("source", "reference", "generated")
    for row in rows:
        if any(not isinstance(row.get(field), str) for field in required_text):
            raise ValueError(
                f"{generated_paths[0]} lacks source/reference/generated text"
            )

    metric_rows = [
        row for row in _read_jsonl(metric_paths[0]) if row.get("step") == step
    ]
    if len(metric_rows) != 1:
        raise ValueError(
            f"{metric_paths[0]} must contain exactly one row for step {step}"
        )
    recorded = metric_rows[0]
    required_recorded = (
        "bleu", "rouge1", "rouge2", "rougeL", "num_samples",
        "generation_seconds", "decode_seconds", "samples_per_second",
    )
    if any(not _finite(recorded.get(field)) for field in required_recorded):
        raise ValueError(
            f"{metric_paths[0]} lacks finite evaluation/resource metrics"
        )
    if recorded.get("timing_scope") != "cuda_synchronized_sampler_and_decoder":
        raise ValueError(f"{metric_paths[0]} does not use synchronized CUDA timing")
    if int(recorded["num_samples"]) != expected_count:
        raise ValueError(f"{metric_paths[0]} records the wrong sample count")

    hypotheses = [row["generated"] for row in rows]
    references = [row["reference"] for row in rows]
    sources = [row["source"] for row in rows]
    recomputed = _metric_bundle(hypotheses, references)
    if not math.isclose(
        recomputed["bleu"], recorded["bleu"], rel_tol=0.0, abs_tol=1e-8,
    ):
        raise ValueError(f"{metric_paths[0]} BLEU does not match generated artifacts")
    return {
        "step": step,
        "rows": rows,
        "hypotheses": hypotheses,
        "references": references,
        "sources": sources,
        "recorded": recorded,
        "metrics": recomputed,
        "sampling_run": generated_paths[0].parent.name,
    }


def _validate_resolved_config(run_dir: Path, label: str, spec: dict):
    config_path = run_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("model") != spec["config_model"]:
        raise ValueError(f"{config_path} does not describe {label}")
    if config.get("batch_size") != 12 or config.get("seed") != 42:
        raise ValueError(f"{config_path} has the wrong batch size or seed")
    if config.get("init_from") is not None:
        raise ValueError(f"{config_path} is not a from-scratch run")
    expected_inner_steps = spec["wonn_inner_steps"]
    if expected_inner_steps is not None:
        if config.get("wonn_num_inner_steps") != expected_inner_steps:
            raise ValueError(
                f"{config_path} is not the declared {label} architecture"
            )
    for field in (
        "data_revision", "eval_data_revision", "encoder_revision",
        "tokenizer_revision",
    ):
        if not config.get(field):
            raise ValueError(f"{config_path} does not pin {field}")
    return config


def _load_run(
    run_dir: Path, label: str, spec: dict, stage: dict,
    verify_checkpoints: bool,
) -> dict:
    target_step = stage["target_step"]
    completion_path = run_dir / "training_complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"missing {completion_path}")
    completion = _read_json(completion_path)
    if (
        completion.get("status") != "complete"
        or completion.get("completed_optimizer_step") != target_step
    ):
        raise ValueError(f"{completion_path} is not a complete {target_step}-step run")
    if completion.get("model") != spec["config_model"]:
        raise ValueError(f"{completion_path} records the wrong model factory")
    expected_samples_seen = target_step * 12
    if (
        completion.get("effective_batch_size") != 12
        or completion.get("samples_seen") != expected_samples_seen
    ):
        raise ValueError(
            f"{completion_path} does not use the paired batch/sample budget"
        )
    config = _validate_resolved_config(run_dir, label, spec)

    checkpoint_audits = []
    for step in stage["checkpoint_steps"]:
        checkpoint = run_dir / f"checkpoint_{step}"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty {checkpoint}")
        if verify_checkpoints:
            checkpoint_audits.append(_audit_checkpoint(checkpoint, step))

    metric_rows = _read_jsonl(run_dir / "train_metrics.jsonl")
    steps = [row.get("optimizer_step") for row in metric_rows]
    if steps != sorted(set(steps)) or not steps or steps[-1] != target_step:
        raise ValueError(
            f"{run_dir}/train_metrics.jsonl is not unique, ordered, and complete"
        )
    required_training = (
        "loss", "l2_loss", "ce_loss", "lr", "samples_seen",
        "elapsed_training_seconds", "samples_per_second",
    )
    if any(
        any(not _finite(row.get(field)) for field in required_training)
        for row in metric_rows
    ):
        raise ValueError(f"{run_dir}/train_metrics.jsonl has incomplete metrics")

    post_warmup = [
        row for row in metric_rows if row["optimizer_step"] >= 5000
    ]
    evaluations = {
        step: _load_evaluation(
            run_dir, step, stage["expected_samples"][step],
        )
        for step in stage["evaluation_steps"]
    }
    return {
        "label": label,
        "run_dir": run_dir,
        "config": config,
        "completion": completion,
        "checkpoint_audits": checkpoint_audits,
        "training_rows": metric_rows,
        "training_summary": {
            "model": label,
            "parameters": completion.get("model_parameters"),
            "elapsed_training_seconds": completion["elapsed_training_seconds"],
            "median_samples_per_second_post_warmup": statistics.median(
                row["samples_per_second"] for row in post_warmup
            ),
            "peak_allocated_cuda_mib": completion.get("peak_allocated_cuda_mib"),
            "final_loss": metric_rows[-1]["loss"],
            "final_l2_loss": metric_rows[-1]["l2_loss"],
            "final_ce_loss": metric_rows[-1]["ce_loss"],
        },
        "evaluations": evaluations,
    }


def _paired_bootstrap(elf_eval: dict, wonn_eval: dict, resamples: int, seed: int):
    if elf_eval["sources"] != wonn_eval["sources"]:
        raise ValueError("ELF and WONN evaluation sources are not paired")
    if elf_eval["references"] != wonn_eval["references"]:
        raise ValueError("ELF and WONN evaluation references are not paired")
    references = np.asarray(elf_eval["references"], dtype=object)
    elf_hypotheses = np.asarray(elf_eval["hypotheses"], dtype=object)
    wonn_hypotheses = np.asarray(wonn_eval["hypotheses"], dtype=object)
    rng = np.random.default_rng(seed)
    bleu_deltas = np.empty(resamples, dtype=np.float64)
    chrf_deltas = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.integers(0, len(references), size=len(references))
        refs = references[sample].tolist()
        elf = elf_hypotheses[sample].tolist()
        wonn = wonn_hypotheses[sample].tolist()
        bleu_deltas[index] = (
            sacrebleu.corpus_bleu(
                wonn, [refs], lowercase=True, use_effective_order=True,
            ).score
            - sacrebleu.corpus_bleu(
                elf, [refs], lowercase=True, use_effective_order=True,
            ).score
        )
        chrf_deltas[index] = (
            sacrebleu.corpus_chrf(wonn, [refs], word_order=2).score
            - sacrebleu.corpus_chrf(elf, [refs], word_order=2).score
        )

    def summarize(values):
        return {
            "mean": float(values.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "probability_delta_gt_zero": float((values > 0).mean()),
        }

    return {
        "resamples": resamples,
        "seed": seed,
        "bleu_delta_wonn_minus_elf": summarize(bleu_deltas),
        "chrf2_delta_wonn_minus_elf": summarize(chrf_deltas),
    }


def _evaluate_gate(runs: dict) -> dict:
    final_metrics = {
        label: run["evaluations"][20000]["metrics"]
        for label, run in runs.items()
    }
    improving_models = []
    for label, run in runs.items():
        at_10k = run["evaluations"][10000]["metrics"]
        at_20k = run["evaluations"][20000]["metrics"]
        if at_20k["bleu"] > at_10k["bleu"] and at_20k["chrf2"] > at_10k["chrf2"]:
            improving_models.append(label)
    checks = {
        "all_20k_empty_rates_at_most_10_pct": all(
            metrics["empty_rate_pct"] <= 10.0
            for metrics in final_metrics.values()
        ),
        "at_least_one_20k_bleu_at_least_1": any(
            metrics["bleu"] >= 1.0 for metrics in final_metrics.values()
        ),
        "at_least_one_model_improves_bleu_and_chrf2_from_10k": bool(
            improving_models
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "improving_models": improving_models,
        "thresholds": {
            "maximum_empty_rate_pct_per_model": 10.0,
            "minimum_bleu_for_at_least_one_model": 1.0,
            "trend": "at least one model improves both BLEU and chrF++ from 10K to 20K",
        },
    }


def analyze(
    root: Path, output_dir: Path, stage_name: str,
    verify_checkpoints: bool, bootstrap_resamples: int,
) -> dict:
    if stage_name not in STAGES:
        raise ValueError(f"unknown stage {stage_name!r}")
    stage = STAGES[stage_name]
    manifest_path = root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "ready":
        raise ValueError(f"{manifest_path} is not ready")

    runs = {
        label: _load_run(
            root / spec["directory"], label, spec, stage, verify_checkpoints,
        )
        for label, spec in MODEL_SPECS.items()
    }
    evaluation_rows = []
    for label, run in runs.items():
        for step, evaluation in run["evaluations"].items():
            recorded = evaluation["recorded"]
            evaluation_rows.append({
                "model": label,
                "step": step,
                "num_samples": len(evaluation["rows"]),
                **evaluation["metrics"],
                **{field: recorded.get(field) for field in (
                    "rouge1", "rouge2", "rougeL", "generation_seconds",
                    "decode_seconds", "samples_per_second",
                    "peak_allocated_cuda_mib", "timing_scope",
                )},
            })

    terminal_step = stage["target_step"]
    terminal_elf = runs["ELF-B"]["evaluations"][terminal_step]
    terminal_wonn = runs["WONN-L6T3"]["evaluations"][terminal_step]
    bootstrap = _paired_bootstrap(
        terminal_elf, terminal_wonn,
        resamples=bootstrap_resamples, seed=20260818,
    )
    payload = {
        "status": "complete",
        "stage": stage_name,
        "experiment_manifest": manifest,
        "comparison_contract": {
            "training_seed": 42,
            "effective_batch_size": 12,
            "optimizer_steps": terminal_step,
            "samples_seen_per_model": terminal_step * 12,
            "checkpoint_steps": list(stage["checkpoint_steps"]),
            "evaluation_steps": list(stage["evaluation_steps"]),
            "primary_metric": "case-insensitive sacreBLEU",
            "secondary_metrics": [
                "chrF++", "TER", "empty rate", "length ratio",
            ],
        },
        "training": {
            label: run["training_summary"] for label, run in runs.items()
        },
        "evaluations": evaluation_rows,
        "paired_terminal_bootstrap": bootstrap,
        "checkpoint_audits": {
            label: run["checkpoint_audits"] for label, run in runs.items()
        },
    }
    if stage_name == "gate20k":
        payload["gate"] = _evaluate_gate(runs)

    _write_json(output_dir / "comparison.json", payload)
    _write_csv(
        output_dir / "training_summary.csv",
        [run["training_summary"] for run in runs.values()],
        [
            "model", "parameters", "elapsed_training_seconds",
            "median_samples_per_second_post_warmup", "peak_allocated_cuda_mib",
            "final_loss", "final_l2_loss", "final_ce_loss",
        ],
    )
    evaluation_fields = [
        "model", "step", "num_samples", "bleu", "chrf2", "ter",
        "empty_rate_pct", "unique_rate_pct", "mean_generated_words",
        "mean_reference_words", "length_ratio", "rouge1", "rouge2", "rougeL",
        "generation_seconds", "decode_seconds", "samples_per_second",
        "peak_allocated_cuda_mib", "timing_scope",
    ]
    _write_csv(output_dir / "evaluation_summary.csv", evaluation_rows, evaluation_fields)

    terminal_rows = {
        row["model"]: row
        for row in evaluation_rows if row["step"] == terminal_step
    }
    markdown = [
        f"# Phase 5 WMT14 {stage_name} paired comparison\n",
        f"Both models used seed 42, effective batch 12, and {terminal_step} optimizer steps.\n",
        f"\n## {terminal_step}-step metrics\n",
        "| Model | BLEU | chrF++ | TER | Empty % | Length ratio | Samples |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for label in MODEL_SPECS:
        row = terminal_rows[label]
        markdown.append(
            f"| {label} | {row['bleu']:.4f} | {row['chrf2']:.4f} "
            f"| {row['ter']:.4f} | {row['empty_rate_pct']:.2f} "
            f"| {row['length_ratio']:.4f} | {row['num_samples']} |\n"
        )
    if stage_name == "gate20k":
        gate = payload["gate"]
        markdown.extend([
            "\n## 20K gate\n",
            f"Gate passed: **{gate['passed']}**\n",
        ])
        for name, passed in gate["checks"].items():
            markdown.append(f"- {name}: {passed}\n")
    (output_dir / "comparison.md").write_text("".join(markdown), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/phase5/redesign_v2/formal50k"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    result = analyze(
        args.root, args.output_dir, args.stage,
        args.verify_checkpoints, args.bootstrap_resamples,
    )
    print(json.dumps({
        "status": result["status"],
        "stage": result["stage"],
        "gate_passed": result.get("gate", {}).get("passed"),
        "output": str((args.output_dir / "comparison.json").resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

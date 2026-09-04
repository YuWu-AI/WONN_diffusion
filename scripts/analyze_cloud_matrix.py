#!/usr/bin/env python
"""Validate and summarize the four-model WMT14 100K cloud matrix."""

import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path

import numpy as np
import sacrebleu
import yaml


CHECKPOINT_STEPS = (5000, 10000, 25000, 40000, 60000, 80000, 100000)
EFFECTIVE_BATCH = 512
GRAD_ACCUM_STEPS = 32
MICRO_BATCH = 16
WARMUP_STEPS = 5000
EXPECTED_SAMPLES = 1000

MODEL_SPECS = {
    "E0": {
        "label": "E0 ELF-B",
        "slug": "e0",
        "config": "train_de-en-ELF-B-E0.yml",
        "model": "ELF-B",
        "lr": 0.002,
        "layers": None,
    },
    "W0": {
        "label": "W0 WONN-L12/K768/T3",
        "slug": "w0",
        "config": "train_de-en-WONN-L12K768T3-W0.yml",
        "model": "ELF-WONN-B",
        "lr": 0.001,
        "layers": 12,
    },
    "W1": {
        "label": "W1 WONN-L6/K768/T3",
        "slug": "w1",
        "config": "train_de-en-WONN-L6K768T3-W1.yml",
        "model": "ELF-WONN-B",
        "lr": 0.001,
        "layers": 6,
    },
    "W2": {
        "label": "W2 WONN-L9/K768/T3",
        "slug": "w2",
        "config": "train_de-en-WONN-L9K768T3-W2.yml",
        "model": "ELF-WONN-B",
        "lr": 0.001,
        "layers": 9,
    },
}
MODEL_SPECIFIC_FIELDS = {"model", "lr", "output_dir", "resume"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _shared_config_mismatches(configs: dict[str, dict]) -> dict:
    ignored = MODEL_SPECIFIC_FIELDS | {
        field
        for config in configs.values()
        for field in config
        if field.startswith("wonn_")
    }
    fields = set().union(*(config.keys() for config in configs.values())) - ignored
    baseline = configs["E0"]
    return {
        field: {key: config.get(field) for key, config in configs.items()}
        for field in sorted(fields)
        if any(config.get(field) != baseline.get(field) for config in configs.values())
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_matrix_configs(config_dir: Path) -> dict:
    configs = {
        key: _load_yaml(config_dir / spec["config"])
        for key, spec in MODEL_SPECS.items()
    }
    save_steps = ",".join(str(step) for step in CHECKPOINT_STEPS)
    fixed = {
        "global_batch_size": EFFECTIVE_BATCH,
        "batch_size": None,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "epochs": 100,
        "max_optimizer_steps": CHECKPOINT_STEPS[-1],
        "stop_optimizer_steps": None,
        "lr_schedule": "constant",
        "warmup_steps": WARMUP_STEPS,
        "optimizer": "muon",
        "num_samples": EXPECTED_SAMPLES,
        "save_optimizer_steps": save_steps,
        "save_freq": 0,
        "eval_freq": 0,
        "final_eval": False,
        "resume": None,
        "init_from": None,
        "seed": 42,
    }
    for key, spec in MODEL_SPECS.items():
        config = configs[key]
        wrong = {
            field: {"actual": config.get(field), "expected": expected}
            for field, expected in fixed.items()
            if config.get(field) != expected
        }
        if config.get("model") != spec["model"]:
            wrong["model"] = {
                "actual": config.get("model"), "expected": spec["model"],
            }
        if config.get("lr") != spec["lr"]:
            wrong["lr"] = {"actual": config.get("lr"), "expected": spec["lr"]}
        if wrong:
            raise ValueError(f"{key} violates the fixed matrix contract: {wrong}")
        if spec["layers"] is not None and (
            config.get("wonn_num_layers") != spec["layers"]
            or config.get("wonn_num_oscillators") != 768
            or config.get("wonn_num_inner_steps") != 3
            or config.get("wonn_num_heads") != 12
            or config.get("wonn_step_init") != 0.1
            or config.get("wonn_step_max") != 0.25
        ):
            raise ValueError(f"{key} has the wrong WONN architecture")

    baseline = configs["E0"]
    mismatches = _shared_config_mismatches(configs)
    if mismatches:
        raise ValueError(f"matrix configs differ outside model/LR: {mismatches}")

    return {
        "status": "valid",
        "terminal_optimizer_step": CHECKPOINT_STEPS[-1],
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "effective_batch_size": EFFECTIVE_BATCH,
        "world_size_per_model": 1,
        "batch_size_per_device": MICRO_BATCH,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "warmup_optimizer_steps": WARMUP_STEPS,
        "evaluation_samples": EXPECTED_SAMPLES,
        "learning_rates": {key: spec["lr"] for key, spec in MODEL_SPECS.items()},
        "seed": 42,
        "data_revision": baseline["data_revision"],
        "eval_data_revision": baseline["eval_data_revision"],
        "encoder_revision": baseline["encoder_revision"],
        "tokenizer_revision": baseline["tokenizer_revision"],
    }


def _metric_bundle(hypotheses: list[str], references: list[str]) -> dict:
    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("hypotheses and references must be non-empty and aligned")
    generated_words = np.asarray([len(text.split()) for text in hypotheses], dtype=np.float64)
    reference_words = np.asarray([len(text.split()) for text in references], dtype=np.float64)
    return {
        "bleu": sacrebleu.corpus_bleu(
            hypotheses, [references], lowercase=True, use_effective_order=True,
        ).score,
        "chrf2": sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score,
        "ter": sacrebleu.corpus_ter(
            hypotheses, [references], normalized=True, case_sensitive=False,
        ).score,
        "empty_rate_pct": 100.0 * sum(not text.strip() for text in hypotheses) / len(hypotheses),
        "unique_rate_pct": 100.0 * len(set(hypotheses)) / len(hypotheses),
        "length_ratio": float(generated_words.sum() / max(reference_words.sum(), 1.0)),
    }


def _load_evaluation(run_dir: Path, step: int, expected_count: int) -> dict:
    eval_dir = run_dir / "evaluations" / f"checkpoint_{step}"
    generated_paths = sorted(eval_dir.glob("*/all_generated_*_*.jsonl"))
    metric_paths = sorted(eval_dir.glob("*/metrics.jsonl"))
    if len(generated_paths) != 1 or len(metric_paths) != 1:
        raise ValueError(f"{eval_dir} must contain one generated file and one metrics file")
    rows = _read_jsonl(generated_paths[0])
    if len(rows) != expected_count or [row.get("id") for row in rows] != list(range(expected_count)):
        raise ValueError(f"{generated_paths[0]} does not contain the expected aligned rows")
    for row in rows:
        if any(not isinstance(row.get(field), str) for field in ("source", "reference", "generated")):
            raise ValueError(f"{generated_paths[0]} lacks source/reference/generated text")
    recorded_rows = [row for row in _read_jsonl(metric_paths[0]) if row.get("step") == step]
    if len(recorded_rows) != 1:
        raise ValueError(f"{metric_paths[0]} must contain one row for optimizer step {step}")
    recorded = recorded_rows[0]
    required = (
        "bleu", "rouge1", "rouge2", "rougeL", "num_samples",
        "generation_seconds", "decode_seconds", "samples_per_second",
        "peak_allocated_cuda_mib",
    )
    if any(not _finite(recorded.get(field)) for field in required):
        raise ValueError(f"{metric_paths[0]} lacks finite evaluation/resource metrics")
    if recorded["num_samples"] != expected_count:
        raise ValueError(f"{metric_paths[0]} has the wrong sample count")
    if (
        recorded["generation_seconds"] < 0
        or recorded["decode_seconds"] < 0
        or recorded["samples_per_second"] <= 0
        or recorded["peak_allocated_cuda_mib"] <= 0
    ):
        raise ValueError(f"{metric_paths[0]} has invalid evaluation/resource metrics")
    if recorded.get("timing_scope") != "cuda_synchronized_sampler_and_decoder":
        raise ValueError(f"{metric_paths[0]} does not use synchronized CUDA timing")
    hypotheses = [row["generated"] for row in rows]
    references = [row["reference"] for row in rows]
    metrics = _metric_bundle(hypotheses, references)
    if not math.isclose(metrics["bleu"], recorded["bleu"], rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{metric_paths[0]} BLEU does not match generated artifacts")
    return {
        "inputs": ([row["source"] for row in rows], references),
        "metrics": metrics,
        "recorded": recorded,
    }


def _training_rows(run_dir: Path) -> dict[int, dict]:
    path = run_dir / "train_metrics.jsonl"
    rows = {}
    previous_step = previous_samples = -1
    for line_number, row in enumerate(_read_jsonl(path), 1):
        step, samples = row.get("optimizer_step"), row.get("samples_seen")
        if not isinstance(step, int) or not isinstance(samples, int):
            raise ValueError(f"{path}:{line_number} lacks integer step/samples_seen")
        if step <= previous_step or samples <= previous_samples:
            raise ValueError(f"{path} is not strictly monotonic at line {line_number}")
        rows[step] = row
        previous_step, previous_samples = step, samples
    missing = [step for step in CHECKPOINT_STEPS if step not in rows]
    if missing:
        raise ValueError(f"{path} lacks checkpoint-step metrics: {missing}")
    return rows


def _panel_svg(
    title: str, x_field: str, y_field: str, rows: list[dict], x: int, y: int,
) -> str:
    width, height = 470, 285
    left, right, top, bottom = x + 58, x + width - 18, y + 35, y + height - 45
    series = [
        (key, [(row[x_field], row[y_field])
               for row in rows if row["model_key"] == key])
        for key in MODEL_SPECS
    ]
    points = [point for _, values in series for point in values]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    padding = max((y_max - y_min) * 0.08, 0.1)
    y_min, y_max = y_min - padding, y_max + padding
    px = lambda value: left + (value - x_min) * (right - left) / max(x_max - x_min, 1)
    py = lambda value: bottom - (value - y_min) * (bottom - top) / (y_max - y_min)
    colors = ("#111827", "#dc2626", "#2563eb", "#16a34a")
    parts = [
        f'<text x="{x + width / 2:.1f}" y="{y + 18}" text-anchor="middle" font-size="15" font-weight="600">{escape(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#444"/>',
        f'<text x="{(left + right) / 2:.1f}" y="{y + height - 8}" text-anchor="middle" font-size="12">{escape(x_field.replace("_", " "))}</text>',
        f'<text x="{left}" y="{bottom + 17}" text-anchor="middle" font-size="10">{x_min:,.0f}</text>',
        f'<text x="{right}" y="{bottom + 17}" text-anchor="middle" font-size="10">{x_max:,.0f}</text>',
        f'<text x="{left - 7}" y="{bottom + 4}" text-anchor="end" font-size="10">{y_min:.1f}</text>',
        f'<text x="{left - 7}" y="{top + 4}" text-anchor="end" font-size="10">{y_max:.1f}</text>',
    ]
    for index, (_, values) in enumerate(series):
        encoded = " ".join(f"{px(xv):.2f},{py(yv):.2f}" for xv, yv in values)
        parts.append(f'<polyline points="{encoded}" fill="none" stroke="{colors[index]}" stroke-width="2.5"/>')
        parts.extend(f'<circle cx="{px(xv):.2f}" cy="{py(yv):.2f}" r="3" fill="{colors[index]}"/>' for xv, yv in values)
    return "".join(parts)


def _quality_chart(rows: list[dict]) -> str:
    panels = [
        _panel_svg("BLEU vs optimizer step", "optimizer_step", "bleu", rows, 20, 90),
        _panel_svg("chrF++ vs optimizer step", "optimizer_step", "chrf2", rows, 530, 90),
        _panel_svg("BLEU vs samples seen", "samples_seen", "bleu", rows, 20, 390),
        _panel_svg("chrF++ vs samples seen", "samples_seen", "chrf2", rows, 530, 390),
    ]
    colors = ("#111827", "#dc2626", "#2563eb", "#16a34a")
    legend = []
    for index, (key, spec) in enumerate(MODEL_SPECS.items()):
        x = 140 + index * 220
        legend.append(f'<line x1="{x}" y1="58" x2="{x + 28}" y2="58" stroke="{colors[index]}" stroke-width="3"/>')
        legend.append(f'<text x="{x + 35}" y="62" font-size="11">{escape(spec["label"])}</text>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="715" viewBox="0 0 1020 715">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="510" y="28" text-anchor="middle" font-size="20" font-weight="700">WMT14 100K architecture screen</text>'
        + "".join(legend + panels) + "</svg>\n"
    )


def analyze_matrix(run_root: Path, output_dir: Path, expected_samples: int = EXPECTED_SAMPLES) -> dict:
    configs, completions, training = {}, {}, {}
    for key, spec in MODEL_SPECS.items():
        run_dir = run_root / spec["slug"]
        configs[key] = _load_yaml(run_dir / "config.yml")
        completions[key] = _read_json(run_dir / "training_complete.json")
        training[key] = _training_rows(run_dir)
        resolved_expected = {
            "model": spec["model"],
            "global_batch_size": EFFECTIVE_BATCH,
            "batch_size": MICRO_BATCH,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "epochs": 100,
            "max_optimizer_steps": CHECKPOINT_STEPS[-1],
            "stop_optimizer_steps": None,
            "save_optimizer_steps": ",".join(str(step) for step in CHECKPOINT_STEPS),
            "lr": spec["lr"],
            "lr_schedule": "constant",
            "warmup_steps": WARMUP_STEPS,
            "optimizer": "muon",
            "compile_train": True,
            "gradient_checkpointing": False,
            "num_samples": EXPECTED_SAMPLES,
            "online_eval": True,
            "save_freq": 0,
            "eval_freq": 0,
            "final_eval": False,
            "seed": 42,
        }
        wrong_config = {
            field: {"actual": configs[key].get(field), "expected": expected}
            for field, expected in resolved_expected.items()
            if configs[key].get(field) != expected
        }
        if wrong_config:
            raise ValueError(f"{key} resolved config violates the matrix contract: {wrong_config}")
        if spec["layers"] is not None and (
            configs[key].get("wonn_num_layers") != spec["layers"]
            or configs[key].get("wonn_num_oscillators") != 768
            or configs[key].get("wonn_num_inner_steps") != 3
            or configs[key].get("wonn_num_heads") != 12
            or configs[key].get("wonn_step_init") != 0.1
            or configs[key].get("wonn_step_max") != 0.25
        ):
            raise ValueError(f"{key} resolved config has the wrong WONN architecture")
        completion = completions[key]
        if completion.get("status") != "complete" or completion.get("model") != spec["model"]:
            raise ValueError(f"{key} training is not complete for {spec['model']}")
        expected_completion = {
            "completed_optimizer_step": CHECKPOINT_STEPS[-1],
            "effective_batch_size": EFFECTIVE_BATCH,
            "world_size": 1,
            "batch_size_per_device": MICRO_BATCH,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "learning_rate": spec["lr"],
            "samples_seen": CHECKPOINT_STEPS[-1] * EFFECTIVE_BATCH,
        }
        wrong = {
            field: {"actual": completion.get(field), "expected": expected}
            for field, expected in expected_completion.items()
            if completion.get(field) != expected
        }
        if wrong:
            raise ValueError(f"{key} completion marker violates the matrix contract: {wrong}")
        for field in (
            "model_parameters", "elapsed_training_seconds", "peak_allocated_cuda_mib",
        ):
            if not _finite(completion.get(field)) or completion[field] <= 0:
                raise ValueError(f"{key} completion marker lacks valid {field}")
        if configs[key].get("resume") is not None:
            if Path(configs[key]["resume"]).resolve() != run_dir.resolve():
                raise ValueError(f"{key} resumed from outside its own run directory")
            if not (run_dir / "training_resumed.json").is_file():
                raise ValueError(f"{key} lacks in-place resume evidence")

    mismatches = _shared_config_mismatches(configs)
    if mismatches:
        raise ValueError(f"resolved matrix configs differ outside model/LR: {mismatches}")

    rows, paired_inputs = [], None
    for key, spec in MODEL_SPECS.items():
        run_dir = run_root / spec["slug"]
        for step in CHECKPOINT_STEPS:
            checkpoint = run_dir / f"checkpoint_{step}"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise FileNotFoundError(f"missing or empty checkpoint {checkpoint}")
            marker = _read_json(run_dir / "evaluations" / f"checkpoint_{step}" / "evaluation_complete.json")
            if marker != {
                "status": "complete", "model": spec["label"],
                "step": step, "num_samples": expected_samples,
            }:
                raise ValueError(f"{key} checkpoint {step} evaluation is incomplete")
            evaluation = _load_evaluation(run_dir, step, expected_samples)
            if paired_inputs is None:
                paired_inputs = evaluation["inputs"]
            elif paired_inputs != evaluation["inputs"]:
                raise ValueError(f"evaluation inputs differ at {key} checkpoint {step}")
            train_row = training[key][step]
            required = ("loss", "l2_loss", "ce_loss", "lr", "samples_seen", "elapsed_training_seconds", "samples_per_second")
            if any(not _finite(train_row.get(field)) for field in required):
                raise ValueError(f"{key} has non-finite training metrics at step {step}")
            expected_train_step = step * GRAD_ACCUM_STEPS
            if train_row.get("step") != expected_train_step:
                raise ValueError(
                    f"{key} checkpoint {step} has train step {train_row.get('step')}, "
                    f"expected {expected_train_step}"
                )
            if train_row["samples_seen"] != step * EFFECTIVE_BATCH:
                raise ValueError(f"{key} checkpoint {step} has inconsistent samples_seen")
            if not math.isclose(train_row["lr"], spec["lr"], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{key} checkpoint {step} has the wrong learning rate")
            metrics, recorded = evaluation["metrics"], evaluation["recorded"]
            rows.append({
                "model_key": key,
                "model": spec["label"],
                "optimizer_step": step,
                "samples_seen": train_row["samples_seen"],
                "loss": train_row["loss"],
                "l2_loss": train_row["l2_loss"],
                "ce_loss": train_row["ce_loss"],
                "learning_rate": train_row["lr"],
                "training_samples_per_second": train_row["samples_per_second"],
                "elapsed_training_seconds": train_row["elapsed_training_seconds"],
                **metrics,
                "evaluation_samples_per_second": recorded["samples_per_second"],
                "sampler_latency_seconds": recorded["generation_seconds"],
                "decode_latency_seconds": recorded["decode_seconds"],
                "evaluation_peak_allocated_cuda_mib": recorded.get("peak_allocated_cuda_mib"),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_by_checkpoint.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "quality_curves.svg").write_text(_quality_chart(rows), encoding="utf-8")
    terminal = {
        key: next(row for row in rows if row["model_key"] == key and row["optimizer_step"] == CHECKPOINT_STEPS[-1])
        for key in MODEL_SPECS
    }
    payload = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_contract": {
            "terminal_optimizer_step": CHECKPOINT_STEPS[-1],
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "samples_per_model_per_checkpoint": expected_samples,
            "effective_batch_size": EFFECTIVE_BATCH,
            "seed": 42,
            "learning_rates": {key: spec["lr"] for key, spec in MODEL_SPECS.items()},
        },
        "models": {
            key: {
                "label": spec["label"],
                "parameters": completions[key].get("model_parameters"),
                "elapsed_training_seconds": completions[key].get("elapsed_training_seconds"),
                "peak_allocated_cuda_mib": completions[key].get("peak_allocated_cuda_mib"),
                "terminal_metrics": terminal[key],
            }
            for key, spec in MODEL_SPECS.items()
        },
        "metrics_by_checkpoint": rows,
        "artifacts": {
            "table_csv": "metrics_by_checkpoint.csv",
            "quality_chart": "quality_curves.svg",
            "markdown": "comparison.md",
        },
    }
    _write_json(output_dir / "comparison.json", payload)
    markdown = [
        "# WMT14 100K architecture screen\n\n",
        "All four models use effective batch 512, seed 42 and the same 1,000 validation examples.\n\n",
        "| Model | BLEU | chrF++ | TER | Empty % | Unique % | Length ratio |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for key, spec in MODEL_SPECS.items():
        row = terminal[key]
        markdown.append(
            f"| {spec['label']} | {row['bleu']:.4f} | {row['chrf2']:.4f} | "
            f"{row['ter']:.4f} | {row['empty_rate_pct']:.2f} | "
            f"{row['unique_rate_pct']:.2f} | {row['length_ratio']:.4f} |\n"
        )
    markdown.extend(["\n## Curves\n\n", "![Quality curves](quality_curves.svg)\n"])
    (output_dir / "comparison.md").write_text("".join(markdown), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-configs")
    validate.add_argument("--config-dir", type=Path, required=True)
    validate.add_argument("--output", type=Path)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--run-root", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--expected-samples", type=int, default=EXPECTED_SAMPLES)
    args = parser.parse_args()
    if args.command == "validate-configs":
        payload = validate_matrix_configs(args.config_dir)
        if args.output:
            _write_json(args.output, payload)
    else:
        payload = analyze_matrix(args.run_root, args.output_dir, args.expected_samples)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

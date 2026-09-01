#!/usr/bin/env python
"""Build the 10K--90K ELF/WONN BLEU and chrF++ comparison artifacts."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_phase5_50k import _load_evaluation, _metric_bundle
from summarize_phase5_run import _polyline_svg


EXPECTED_STEPS = tuple(range(10000, 90001, 10000))
COMPARISON_SAMPLES = 1000
ELF_LABEL = "Transformer ELF-B"
WONN_LABEL = "WONN-L12K768T3"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_first_n(run_dir: Path, step: int, count: int) -> dict:
    marker_path = run_dir / "evaluations" / f"checkpoint_{step}" / "evaluation_complete.json"
    marker = _read_json(marker_path)
    recorded_count = marker.get("num_samples")
    if (
        marker.get("status") != "complete"
        or marker.get("step") != step
        or not isinstance(recorded_count, int)
        or isinstance(recorded_count, bool)
        or recorded_count < count
    ):
        raise ValueError(f"{marker_path} does not contain at least {count} completed samples")
    evaluation = _load_evaluation(run_dir, step, recorded_count)
    rows = evaluation["rows"][:count]
    hypotheses = [row["generated"] for row in rows]
    references = [row["reference"] for row in rows]
    return {
        "sources": [row["source"] for row in rows],
        "references": references,
        "metrics": _metric_bundle(hypotheses, references),
    }


def analyze_curve(
    wonn_run_dir: Path,
    elf_run_dir: Path,
    output_dir: Path,
    steps: tuple[int, ...] = EXPECTED_STEPS,
    comparison_samples: int = COMPARISON_SAMPLES,
) -> dict:
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("steps must be non-empty and positive")
    if comparison_samples <= 0:
        raise ValueError("comparison_samples must be positive")

    completion = _read_json(wonn_run_dir / "training_complete.json")
    if (
        completion.get("status") != "complete"
        or completion.get("completed_optimizer_step") != max(steps)
    ):
        raise ValueError("WONN training is not complete at the requested terminal step")
    config = yaml.safe_load((wonn_run_dir / "config.yml").read_text(encoding="utf-8")) or {}
    expected_architecture = {
        "model": "ELF-WONN-B",
        "wonn_num_layers": 12,
        "wonn_num_oscillators": 768,
        "wonn_num_inner_steps": 3,
        "wonn_num_heads": 12,
        "batch_size": 12,
        "seed": 42,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_architecture.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"WONN resolved config mismatch: {mismatches}")

    rows = []
    for step in steps:
        checkpoint = wonn_run_dir / f"checkpoint_{step}"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty {checkpoint}")
        elf = _load_first_n(elf_run_dir, step, comparison_samples)
        wonn = _load_first_n(wonn_run_dir, step, comparison_samples)
        if elf["sources"] != wonn["sources"]:
            raise ValueError(f"ELF and WONN sources are not aligned at step {step}")
        if elf["references"] != wonn["references"]:
            raise ValueError(f"ELF and WONN references are not aligned at step {step}")
        rows.append({
            "step": step,
            "comparison_samples": comparison_samples,
            "elf_bleu": elf["metrics"]["bleu"],
            "wonn_bleu": wonn["metrics"]["bleu"],
            "elf_chrf2": elf["metrics"]["chrf2"],
            "wonn_chrf2": wonn["metrics"]["chrf2"],
        })

    elf_parameters = None
    for candidate in (
        elf_run_dir / "provenance" / "training_complete_90000.json",
        elf_run_dir / "training_complete.json",
    ):
        if candidate.is_file():
            elf_parameters = _read_json(candidate).get("model_parameters")
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "step", "comparison_samples", "elf_bleu", "wonn_bleu",
        "elf_chrf2", "wonn_chrf2",
    ]
    with (output_dir / "metrics_by_step.csv").open(
        "w", encoding="utf-8", newline="",
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "bleu_by_step.svg").write_text(
        _polyline_svg(
            "WMT14 BLEU by optimizer step",
            "optimizer step",
            [
                (ELF_LABEL, [(row["step"], row["elf_bleu"]) for row in rows]),
                (WONN_LABEL, [(row["step"], row["wonn_bleu"]) for row in rows]),
            ],
        ),
        encoding="utf-8",
    )
    (output_dir / "chrf_by_step.svg").write_text(
        _polyline_svg(
            "WMT14 chrF++ by optimizer step",
            "optimizer step",
            [
                (ELF_LABEL, [(row["step"], row["elf_chrf2"]) for row in rows]),
                (WONN_LABEL, [(row["step"], row["wonn_chrf2"]) for row in rows]),
            ],
        ),
        encoding="utf-8",
    )

    payload = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_contract": {
            "dataset": "WMT14 de-en validation",
            "steps": list(steps),
            "samples_per_model_per_step": comparison_samples,
            "seed": 42,
            "bleu": "sacrebleu corpus_bleu lowercase effective_order",
            "chrf": "sacrebleu corpus_chrf word_order=2 (chrF++)",
            "elf_run_dir": str(elf_run_dir.resolve()),
            "wonn_run_dir": str(wonn_run_dir.resolve()),
        },
        "models": {
            ELF_LABEL: {"parameters": elf_parameters},
            WONN_LABEL: {"parameters": completion.get("model_parameters")},
        },
        "metrics_by_step": rows,
        "artifacts": {
            "table_csv": "metrics_by_step.csv",
            "bleu_chart": "bleu_by_step.svg",
            "chrf_chart": "chrf_by_step.svg",
            "markdown": "comparison.md",
        },
    }
    _write_json(output_dir / "comparison.json", payload)

    markdown = [
        "# WMT14 Transformer ELF-B vs WONN-L12K768T3\n\n",
        f"Each value is recomputed on the same first {comparison_samples} validation examples at that step.\n\n",
        "| Step | ELF BLEU | WONN BLEU | ELF chrF++ | WONN chrF++ |\n",
        "|---:|---:|---:|---:|---:|\n",
    ]
    markdown.extend(
        f"| {row['step']} | {row['elf_bleu']:.4f} | {row['wonn_bleu']:.4f} "
        f"| {row['elf_chrf2']:.4f} | {row['wonn_chrf2']:.4f} |\n"
        for row in rows
    )
    markdown.extend([
        "\n## Charts\n\n",
        "![BLEU by optimizer step](bleu_by_step.svg)\n\n",
        "![chrF++ by optimizer step](chrf_by_step.svg)\n",
    ])
    (output_dir / "comparison.md").write_text("".join(markdown), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wonn-run-dir", type=Path, required=True)
    parser.add_argument("--elf-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_curve(args.wonn_run_dir, args.elf_run_dir, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "output": str((args.output_dir / "comparison.json").resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Summarize the three Phase 5 mechanism runs into auditable tables."""

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_phase5_50k import _metric_bundle


RUNS = {
    "S-Base": "s_base",
    "S-Token": "s_token",
    "S-Token-Contrast": "s_token_contrast",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _generated_path(eval_dir: Path) -> Path:
    matches = list(eval_dir.glob("*/all_generated_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected one generated file under {eval_dir}")
    return matches[0]


def summarize(root: Path, output_dir: Path) -> dict:
    training_rows = []
    evaluation_rows = []
    diagnostic_rows = []
    for label, directory in RUNS.items():
        run_dir = root / directory
        completion = _read_json(run_dir / "training_complete.json")
        if completion.get("completed_optimizer_step") != 50000:
            raise ValueError(f"{label} training is not complete at 50k")
        metrics_by_step = {
            int(row["step"]): row
            for row in _read_jsonl(run_dir / "train_metrics.jsonl")
        }
        for step in (5000, 10000, 20000, 30000, 40000, 50000):
            row = metrics_by_step[step]
            training_rows.append({
                "model": label,
                **{key: row.get(key) for key in (
                    "step", "loss", "l2_loss", "ce_loss", "aux_loss",
                    "token_loss", "source_contrastive_loss", "aux_scale",
                    "samples_seen", "elapsed_training_seconds",
                )},
            })
            for split in ("train", "heldout"):
                diagnostic_path = (
                    run_dir / f"diagnostics/checkpoint_{step}/{split}/diagnostics.json"
                )
                diagnostic = _read_json(diagnostic_path)
                model_payload = diagnostic["models"][label]
                for time_row in model_payload["time_bins"]:
                    diagnostic_rows.append({
                        "model": label,
                        "step": step,
                        "split": split,
                        "probe": "time_bin",
                        **time_row,
                    })
                correct = model_payload["conditioning"]["correct"]
                zero = model_payload["conditioning"]["zero"]
                diagnostic_rows.append({
                    "model": label,
                    "step": step,
                    "split": split,
                    "probe": "conditioning",
                    "correct_bleu": correct["bleu"],
                    "zero_bleu": zero["bleu"],
                    "correct_chrf2": correct["chrf2"],
                    "zero_chrf2": zero["chrf2"],
                    "correct_minus_zero_chrf2": correct["chrf2"] - zero["chrf2"],
                })
                for name, decoder_row in model_payload["decoder_probe"].items():
                    if not isinstance(decoder_row, dict):
                        continue
                    diagnostic_rows.append({
                        "model": label,
                        "step": step,
                        "split": split,
                        "probe": name,
                        **decoder_row,
                    })
        for step in (20000, 40000, 50000):
            eval_dir = run_dir / f"evaluations/checkpoint_{step}"
            generated = _read_jsonl(_generated_path(eval_dir))
            metrics = _metric_bundle(
                [row["generated"] for row in generated],
                [row["reference"] for row in generated],
            )
            evaluation_rows.append({
                "model": label,
                "step": step,
                "num_samples": len(generated),
                **metrics,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "training": training_rows,
        "evaluation": evaluation_rows,
        "diagnostics": diagnostic_rows,
    }
    for name, rows in tables.items():
        _write_csv(output_dir / f"{name}.csv", rows)
    payload = {"status": "complete", "runs": RUNS, "tables": tables}
    temporary = output_dir / "mechanism_summary.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir / "mechanism_summary.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/phase5/redesign_v2/mechanism50k")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/phase5/redesign_v2/mechanism50k/analysis"),
    )
    args = parser.parse_args()
    result = summarize(args.root, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "output": str((args.output_dir / "mechanism_summary.json").resolve()),
    }))


if __name__ == "__main__":
    main()

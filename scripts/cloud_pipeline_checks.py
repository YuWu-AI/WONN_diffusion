#!/usr/bin/env python
"""Artifact and lifecycle checks for resumable cloud pipelines."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def checkpoint_is_usable(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _evaluation_artifacts(directory: Path, step: int, samples: int) -> bool:
    try:
        generated = list(directory.glob(f"*/all_generated_*_{step}.jsonl"))
        metrics = list(directory.glob("*/metrics.jsonl"))
        if (
            len(generated) != 1
            or len(metrics) != 1
            or metrics[0].stat().st_size == 0
        ):
            return False
        with generated[0].open("r", encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        if len(rows) != samples or any(
            not isinstance(row, dict)
            or not all(field in row for field in ("source", "reference", "generated"))
            for row in rows
        ):
            return False
        with metrics[0].open("r", encoding="utf-8") as source:
            metric_rows = [json.loads(line) for line in source if line.strip()]
        return any(
            isinstance(row, dict)
            and row.get("step") == step
            and row.get("num_samples") == samples
            for row in metric_rows
        )
    except (json.JSONDecodeError, OSError):
        return False


def evaluation_is_complete(
    directory: Path, label: str, step: int, samples: int
) -> bool:
    marker = _read_json(directory / "evaluation_complete.json")
    return bool(
        marker
        and marker.get("status") == "complete"
        and marker.get("model") == label
        and marker.get("step") == step
        and marker.get("num_samples") == samples
        and _evaluation_artifacts(directory, step, samples)
    )


def finalize_evaluation(
    directory: Path, label: str, step: int, samples: int
) -> bool:
    if not _evaluation_artifacts(directory, step, samples):
        return False
    _atomic_write_json(
        directory / "evaluation_complete.json",
        {
            "status": "complete",
            "model": label,
            "step": step,
            "num_samples": samples,
        },
    )
    return True


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_pipeline_status(
    output: Path,
    status: str,
    experiment_commit: str,
    runner_commit: str,
    stage: str,
    pid: int,
    exit_code: int | None,
) -> None:
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_commit": experiment_commit,
        "runner_commit": runner_commit,
        "stage": stage,
        "pid": pid,
    }
    if exit_code is not None:
        payload["exit_code"] = exit_code
    _atomic_write_json(output, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("path", type=Path)

    evaluation = subparsers.add_parser("evaluation")
    evaluation.add_argument("directory", type=Path)
    evaluation.add_argument("label")
    evaluation.add_argument("step", type=int)
    evaluation.add_argument("samples", type=int)

    finalize = subparsers.add_parser("finalize-evaluation")
    finalize.add_argument("directory", type=Path)
    finalize.add_argument("label")
    finalize.add_argument("step", type=int)
    finalize.add_argument("samples", type=int)

    status = subparsers.add_parser("status")
    status.add_argument("output", type=Path)
    status.add_argument("value", choices=("running", "failed", "complete"))
    status.add_argument("experiment_commit")
    status.add_argument("runner_commit")
    status.add_argument("stage")
    status.add_argument("pid", type=int)
    status.add_argument("--exit-code", type=int)

    args = parser.parse_args()
    if args.command == "checkpoint":
        return 0 if checkpoint_is_usable(args.path) else 1
    if args.command == "evaluation":
        return 0 if evaluation_is_complete(
            args.directory, args.label, args.step, args.samples
        ) else 1
    if args.command == "finalize-evaluation":
        return 0 if finalize_evaluation(
            args.directory, args.label, args.step, args.samples
        ) else 1
    write_pipeline_status(
        args.output,
        args.value,
        args.experiment_commit,
        args.runner_commit,
        args.stage,
        args.pid,
        args.exit_code,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

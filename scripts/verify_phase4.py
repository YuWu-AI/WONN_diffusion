#!/usr/bin/env python
"""Strict CUDA verification for Phase 4 learnability."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _test_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _test_ids(item)
        else:
            yield item.id()


def main() -> int:
    if not torch.cuda.is_available():
        print("Phase 4 verification requires a visible CUDA device.", file=sys.stderr)
        return 2

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(REPO_ROOT / "tests"), top_level_dir=str(REPO_ROOT)
    )
    test_ids = set(_test_ids(suite))
    required_test = (
        "tests.test_phase4_learnability.Phase4HarnessTest."
        "test_frozen_encoder_reuses_only_the_exact_real_batch"
    )
    if required_test not in test_ids:
        print("Phase 4 harness regression test was not discovered.", file=sys.stderr)
        return 2
    if len(test_ids) < 41:
        print(
            f"Phase 4 verification discovered {len(test_ids)} tests; expected at least 41.",
            file=sys.stderr,
        )
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print("Phase 4 verification does not allow skipped tests.", file=sys.stderr)
        return 1
    if not result.wasSuccessful():
        return 1

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_phase4_learnability.py"),
        "--output",
        str(REPO_ROOT / "outputs/phase4/learnability_metrics.json"),
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

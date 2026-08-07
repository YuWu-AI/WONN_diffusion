#!/usr/bin/env python
"""Run the Phase 2 contract suite and require CUDA coverage."""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    if not torch.cuda.is_available():
        print("Phase 2 verification requires a visible CUDA device.", file=sys.stderr)
        return 2

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(REPO_ROOT / "tests"),
        top_level_dir=str(REPO_ROOT),
    )
    test_count = suite.countTestCases()
    if test_count < 15:
        print(
            f"Phase 2 verification discovered {test_count} tests; expected at least 15.",
            file=sys.stderr,
        )
        return 2

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print("Phase 2 verification does not allow skipped tests.", file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Strict CUDA verification for the ELF-WONN Phase 3 implementation."""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _test_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _test_ids(item)
        else:
            yield item.id()


def main() -> int:
    if not torch.cuda.is_available():
        print("Phase 3 verification requires a visible CUDA device.", file=sys.stderr)
        return 2

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(REPO_ROOT / "tests"),
        top_level_dir=str(REPO_ROOT),
    )
    test_ids = set(_test_ids(suite))
    required_prefixes = {
        "tests.test_model_contract.WONNModelContractTest.",
        "tests.test_model_contract.WONNCudaContractTest.",
        "tests.test_sampling_contract.WONNSamplingContractTest.",
        "tests.test_wonn_layers.WONNLayerTest.",
        "tests.test_wonn_model.WONNFormalCudaTest.",
        "tests.test_wonn_train_step.WONNTrainStepTest.",
    }
    missing = sorted(
        prefix
        for prefix in required_prefixes
        if not any(test_id.startswith(prefix) for test_id in test_ids)
    )
    if missing:
        print(
            "Phase 3 verification is missing required test groups: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    if len(test_ids) < 40:
        print(
            f"Phase 3 verification discovered {len(test_ids)} tests; expected at least 40.",
            file=sys.stderr,
        )
        return 2

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print("Phase 3 verification does not allow skipped tests.", file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

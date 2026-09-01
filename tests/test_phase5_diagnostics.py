import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from diagnose_phase5_checkpoint import (
    _all_fieldnames,
    _checkpoint_step,
    _fixed_time_steps,
    _nearest_length_permutation,
    _rollout_steps,
    _shuffled_condition,
)


class Phase5DiagnosticHelpersTest(unittest.TestCase):
    def test_checkpoint_step_accepts_intermediate_custom_checkpoint(self):
        self.assertEqual(_checkpoint_step(Path("custom/checkpoint_5000")), 5000)
        with self.assertRaisesRegex(ValueError, "checkpoint_<step>"):
            _checkpoint_step(Path("custom/checkpoint_final"))

    def test_all_fieldnames_preserves_first_seen_order(self):
        self.assertEqual(
            _all_fieldnames([{"a": 1, "b": 2}, {"b": 3, "c": 4}]),
            ["a", "b", "c"],
        )

    def test_fixed_time_steps_are_sorted_seeded_and_bounded(self):
        first = _fixed_time_steps(
            num_steps=8,
            p_mean=-1.5,
            p_std=0.8,
            seed=42,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        second = _fixed_time_steps(
            num_steps=8,
            p_mean=-1.5,
            p_std=0.8,
            seed=42,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(first.shape, (9,))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(float(first[0]), 0.0)
        self.assertEqual(float(first[-1]), 1.0)
        self.assertTrue(torch.all(first[1:] >= first[:-1]))

    def test_rollout_steps_start_at_requested_time_and_end_at_one(self):
        base = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0])
        result = _rollout_steps(base, 0.25)
        self.assertTrue(torch.equal(result, torch.tensor([0.25, 0.3, 0.6, 1.0])))

    def test_nearest_length_permutation_is_a_derangement(self):
        lengths = torch.tensor([7, 2, 9, 3, 5])
        permutation = _nearest_length_permutation(lengths)
        self.assertEqual(sorted(permutation.tolist()), list(range(len(lengths))))
        self.assertTrue(torch.all(permutation != torch.arange(len(lengths))))

    def test_shuffled_condition_preserves_target_offsets(self):
        cond = torch.arange(4 * 6 * 2, dtype=torch.float32).reshape(4, 6, 2)
        mask = torch.tensor([
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ], dtype=torch.float32)
        shuffled, metadata = _shuffled_condition(cond, mask)
        self.assertEqual(shuffled.shape, cond.shape)
        self.assertTrue(metadata["fixed_target_offsets"])
        self.assertTrue(torch.all(shuffled[mask == 0] == 0))
        self.assertGreater(metadata["copied_slot_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()

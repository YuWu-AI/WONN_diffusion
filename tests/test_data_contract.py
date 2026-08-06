import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.data_utils import get_dataloader
from utils.encoder_utils import build_self_attn_cond_masks


class DataContractTest(unittest.TestCase):
    def test_condition_target_and_padding_masks(self):
        is_cond = np.array([[True, True, False, False, False]])
        is_valid = np.array([[True, True, True, True, False]])

        encoder_mask, attention_mask, cond_mask = build_self_attn_cond_masks(
            is_cond, is_valid, xp=np
        )

        expected_encoder_mask = np.array(
            [[[1, 1, 0, 0, 0],
              [1, 1, 0, 0, 0],
              [1, 1, 1, 1, 0],
              [1, 1, 1, 1, 0],
              [1, 1, 1, 1, 0]]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(encoder_mask, expected_encoder_mask)
        np.testing.assert_array_equal(
            attention_mask, np.array([[1, 1, 1, 1, 0]], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            cond_mask, np.array([[1, 1, 0, 0, 0]], dtype=np.float32)
        )

    def test_dataloader_builds_fixed_length_source_target_batches(self):
        dataset = [
            {
                "condition_input_ids": [11, 12, 13],
                "input_ids": [21, 22],
                "input": "source-a",
                "target": "target-a",
            },
            {
                "condition_input_ids": [14],
                "input_ids": [23, 24, 25, 26],
                "input": "source-b",
                "target": "target-b",
            },
        ]
        dataloader = get_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            max_seq_length=6,
            pad_token_id=0,
            max_input_seq_length=2,
            distributed=False,
        )

        batch = next(iter(dataloader))
        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["encoder_attention_mask"].shape, (2, 6, 6))
        self.assertEqual(batch["attention_mask"].shape, (2, 6))
        self.assertEqual(batch["cond_seq_mask"].shape, (2, 6))
        np.testing.assert_array_equal(batch["cond_seq_mask"][0], [1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(batch["cond_seq_mask"][1], [1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(batch["attention_mask"][0], [1, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(batch["attention_mask"][1], [1, 1, 1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()

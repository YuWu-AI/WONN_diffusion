import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_phase5_mechanism_subset import _materialize, select_short_indices


class MechanismSubsetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = [
            {
                "condition_input_ids": list(range(index % 5 + 1)),
                "input_ids": list(range(index % 4 + 1)),
                "condition_sequence_length": index % 5 + 1,
                "sequence_length": index % 4 + 1,
            }
            for index in range(30)
        ]

    def test_selection_is_seeded_disjoint_and_length_bounded(self):
        first = select_short_indices(
            self.dataset,
            train_size=6,
            heldout_size=3,
            max_source_tokens=3,
            max_target_tokens=3,
            seed=42,
        )
        second = select_short_indices(
            self.dataset,
            train_size=6,
            heldout_size=3,
            max_source_tokens=3,
            max_target_tokens=3,
            seed=42,
        )
        self.assertEqual(first, second)
        train, heldout = first
        self.assertFalse(set(train) & set(heldout))
        for index in train + heldout:
            self.assertLessEqual(self.dataset[index]["condition_sequence_length"], 3)
            self.assertLessEqual(self.dataset[index]["sequence_length"], 3)

    def test_selection_rejects_insufficient_rows(self):
        with self.assertRaisesRegex(ValueError, "eligible"):
            select_short_indices(
                self.dataset,
                train_size=20,
                heldout_size=10,
                max_source_tokens=1,
                max_target_tokens=1,
                seed=42,
            )

    def test_materialize_saves_decoded_text_and_original_indices(self):
        from datasets import Dataset, load_from_disk

        source = Dataset.from_dict({
            "condition_input_ids": [[1, 2], [3], [4, 5]],
            "input_ids": [[6], [7, 8], [9]],
        })

        class Tokenizer:
            @staticmethod
            def batch_decode(rows, skip_special_tokens):
                assert skip_special_tokens
                return [" ".join(str(item) for item in row) for row in rows]

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "subset"
            _materialize(source, [2, 0], Tokenizer(), output)
            restored = load_from_disk(str(output))
            self.assertEqual(restored["index"], [2, 0])
            self.assertEqual(restored["input"], ["4 5", "1 2"])
            self.assertEqual(restored["target"], ["9", "6"])
            self.assertFalse((output.parent / ".subset.flatten.arrow").exists())


if __name__ == "__main__":
    unittest.main()

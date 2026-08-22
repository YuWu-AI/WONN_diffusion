import unittest

import torch

from scripts.run_phase4_learnability import FrozenBatchEncoder, _parameter_snapshot
from tests.contract_factories import make_tiny_wonn


class Phase4HarnessTest(unittest.TestCase):
    def test_frozen_encoder_reuses_only_the_exact_real_batch(self):
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        attention_mask = torch.ones(2, 3, 3)
        latents = torch.randn(2, 3, 4)
        encoder = FrozenBatchEncoder(input_ids, attention_mask, latents)

        output = encoder(input_ids.clone(), attention_mask.clone())
        torch.testing.assert_close(output, latents)

        with self.assertRaisesRegex(ValueError, "different batch"):
            encoder(input_ids.flip(0), attention_mask)
        with self.assertRaisesRegex(ValueError, "different attention mask"):
            encoder(input_ids, torch.zeros_like(attention_mask))

    def test_parameter_snapshot_tracks_every_new_omega_transition_path(self):
        model = make_tiny_wonn(depth=2)
        snapshot = _parameter_snapshot(model)
        self.assertEqual(
            set(snapshot),
            {
                "alpha",
                "theta_embedding",
                "input_projection",
                "output_projection",
            },
        )
        self.assertEqual(snapshot["alpha"].shape, torch.Size([]))
        self.assertEqual(snapshot["theta_embedding"].shape, (16, 1, 2))
        self.assertEqual(snapshot["input_projection"].shape, (16, 32))
        self.assertEqual(snapshot["output_projection"].shape, (16, 16))


if __name__ == "__main__":
    unittest.main()

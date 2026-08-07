import unittest

import torch

from scripts.run_phase4_learnability import FrozenBatchEncoder


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


if __name__ == "__main__":
    unittest.main()

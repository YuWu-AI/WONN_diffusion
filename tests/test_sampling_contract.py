import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs.config import Config, SamplingConfig
from modules.model import ELF
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch
from utils.sampling_utils import add_noise, restore_cond, restore_vx


def make_sampler_fixture():
    torch.manual_seed(17)
    model = ELF(
        text_encoder_dim=16,
        max_length=6,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        bottleneck_dim=8,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=1,
        num_model_mode_tokens=1,
        vocab_size=23,
    ).eval()
    config = Config()
    config.max_length = 6
    config.num_self_cond_cfg_tokens = 1
    config.self_cond_prob = 0.5
    config.denoiser_noise_scale = 1.0
    config.use_bf16 = False
    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[2],
        cfgs=[1],
        self_cond_cfg_scales=[1.0],
        time_schedule="uniform",
    )
    return model, config, sampling_config


class SamplingContractTest(unittest.TestCase):
    def setUp(self):
        self.model, self.config, self.sampling_config = make_sampler_fixture()
        self.z = torch.randn(2, 6, 16)
        self.t_steps = torch.tensor([0.0, 0.5, 1.0])
        self.cond_seq = torch.randn(2, 6, 16)
        self.cond_mask = torch.tensor(
            [[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.float32
        )

    def generate(self, *, cond_seq, cond_mask):
        return _generate_samples_single_batch(
            model=self.model,
            generator=torch.Generator().manual_seed(23),
            z=self.z.clone(),
            t_steps=self.t_steps,
            cond_seq=cond_seq,
            cond_seq_mask=cond_mask,
            config=self.config,
            sampling_config=self.sampling_config,
            cfg_scale=1.0,
            self_cond_cfg_scale=1.0,
        )

    def test_condition_helpers_restore_clean_source_positions(self):
        updated = torch.zeros_like(self.cond_seq)
        restored = restore_cond(updated, self.cond_seq, self.cond_mask)
        mask = self.cond_mask.bool().unsqueeze(-1).expand_as(restored)
        torch.testing.assert_close(restored[mask], self.cond_seq[mask])
        torch.testing.assert_close(restored[~mask], updated[~mask])

        velocity, clean = restore_vx(
            torch.ones_like(updated), updated, self.cond_seq, self.cond_mask
        )
        torch.testing.assert_close(clean[mask], self.cond_seq[mask])
        torch.testing.assert_close(velocity[mask], torch.zeros_like(velocity[mask]))

        noised = add_noise(
            self.cond_seq,
            torch.randn_like(self.cond_seq),
            torch.tensor([0.2, 0.8]),
            self.config,
            self.cond_mask.unsqueeze(-1),
        )
        torch.testing.assert_close(noised[mask], self.cond_seq[mask])

    def test_sampler_preserves_full_shape_and_condition_prefix(self):
        latent = self.generate(cond_seq=self.cond_seq, cond_mask=self.cond_mask)
        self.assertEqual(latent.shape, self.z.shape)
        mask = self.cond_mask.bool().unsqueeze(-1).expand_as(latent)
        torch.testing.assert_close(latent[mask], self.cond_seq[mask])

        predicted_ids = _dlm_decode_batch(
            latent,
            self.model,
            t_final_val=1.0,
            config=self.config,
            self_cond_cfg_scale=1.0,
        )
        self.assertEqual(predicted_ids.shape, (2, 6))
        self.assertEqual(predicted_ids.dtype, torch.int64)
        self.assertTrue(((predicted_ids >= 0) & (predicted_ids < 23)).all())

    def test_unconditional_sampler_is_deterministic_for_fixed_input(self):
        first = self.generate(cond_seq=None, cond_mask=None)
        second = self.generate(cond_seq=None, cond_mask=None)
        self.assertEqual(first.shape, self.z.shape)
        torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

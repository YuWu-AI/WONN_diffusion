import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs.config import Config, load_config_from_yaml
from modules.denoiser_objectives import (
    _auxiliary_scale,
    _ema_decoder_logits,
    _nearest_length_permutation,
    _replace_condition_with_permuted_source,
    compute_denoiser_auxiliary_loss,
    validate_denoiser_objective_config,
)
from tests.contract_factories import make_tiny_wonn
from utils.train_utils import TrainState


class DenoiserObjectiveHelperTest(unittest.TestCase):
    def test_auxiliary_scale_respects_start_and_warmup(self):
        self.assertEqual(_auxiliary_scale(9, 10, 5), 0.0)
        self.assertAlmostEqual(_auxiliary_scale(10, 10, 5), 0.2)
        self.assertEqual(_auxiliary_scale(14, 10, 5), 1.0)
        self.assertEqual(_auxiliary_scale(20, 10, 0), 1.0)

    def test_invalid_config_is_rejected(self):
        config = Config()
        config.denoiser_aux_max_t = 1.1
        with self.assertRaisesRegex(ValueError, "max_t"):
            validate_denoiser_objective_config(config)

    def test_length_permutation_is_a_derangement(self):
        lengths = torch.tensor([4, 1, 3, 2, 5])
        permutation = _nearest_length_permutation(lengths)
        self.assertEqual(sorted(permutation.tolist()), list(range(5)))
        self.assertTrue(torch.all(permutation != torch.arange(5)))

    def test_source_replacement_keeps_target_slots_fixed(self):
        latent = torch.arange(4 * 6 * 2, dtype=torch.float32).reshape(4, 6, 2)
        mask = torch.tensor([
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ], dtype=torch.float32)
        shuffled = _replace_condition_with_permuted_source(latent, mask)
        expanded_target = (mask == 0).unsqueeze(-1).expand_as(latent)
        self.assertTrue(torch.equal(shuffled[expanded_target], latent[expanded_target]))
        self.assertFalse(torch.equal(shuffled[mask == 1], latent[mask == 1]))

    def test_ema_decoder_backpropagates_only_to_latent(self):
        model = make_tiny_wonn(self_cond_tokens=1).train()
        ema = TrainState.init_ema(model)
        latent = torch.randn(3, 6, 16, requires_grad=True)
        logits = _ema_decoder_logits(
            model=model,
            ema_params=ema,
            latent=latent,
            self_cond_cfg_scale=torch.ones(3),
        )
        logits.square().mean().backward()
        self.assertIsNotNone(latent.grad)
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_auxiliary_loss_returns_finite_auditable_counts(self):
        model = make_tiny_wonn(self_cond_tokens=1).train()
        config = Config()
        config.denoiser_token_loss_weight = 0.02
        config.denoiser_source_contrastive_weight = 0.1
        config.denoiser_source_contrastive_prob = 1.0
        config.denoiser_aux_max_t = 0.25
        x_pred = torch.randn(4, 6, 16, requires_grad=True)
        target_mask = torch.tensor([
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ], dtype=torch.float32)
        loss, metrics = compute_denoiser_auxiliary_loss(
            model=model,
            ema_params=TrainState.init_ema(model),
            x_pred=x_pred,
            targets=torch.randint(0, 23, (4, 6)),
            target_mask=target_mask,
            cond_mask=1.0 - target_mask,
            timesteps=torch.tensor([0.05, 0.15, 0.20, 0.75]),
            denoiser_rows=torch.ones(4),
            label_drop_mask=torch.zeros(4, dtype=torch.bool),
            self_cond_cfg_scale=torch.ones(4),
            config=config,
            step=0,
            generator=torch.Generator().manual_seed(42),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(metrics["aux_example_count"]), 3)
        self.assertGreater(int(metrics["token_count"]), 0)
        self.assertEqual(int(metrics["contrastive_count"]), 3)
        loss.backward()
        self.assertGreater(float(x_pred.grad.abs().sum()), 0.0)


class MechanismConfigTest(unittest.TestCase):
    def test_three_configs_only_differ_in_objective_and_output(self):
        root = REPO_ROOT / "src/configs/training_configs/phase5_mechanism"
        paths = (
            root / "train_de-en-WONN-S-base-50k.yml",
            root / "train_de-en-WONN-S-token-50k.yml",
            root / "train_de-en-WONN-S-token-contrast-50k.yml",
        )
        configs = [load_config_from_yaml(str(path)) for path in paths]
        allowed = {
            "denoiser_token_loss_weight",
            "denoiser_source_contrastive_weight",
            "output_dir",
        }
        fields = {
            name for name in Config.__annotations__
            if name not in allowed and name != "sampling_configs"
        }
        for field in fields:
            values = [getattr(config, field) for config in configs]
            self.assertEqual(values[1:], values[:-1], field)
        for config in configs:
            self.assertEqual(config.max_optimizer_steps, 50000)
            self.assertEqual(config.max_length, 64)
            self.assertEqual(config.max_input_length, 32)
            self.assertEqual(config.wonn_num_layers, 4)
            self.assertEqual(config.wonn_num_inner_steps, 3)
            self.assertEqual(config.wonn_num_oscillators, 192)
            self.assertTrue(config.data_manifest_path)


if __name__ == "__main__":
    unittest.main()

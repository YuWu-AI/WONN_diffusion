import math
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from modules.wonn_layers import (
    AttentiveWinfreeCoupling,
    GroupedPhaseMap,
    WONNLayer,
    phase_features,
    wrap_phase,
)


class WONNLayerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.theta = torch.randn(2, 5, 8)
        self.omega = torch.randn(2, 5, 8)

    def test_phase_wrapping_is_periodic_and_bounded(self):
        wrapped = wrap_phase(self.theta)
        shifted = wrap_phase(self.theta + 4 * math.pi)
        torch.testing.assert_close(wrapped, shifted, atol=2e-6, rtol=0)
        self.assertTrue((wrapped <= math.pi).all())
        self.assertTrue((wrapped >= -math.pi).all())
        torch.testing.assert_close(
            phase_features(self.theta),
            phase_features(self.theta + 2 * math.pi),
            atol=2e-6,
            rtol=0,
        )

    def test_grouped_phase_map_is_bounded(self):
        mapping = GroupedPhaseMap(num_heads=2, oscillators_per_head=4)
        features = torch.randn(2, 5, 2, 8)
        output = mapping(features)
        self.assertEqual(output.shape, (2, 5, 2, 4))
        self.assertTrue((output.abs() <= 1).all())

    def test_masked_keys_do_not_change_valid_phase_velocity(self):
        coupling = AttentiveWinfreeCoupling(
            num_oscillators=8,
            num_heads=2,
            qk_head_dim=4,
        ).eval()
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
        changed_theta = self.theta.clone()
        changed_theta[:, 3:] += 100.0
        velocity, _ = coupling(
            self.theta, self.omega, attention_mask=mask
        )
        changed_velocity, _ = coupling(
            changed_theta, self.omega, attention_mask=mask
        )
        torch.testing.assert_close(
            velocity[:, :3], changed_velocity[:, :3], atol=2e-6, rtol=1e-5
        )

    def test_fixed_trig_coupling_has_expected_contract(self):
        coupling = AttentiveWinfreeCoupling(
            num_oscillators=8,
            num_heads=2,
            qk_head_dim=4,
            coupling_mode="fixed_trig",
        )
        velocity, diagnostics = coupling(
            self.theta,
            self.omega,
            collect_diagnostics=True,
        )
        self.assertEqual(velocity.shape, self.theta.shape)
        self.assertTrue(torch.isfinite(velocity).all())
        self.assertTrue(torch.isfinite(diagnostics["attention_entropy"]))

    def test_layer_updates_phase_then_frequency_once(self):
        layer = WONNLayer(
            num_oscillators=8,
            num_heads=2,
            qk_head_dim=4,
            num_inner_steps=2,
        )
        theta, omega, diagnostics = layer(
            self.theta,
            self.omega,
            collect_diagnostics=True,
        )
        self.assertFalse(torch.allclose(theta, self.theta))
        torch.testing.assert_close(omega, self.omega)
        self.assertAlmostEqual(layer.step_size.item(), 0.1, places=6)
        self.assertLess(layer.step_size.item(), layer.step_max)
        self.assertGreater(diagnostics["phase_update_rms"].item(), 0)
        self.assertEqual(diagnostics["frequency_update_rms"].item(), 0)

    def test_layer_backward_produces_finite_gradients(self):
        layer = WONNLayer(
            num_oscillators=8,
            num_heads=2,
            qk_head_dim=4,
            num_inner_steps=1,
        )
        theta = self.theta.clone().requires_grad_(True)
        omega = self.omega.clone().requires_grad_(True)
        next_theta, next_omega, _ = layer(theta, omega)
        loss = next_theta.square().mean() + next_omega.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(theta.grad).all())
        self.assertTrue(torch.isfinite(omega.grad).all())
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))


if __name__ == "__main__":
    unittest.main()

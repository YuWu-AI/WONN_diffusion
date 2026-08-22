import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from modules.wonn_layers import (
    AttentiveWinfreeCoupling,
    OmegaTransition,
    PerOscillatorMLP,
    ThetaEmbedding,
    WONNLayer,
    phase_features,
    wrap_phase,
)


class WONNCouplingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.theta = torch.randn(2, 5, 8)
        self.omega = torch.randn(2, 5, 8)

    def test_phase_wrapping_and_per_oscillator_maps_are_periodic(self):
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

        sensitivity = PerOscillatorMLP(8, hidden_features=2)
        influence = PerOscillatorMLP(8, hidden_features=4)
        self.assertEqual(sensitivity(self.theta).shape, self.theta.shape)
        self.assertEqual(influence(self.theta).shape, self.theta.shape)
        torch.testing.assert_close(
            sensitivity(self.theta),
            sensitivity(self.theta + 2 * math.pi),
            atol=2e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            influence(self.theta),
            influence(self.theta + 2 * math.pi),
            atol=2e-6,
            rtol=0,
        )

    def test_per_oscillator_mlp_has_independent_parameters_and_channels(self):
        mapping = PerOscillatorMLP(8, hidden_features=4)
        theta = self.theta.clone().requires_grad_(True)
        mapping(theta)[..., 0].sum().backward()
        self.assertEqual(mapping.input_weight.shape, (8, 4, 2))
        self.assertEqual(mapping.output_weight.shape, (8, 1, 4))
        self.assertGreater(theta.grad[..., 0].abs().sum().item(), 0)
        torch.testing.assert_close(
            theta.grad[..., 1:], torch.zeros_like(theta.grad[..., 1:])
        )

    def test_head_dimension_is_derived_and_invalid_division_fails_fast(self):
        coupling = AttentiveWinfreeCoupling(12, 3)
        self.assertEqual(coupling.head_dim, 4)
        self.assertEqual(coupling.W_qkv.in_features, 12)
        self.assertEqual(coupling.W_qkv.out_features, 36)
        with self.assertRaisesRegex(ValueError, "divisible"):
            AttentiveWinfreeCoupling(10, 3)

    def test_complete_attention_matches_direct_reference(self):
        coupling = AttentiveWinfreeCoupling(8, 2).eval()
        velocity, diagnostics = coupling(
            self.theta, self.omega, collect_diagnostics=True
        )

        influence = coupling.influence(self.theta)
        q, k, v = coupling.W_qkv(influence).chunk(3, dim=-1)
        q = q.reshape(2, 5, 2, 4).transpose(1, 2)
        k = k.reshape(2, 5, 2, 4).transpose(1, 2)
        v = v.reshape(2, 5, 2, 4).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(4), dim=-1)
        message = (weights @ v).transpose(1, 2).reshape(2, 5, 8)
        field = F.relu(coupling.output_norm(coupling.W_o(message)))
        expected = self.omega + coupling.sensitivity(self.theta) * field

        torch.testing.assert_close(velocity, expected, atol=2e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(diagnostics["attention_entropy"]))
        self.assertTrue(torch.isfinite(diagnostics["coupling_field_rms"]))

    def test_sdpa_and_explicit_diagnostics_paths_are_equivalent(self):
        coupling = AttentiveWinfreeCoupling(8, 2, attn_drop=0.0).eval()
        sdpa_velocity, _ = coupling(self.theta, self.omega)
        explicit_velocity, _ = coupling(
            self.theta, self.omega, collect_diagnostics=True
        )
        torch.testing.assert_close(
            sdpa_velocity, explicit_velocity, atol=2e-6, rtol=1e-5
        )

    def test_masked_key_changes_do_not_affect_valid_queries_or_layer_updates(self):
        coupling = AttentiveWinfreeCoupling(8, 2).eval()
        layer = WONNLayer(8, 2, num_inner_steps=2).eval()
        layer.coupling.load_state_dict(coupling.state_dict())
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
        changed_theta = self.theta.clone()
        changed_theta[:, 3:] += 100.0

        velocity, _ = coupling(self.theta, self.omega, attention_mask=mask)
        changed_velocity, _ = coupling(
            changed_theta, self.omega, attention_mask=mask
        )
        torch.testing.assert_close(
            velocity[:, :3], changed_velocity[:, :3], atol=2e-6, rtol=1e-5
        )

        next_theta, _ = layer(self.theta, self.omega, attention_mask=mask)
        changed_next_theta, _ = layer(
            changed_theta, self.omega, attention_mask=mask
        )
        torch.testing.assert_close(
            next_theta[:, :3], changed_next_theta[:, :3], atol=2e-6, rtol=1e-5
        )

    def test_all_masked_sample_fails_fast(self):
        coupling = AttentiveWinfreeCoupling(8, 2)
        mask = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "no valid keys.*1"):
            coupling(self.theta, self.omega, attention_mask=mask)

    def test_output_projection_can_mix_values_across_heads(self):
        coupling = AttentiveWinfreeCoupling(4, 2).eval()
        theta = torch.zeros(1, 1, 4)
        omega = torch.zeros_like(theta)
        influence = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
        sensitivity = torch.ones_like(theta)
        with torch.no_grad():
            coupling.W_qkv.weight.zero_()
            coupling.W_qkv.bias.zero_()
            coupling.W_qkv.weight[8:12].copy_(torch.eye(4))
            coupling.W_o.weight.zero_()
            coupling.W_o.bias.zero_()
            coupling.W_o.weight[0, 2] = 1.0
        with mock.patch.object(
            coupling.influence, "forward", return_value=influence
        ), mock.patch.object(
            coupling.sensitivity, "forward", return_value=sensitivity
        ):
            velocity, _ = coupling(theta, omega)
        self.assertGreater(velocity[0, 0, 0].item(), 0)
        torch.testing.assert_close(velocity[0, 0, 1:], torch.zeros(3))

    def test_zero_output_projection_proves_there_is_no_hidden_field_residual(self):
        coupling = AttentiveWinfreeCoupling(8, 2).eval()
        with torch.no_grad():
            coupling.W_o.weight.zero_()
            coupling.W_o.bias.zero_()
        velocity, diagnostics = coupling(
            self.theta, self.omega, collect_diagnostics=True
        )
        torch.testing.assert_close(velocity, self.omega, atol=0, rtol=0)
        self.assertEqual(diagnostics["coupling_field_rms"].item(), 0.0)

    def test_recurrent_steps_share_parameters_and_recompute_updated_phase(self):
        layer = WONNLayer(8, 2, num_inner_steps=3)
        seen_phases = []
        handle = layer.coupling.register_forward_pre_hook(
            lambda _module, args: seen_phases.append(args[0].detach().clone())
        )
        parameter_id = id(layer.coupling.W_qkv.weight)
        try:
            layer(self.theta, self.omega)
        finally:
            handle.remove()
        self.assertEqual(len(seen_phases), 3)
        self.assertFalse(torch.allclose(seen_phases[0], seen_phases[1]))
        self.assertFalse(torch.allclose(seen_phases[1], seen_phases[2]))
        self.assertEqual(id(layer.coupling.W_qkv.weight), parameter_id)

    def test_forward_backward_reaches_all_coupling_paths(self):
        layer = WONNLayer(8, 2, num_inner_steps=2)
        with torch.no_grad():
            layer.coupling.W_o.bias.fill_(1.0)
        theta = self.theta.clone().requires_grad_(True)
        omega = self.omega.clone().requires_grad_(True)
        next_theta, _ = layer(theta, omega)
        next_theta.square().mean().backward()
        self.assertTrue(torch.isfinite(theta.grad).all())
        self.assertTrue(torch.isfinite(omega.grad).all())
        required = (
            "coupling.sensitivity",
            "coupling.influence",
            "coupling.W_qkv",
            "coupling.W_o",
        )
        for prefix in required:
            gradients = [
                parameter.grad
                for name, parameter in layer.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(all(gradient is not None for gradient in gradients), prefix)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients), prefix
            )
            self.assertGreater(sum(gradient.abs().sum() for gradient in gradients), 0)


class OmegaTransitionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.theta = torch.randn(2, 5, 8)
        self.omega = torch.randn(2, 5, 8)

    def test_theta_embedding_is_periodic_and_boundary_preserves_theta(self):
        embedding = ThetaEmbedding(8)
        torch.testing.assert_close(
            embedding(self.theta),
            embedding(self.theta + 2 * math.pi),
            atol=2e-6,
            rtol=0,
        )
        transition = OmegaTransition(8)
        next_theta, next_omega, _ = transition(self.theta, self.omega)
        self.assertIs(next_theta, self.theta)
        torch.testing.assert_close(next_theta, self.theta, atol=0, rtol=0)
        self.assertFalse(torch.allclose(next_omega, self.omega))

    def test_theta_and_omega_both_affect_frequency_delta(self):
        transition = OmegaTransition(8)

        def delta(theta, omega):
            _, next_omega, _ = transition(theta, omega)
            return (next_omega - omega) / transition.alpha

        baseline = delta(self.theta, self.omega)
        changed_theta = delta(self.theta + 0.5, self.omega)
        changed_omega = delta(self.theta, self.omega + 0.5)
        self.assertFalse(torch.allclose(baseline, changed_theta))
        self.assertFalse(torch.allclose(baseline, changed_omega))

    def test_alpha_initialization_and_bounds(self):
        transition = OmegaTransition(8)
        self.assertAlmostEqual(transition.alpha.item(), 0.1, places=6)
        with torch.no_grad():
            transition.raw_alpha.fill_(10.0)
        self.assertGreater(transition.alpha.item(), 0.0)
        self.assertLess(transition.alpha.item(), transition.alpha_max)
        with torch.no_grad():
            transition.raw_alpha.fill_(-10.0)
        self.assertGreater(transition.alpha.item(), 0.0)
        self.assertLess(transition.alpha.item(), transition.alpha_max)

    def test_first_backward_reaches_theta_ffn_and_alpha(self):
        transition = OmegaTransition(8)
        with torch.no_grad():
            transition.input_projection.bias.fill_(1.0)
        theta = self.theta.clone().requires_grad_(True)
        omega = self.omega.clone().requires_grad_(True)
        _, next_omega, _ = transition(theta, omega)
        next_omega.square().mean().backward()
        required = (
            "theta_embedding",
            "input_projection",
            "output_projection",
            "raw_alpha",
        )
        for prefix in required:
            gradients = [
                parameter.grad
                for name, parameter in transition.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(all(gradient is not None for gradient in gradients), prefix)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients), prefix
            )
            self.assertGreater(sum(gradient.abs().sum() for gradient in gradients), 0)


if __name__ == "__main__":
    unittest.main()

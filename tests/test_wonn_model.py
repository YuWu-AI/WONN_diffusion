import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests.contract_factories import make_tiny_wonn
from configs.config import load_config_from_yaml
from modules.model_factory import build_model


class WONNModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.x = torch.randn(2, 6, 16)
        self.t = torch.tensor([0.2, 0.8])

    def test_diagnostics_are_finite_and_phase_is_active(self):
        model = make_tiny_wonn().eval()
        output, logits, diagnostics = model.forward_with_diagnostics(
            self.x,
            self.t,
            decoder_step_active=True,
        )
        self.assertEqual(output.shape, (2, 6, 16))
        self.assertEqual(logits.shape, (2, 6, 23))
        self.assertGreater(diagnostics["layer_0/phase_update_rms"].item(), 0)
        self.assertEqual(
            diagnostics["layer_0/frequency_update_rms"].item(), 0
        )
        self.assertGreaterEqual(diagnostics["kuramoto_order"].item(), 0)
        self.assertLessEqual(diagnostics["kuramoto_order"].item(), 1)
        self.assertTrue(all(torch.isfinite(value) for value in diagnostics.values()))

    def test_forward_is_stateless_across_calls(self):
        model = make_tiny_wonn().eval()
        first_output, first_logits, first_diagnostics = model.forward_with_diagnostics(
            self.x, self.t, decoder_step_active=True
        )
        second_output, second_logits, second_diagnostics = model.forward_with_diagnostics(
            self.x, self.t, decoder_step_active=True
        )
        torch.testing.assert_close(first_output, second_output, rtol=0, atol=0)
        torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)
        self.assertEqual(first_diagnostics.keys(), second_diagnostics.keys())
        for key in first_diagnostics:
            torch.testing.assert_close(
                first_diagnostics[key], second_diagnostics[key], rtol=0, atol=0
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the formal WONN contract")
class WONNFormalCudaTest(unittest.TestCase):
    def test_formal_wonn_b_forward_backward_and_diagnostics(self):
        device = torch.device("cuda")
        config = load_config_from_yaml(
            str(
                REPO_ROOT
                / "src/configs/training_configs/train_de-en_ELF-WONN-B.yml"
            )
        )
        self.assertTrue(config.gradient_checkpointing)
        self.assertFalse(config.compile_train)
        model = build_model(
            config,
            text_encoder_dim=512,
            max_length=128,
            vocab_size=32100,
        ).to(device).train()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 30_472_176)

        x = torch.randn(1, 128, 1024, device=device, requires_grad=True)
        t = torch.tensor([0.5], device=device)
        mask = torch.ones(1, 128, device=device)
        scale = torch.ones(1, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output, logits = model(
                x,
                t,
                attention_mask=mask,
                self_cond_cfg_scale=scale,
                decoder_step_active=True,
            )
            loss = output.square().mean() + logits.square().mean()
        loss.backward()
        self.assertEqual(output.shape, (1, 128, 512))
        self.assertEqual(logits.shape, (1, 128, 32100))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, _, diagnostics = model.forward_with_diagnostics(
                x[..., :512].detach(),
                t,
                attention_mask=mask,
                self_cond_cfg_scale=scale,
                decoder_step_active=True,
            )
        self.assertGreater(diagnostics["layer_0/phase_update_rms"].item(), 0)
        self.assertTrue(all(torch.isfinite(value) for value in diagnostics.values()))


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from modules.model import ELF, ELF_models


def make_tiny_model(
    *, self_cond_tokens=0, gradient_checkpointing=False, dropout=0.0, depth=2
):
    torch.manual_seed(7)
    return ELF(
        text_encoder_dim=16,
        max_length=6,
        hidden_size=32,
        depth=depth,
        num_heads=4,
        mlp_ratio=2.0,
        attn_drop=dropout,
        proj_drop=dropout,
        bottleneck_dim=8,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=self_cond_tokens,
        num_model_mode_tokens=1,
        vocab_size=23,
        gradient_checkpointing=gradient_checkpointing,
    )


class ELFModelContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.x = torch.randn(2, 6, 16)
        self.t = torch.tensor([0.25, 0.75])
        self.mask = torch.ones(2, 6)

    def test_denoise_and_decode_shapes_dtype_and_device(self):
        model = make_tiny_model().eval()

        denoised, logits = model(self.x, self.t, attention_mask=self.mask)
        self.assertEqual(denoised.shape, (2, 6, 16))
        self.assertIsNone(logits)

        decoded, logits = model(
            self.x,
            self.t,
            attention_mask=self.mask,
            decoder_step_active=True,
        )
        self.assertEqual(decoded.shape, (2, 6, 16))
        self.assertEqual(logits.shape, (2, 6, 23))
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(decoded.device.type, "cpu")
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertTrue(torch.isfinite(logits).all())

    def test_self_conditioning_off_and_on_preserve_output_contract(self):
        model = make_tiny_model(self_cond_tokens=1).eval()
        scale = torch.ones(2)

        plain_out, plain_logits = model(
            self.x,
            self.t,
            self_cond_cfg_scale=scale,
            decoder_step_active=True,
        )
        self_cond_out, self_cond_logits = model(
            torch.cat([self.x, torch.zeros_like(self.x)], dim=-1),
            self.t,
            self_cond_cfg_scale=scale,
            decoder_step_active=True,
        )

        self.assertEqual(plain_out.shape, self_cond_out.shape)
        self.assertEqual(plain_logits.shape, self_cond_logits.shape)
        self.assertEqual(plain_out.shape, (2, 6, 16))
        self.assertEqual(plain_logits.shape, (2, 6, 23))

    def test_scalar_and_per_example_modes_have_consistent_semantics(self):
        model = make_tiny_model().eval()
        _, logits_false = model(self.x, self.t, decoder_step_active=False)
        _, logits_true = model(self.x, self.t, decoder_step_active=True)
        _, logits_mixed = model(
            self.x,
            self.t,
            decoder_step_active=torch.tensor([0.0, 1.0]),
        )

        torch.testing.assert_close(logits_mixed[0], logits_false[0])
        torch.testing.assert_close(logits_mixed[1], logits_true[1])
        self.assertFalse(torch.allclose(logits_false, logits_true))

    def test_padding_keys_do_not_change_valid_token_outputs(self):
        model = make_tiny_model().eval()
        mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]])
        changed = self.x.clone()
        changed[:, 3:] = changed[:, 3:] + 1000.0

        _, logits = model(self.x, self.t, attention_mask=mask, decoder_step_active=True)
        _, changed_logits = model(
            changed, self.t, attention_mask=mask, decoder_step_active=True
        )

        torch.testing.assert_close(logits[:, :3], changed_logits[:, :3])

    def test_eval_forward_is_deterministic(self):
        model = make_tiny_model(dropout=0.5, depth=4).eval()
        _, first = model(self.x, self.t, decoder_step_active=True, deterministic=True)
        _, second = model(self.x, self.t, decoder_step_active=True, deterministic=True)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

        _, stochastic_first = model(
            self.x, self.t, decoder_step_active=True, deterministic=False
        )
        _, stochastic_second = model(
            self.x, self.t, decoder_step_active=True, deterministic=False
        )
        self.assertFalse(torch.allclose(stochastic_first, stochastic_second))

    def test_rope_requires_fixed_max_length(self):
        model = make_tiny_model().eval()
        with self.assertRaises((RuntimeError, ValueError)):
            model(self.x[:, :-1], self.t, decoder_step_active=True)
        with self.assertRaises((RuntimeError, ValueError)):
            model(
                torch.cat([self.x, self.x[:, :1]], dim=1),
                self.t,
                decoder_step_active=True,
            )

    def test_self_cond_cfg_tokens_require_scale_token(self):
        model = make_tiny_model(self_cond_tokens=1).eval()
        with self.assertRaises((RuntimeError, ValueError)):
            model(self.x, self.t, decoder_step_active=True)

    def test_backward_with_gradient_checkpointing(self):
        model = make_tiny_model(gradient_checkpointing=True).train()
        x = self.x.clone().requires_grad_(True)
        output, logits = model(x, self.t, decoder_step_active=True)
        loss = output.mean() + logits.square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        finite_grads = [
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(finite_grads)
        self.assertTrue(all(finite_grads))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mixed-precision contracts")
class ELFCudaContractTest(unittest.TestCase):
    def test_tiny_model_bf16_autocast_and_backward(self):
        device = torch.device("cuda")
        model = make_tiny_model(gradient_checkpointing=True).to(device).train()
        x = torch.randn(2, 6, 16, device=device, requires_grad=True)
        t = torch.tensor([0.2, 0.8], device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output, logits = model(x, t, decoder_step_active=True)
            loss = output.mean() + logits.square().mean()
        loss.backward()

        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_official_elf_b_factory_contract(self):
        device = torch.device("cuda")
        model = ELF_models["ELF-B"](
            text_encoder_dim=512,
            max_length=128,
            vocab_size=32100,
            bottleneck_dim=128,
            num_time_tokens=4,
            num_self_cond_cfg_tokens=4,
            num_model_mode_tokens=4,
        ).to(device).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 104_579_940)

        x = torch.randn(1, 128, 1024, device=device)
        t = torch.tensor([0.5], device=device)
        mask = torch.ones(1, 128, device=device)
        scale = torch.ones(1, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output, logits = model(
                x,
                t,
                attention_mask=mask,
                self_cond_cfg_scale=scale,
                decoder_step_active=True,
            )

        self.assertEqual(output.shape, (1, 128, 512))
        self.assertEqual(logits.shape, (1, 128, 32100))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()

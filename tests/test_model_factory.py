import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs.config import Config
from modules.model import ELF
from modules.model_factory import build_model
from modules.wonn_model import WONNELF


class ModelFactoryTest(unittest.TestCase):
    def test_elf_factory_still_builds_upstream_model(self):
        config = Config()
        config.model = "ELF-B"
        model = build_model(
            config, text_encoder_dim=16, max_length=6, vocab_size=23
        )
        self.assertIsInstance(model, ELF)

    def test_wonn_factory_uses_backbone_specific_config(self):
        config = Config()
        config.model = "ELF-WONN-B"
        config.wonn_num_oscillators = 16
        config.wonn_num_layers = 2
        config.wonn_num_inner_steps = 1
        config.wonn_num_heads = 4
        config.wonn_qk_head_dim = 8
        model = build_model(
            config, text_encoder_dim=16, max_length=6, vocab_size=23
        )
        self.assertIsInstance(model, WONNELF)
        self.assertEqual(model.num_oscillators, 16)
        self.assertEqual(len(model.layers), 2)

    def test_unknown_model_is_rejected(self):
        config = Config()
        config.model = "missing"
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model(
                config, text_encoder_dim=16, max_length=6, vocab_size=23
            )


if __name__ == "__main__":
    unittest.main()

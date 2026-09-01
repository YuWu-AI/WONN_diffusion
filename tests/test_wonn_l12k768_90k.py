import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_phase5_50k import _metric_bundle
from analyze_wonn_l12k768_90k import analyze_curve
from configs.config import load_config_from_yaml


def _write_evaluation(run_dir: Path, step: int, hypotheses: list[str]) -> None:
    references = [f"translated sentence {index}" for index in range(len(hypotheses))]
    sources = [f"quelle {index}" for index in range(len(hypotheses))]
    eval_dir = run_dir / "evaluations" / f"checkpoint_{step}"
    sample_dir = eval_dir / "ode-test-cond"
    sample_dir.mkdir(parents=True)
    rows = [
        {
            "id": index,
            "source": sources[index],
            "reference": references[index],
            "generated": hypotheses[index],
        }
        for index in range(len(hypotheses))
    ]
    (sample_dir / f"all_generated_0_{step}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    metrics = _metric_bundle(hypotheses, references)
    (sample_dir / "metrics.jsonl").write_text(json.dumps({
        "step": step,
        "num_samples": len(rows),
        "generation_seconds": 1.0,
        "decode_seconds": 0.1,
        "samples_per_second": 10.0,
        "timing_scope": "cuda_synchronized_sampler_and_decoder",
        "bleu": metrics["bleu"],
        "rouge1": 1.0,
        "rouge2": 1.0,
        "rougeL": 1.0,
    }) + "\n", encoding="utf-8")
    (eval_dir / "evaluation_complete.json").write_text(json.dumps({
        "status": "complete",
        "model": "test",
        "step": step,
        "num_samples": len(rows),
    }), encoding="utf-8")


class L12K768ConfigTest(unittest.TestCase):
    def test_capacity_config_preserves_training_contract(self):
        root = REPO_ROOT / "src/configs/training_configs"
        old = load_config_from_yaml(str(root / "train_de-en-WONN-L6T3-phase5-130k.yml"))
        new = load_config_from_yaml(str(root / "train_de-en-WONN-L12K768T3-phase5-90k.yml"))
        for field in (
            "data_path", "eval_data_path", "data_revision", "eval_data_revision",
            "max_length", "max_input_length", "encoder_model_name",
            "encoder_revision", "tokenizer_revision", "batch_size", "lr",
            "lr_schedule", "warmup_steps", "optimizer", "seed", "num_workers",
            "wonn_num_inner_steps", "wonn_num_heads", "wonn_step_init",
            "wonn_step_max",
        ):
            self.assertEqual(getattr(old, field), getattr(new, field), field)
        self.assertEqual(new.wonn_num_layers, 12)
        self.assertEqual(new.wonn_num_oscillators, 768)
        self.assertEqual(new.max_optimizer_steps, 90000)
        self.assertEqual(
            new.save_optimizer_steps,
            "10000,20000,30000,40000,50000,60000,70000,80000,90000",
        )
        self.assertIsNone(new.resume)
        self.assertIsNone(new.init_from)

    def test_pipeline_evaluates_every_ten_thousand_steps(self):
        pipeline = (REPO_ROOT / "scripts/run_wonn_l12k768_90k_pipeline.sh").read_text(
            encoding="utf-8",
        )
        self.assertIn("for step in $(seq 10000 10000 90000)", pipeline)
        self.assertIn("analyze_wonn_l12k768_90k.py", pipeline)
        self.assertIn("num_samples=1000", pipeline)
        self.assertNotIn("phase5-wmt14", pipeline)
        self.assertNotIn(".worktrees", pipeline)
        self.assertIn('$repo_root/.venv/bin/python', pipeline)
        self.assertIn('$repo_root/outputs/phase5/elf_b_seed42_b12_0_90k', pipeline)


class L12K768AnalysisTest(unittest.TestCase):
    def test_report_has_paired_table_and_two_dual_model_charts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            elf = root / "elf"
            wonn = root / "wonn"
            steps = (10000, 20000)
            for step in steps:
                references = [f"translated sentence {index}" for index in range(4)]
                _write_evaluation(elf, step, references)
                _write_evaluation(wonn, step, references[:-1] + ["different"])
                (wonn / f"checkpoint_{step}").write_bytes(b"checkpoint")
            (wonn / "training_complete.json").write_text(json.dumps({
                "status": "complete",
                "completed_optimizer_step": 20000,
                "model_parameters": 76024187,
            }), encoding="utf-8")
            (wonn / "config.yml").write_text(
                "model: ELF-WONN-B\n"
                "wonn_num_layers: 12\n"
                "wonn_num_oscillators: 768\n"
                "wonn_num_inner_steps: 3\n"
                "wonn_num_heads: 12\n"
                "batch_size: 12\n"
                "seed: 42\n",
                encoding="utf-8",
            )
            output = root / "analysis"
            payload = analyze_curve(
                wonn, elf, output, steps=steps, comparison_samples=4,
            )
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(len(payload["metrics_by_step"]), 2)
            self.assertTrue((output / "metrics_by_step.csv").is_file())
            self.assertTrue((output / "comparison.md").is_file())
            for chart in ("bleu_by_step.svg", "chrf_by_step.svg"):
                content = (output / chart).read_text(encoding="utf-8")
                self.assertIn("Transformer ELF-B", content)
                self.assertIn("WONN-L12K768T3", content)


if __name__ == "__main__":
    unittest.main()

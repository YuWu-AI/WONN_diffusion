import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_cloud_matrix import (
    CHECKPOINT_STEPS,
    MODEL_SPECS,
    _metric_bundle,
    _training_rows,
    analyze_matrix,
    validate_matrix_configs,
)


def _write_evaluation(run_dir: Path, label: str, step: int, hypotheses: list[str]) -> None:
    references = [f"translation {index}" for index in range(len(hypotheses))]
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
        "peak_allocated_cuda_mib": 100.0,
        "bleu": metrics["bleu"],
        "rouge1": 1.0,
        "rouge2": 1.0,
        "rougeL": 1.0,
    }) + "\n", encoding="utf-8")
    (eval_dir / "evaluation_complete.json").write_text(json.dumps({
        "status": "complete",
        "model": label,
        "step": step,
        "num_samples": len(rows),
    }), encoding="utf-8")


class CloudMatrixConfigTest(unittest.TestCase):
    def test_configs_define_the_fixed_100k_matrix(self):
        payload = validate_matrix_configs(
            REPO_ROOT / "src/configs/training_configs"
        )
        self.assertEqual(payload["terminal_optimizer_step"], 100000)
        self.assertEqual(payload["checkpoint_steps"], list(CHECKPOINT_STEPS))
        self.assertEqual(payload["effective_batch_size"], 512)
        self.assertEqual(payload["batch_size_per_device"], 16)
        self.assertEqual(payload["grad_accum_steps"], 32)
        self.assertEqual(payload["warmup_optimizer_steps"], 5000)
        self.assertEqual(
            payload["learning_rates"],
            {"E0": 0.002, "W0": 0.001, "W1": 0.001, "W2": 0.001},
        )

    def test_pipeline_assigns_one_model_to_each_gpu(self):
        pipeline = (REPO_ROOT / "scripts/run_cloud_matrix_pipeline.sh").read_text(
            encoding="utf-8",
        )
        self.assertIn("model_keys=(E0 W0 W1 W2)", pipeline)
        self.assertIn("model_gpus=(0 1 2 3)", pipeline)
        self.assertIn("target_steps=100000", pipeline)
        self.assertIn("analyze_cloud_matrix.py validate-configs", pipeline)
        self.assertNotIn("torch.distributed.run", pipeline)
        self.assertIn("evaluation_count\": 28", pipeline)


class CloudMatrixAnalysisTest(unittest.TestCase):
    def test_training_metrics_must_be_strictly_monotonic(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "train_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in (
                    {"optimizer_step": 5, "samples_seen": 512},
                    {"optimizer_step": 5, "samples_seen": 1024},
                )),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not strictly monotonic"):
                _training_rows(run_dir)

    def test_analysis_accepts_four_models_and_28_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key, spec in MODEL_SPECS.items():
                run_dir = root / spec["slug"]
                run_dir.mkdir()
                config = {
                    "model": spec["model"],
                    "global_batch_size": 512,
                    "batch_size": 16,
                    "grad_accum_steps": 32,
                    "epochs": 100,
                    "max_optimizer_steps": 100000,
                    "stop_optimizer_steps": None,
                    "save_optimizer_steps": ",".join(str(step) for step in CHECKPOINT_STEPS),
                    "lr": spec["lr"],
                    "lr_schedule": "constant",
                    "warmup_steps": 5000,
                    "optimizer": "muon",
                    "compile_train": True,
                    "gradient_checkpointing": False,
                    "online_eval": True,
                    "save_freq": 0,
                    "eval_freq": 0,
                    "final_eval": False,
                    "seed": 42,
                    "num_samples": 1000,
                    "resume": None,
                }
                if spec["layers"] is not None:
                    config.update({
                        "wonn_num_layers": spec["layers"],
                        "wonn_num_oscillators": 768,
                        "wonn_num_inner_steps": 3,
                        "wonn_num_heads": 12,
                        "wonn_step_init": 0.1,
                        "wonn_step_max": 0.25,
                    })
                (run_dir / "config.yml").write_text(
                    yaml.safe_dump(config), encoding="utf-8",
                )
                (run_dir / "training_complete.json").write_text(json.dumps({
                    "status": "complete",
                    "completed_optimizer_step": 100000,
                    "effective_batch_size": 512,
                    "samples_seen": 100000 * 512,
                    "world_size": 1,
                    "batch_size_per_device": 16,
                    "grad_accum_steps": 32,
                    "learning_rate": spec["lr"],
                    "model": spec["model"],
                    "model_parameters": 100,
                    "elapsed_training_seconds": 10.0,
                    "peak_allocated_cuda_mib": 200.0,
                }), encoding="utf-8")
                training_rows = []
                for step in CHECKPOINT_STEPS:
                    (run_dir / f"checkpoint_{step}").write_bytes(b"checkpoint")
                    training_rows.append({
                        "step": step * 32,
                        "optimizer_step": step,
                        "samples_seen": step * 512,
                        "loss": 1.0,
                        "l2_loss": 0.5,
                        "ce_loss": 1.5,
                        "lr": spec["lr"],
                        "samples_per_second": 10.0,
                        "elapsed_training_seconds": float(step),
                    })
                    hypotheses = [f"translation {index}" for index in range(4)]
                    hypotheses[-1] = f"{key} output"
                    _write_evaluation(run_dir, spec["label"], step, hypotheses)
                (run_dir / "train_metrics.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in training_rows),
                    encoding="utf-8",
                )

            output = root / "analysis"
            payload = analyze_matrix(root, output, expected_samples=4)
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(len(payload["metrics_by_checkpoint"]), 28)
            self.assertTrue((output / "comparison.json").is_file())
            self.assertTrue((output / "metrics_by_checkpoint.csv").is_file())
            chart = (output / "quality_curves.svg").read_text(encoding="utf-8")
            for spec in MODEL_SPECS.values():
                self.assertIn(spec["label"], chart)

            generated = next(
                (root / "w0" / "evaluations" / "checkpoint_10000").glob(
                    "*/all_generated_*_*.jsonl"
                )
            )
            rows = [
                json.loads(line)
                for line in generated.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["source"] = "different input"
            generated.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "evaluation inputs differ"):
                analyze_matrix(root, output, expected_samples=4)


if __name__ == "__main__":
    unittest.main()

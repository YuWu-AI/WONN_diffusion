import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_cloud_pair import (
    CHECKPOINT_STEPS,
    ELF_LABEL,
    EXPECTED_STEPS,
    WONN_LABEL,
    _metric_bundle,
    _training_rows,
    analyze_pair,
    validate_pair_configs,
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


class CloudPairConfigTest(unittest.TestCase):
    def test_configs_are_fair_from_scratch_20k_pair(self):
        config_root = REPO_ROOT / "src/configs/training_configs"
        payload = validate_pair_configs(
            config_root / "train_de-en-ELF-B-cloud-60k.yml",
            config_root / "train_de-en-WONN-L12K768T3-cloud-60k.yml",
            world_size=2,
        )
        self.assertEqual(payload["terminal_optimizer_step"], 20000)
        self.assertEqual(payload["checkpoint_steps"], {
            label: list(EXPECTED_STEPS[label]) for label in (ELF_LABEL, WONN_LABEL)
        })
        self.assertEqual(payload["effective_batch_size"], 24)
        self.assertEqual(payload["batch_size_per_device"], 12)
        self.assertEqual(payload["learning_rate"], 0.0005)
        self.assertEqual(payload["warmup_steps"], 3000)

    def test_runtime_rejects_a_target_other_than_20k(self):
        config_root = REPO_ROOT / "src/configs/training_configs"
        with self.assertRaisesRegex(ValueError, "target steps must be 20000"):
            validate_pair_configs(
                config_root / "train_de-en-ELF-B-cloud-60k.yml",
                config_root / "train_de-en-WONN-L12K768T3-cloud-60k.yml",
                effective_batch=24,
                world_size=2,
                target_steps=30000,
            )

    def test_four_rank_layout_is_rejected(self):
        config_root = REPO_ROOT / "src/configs/training_configs"
        with self.assertRaisesRegex(ValueError, "world_size must be 2"):
            validate_pair_configs(
                config_root / "train_de-en-ELF-B-cloud-60k.yml",
                config_root / "train_de-en-WONN-L12K768T3-cloud-60k.yml",
                effective_batch=24,
                world_size=4,
            )

    def test_pipeline_owns_training_and_four_gpu_evaluation(self):
        pipeline = (REPO_ROOT / "scripts/run_cloud_pair_pipeline.sh").read_text(
            encoding="utf-8",
        )
        self.assertIn('layout="${DLM_WONN_GPU_LAYOUT:-2+2}"', pipeline)
        self.assertNotIn("4-serial", pipeline)
        self.assertIn('target_steps="${DLM_WONN_TARGET_STEPS:-20000}"', pipeline)
        self.assertIn('--config_override "lr=$learning_rate"', pipeline)
        self.assertIn('--config_override "max_optimizer_steps=$target_steps"', pipeline)
        self.assertIn('--config_override "save_optimizer_steps=$save_steps"', pipeline)
        self.assertIn('elf_gpus="${DLM_WONN_ELF_GPUS:-0,1}"', pipeline)
        self.assertIn('wonn_gpus="${DLM_WONN_WONN_GPUS:-2,3}"', pipeline)
        self.assertIn("-m torch.distributed.run", pipeline)
        self.assertIn("for index in 0 1 2 3", pipeline)
        self.assertIn("quality_curves.svg", pipeline)
        self.assertNotIn("DLM_WONN_ELF_BASELINE", pipeline)


class CloudPairAnalysisTest(unittest.TestCase):
    def test_training_metrics_must_be_strictly_monotonic(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "train_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in (
                    {"optimizer_step": 5, "samples_seen": 120},
                    {"optimizer_step": 5, "samples_seen": 120},
                )),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not strictly monotonic"):
                _training_rows(run_dir, (5,))

    def test_20k_analysis_accepts_missing_elf_5k_and_writes_one_composite_chart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common_config = {
                "global_batch_size": 24,
                "batch_size": 12,
                "grad_accum_steps": 1,
                "lr": 0.0005,
                "blr": 0.001,
                "lr_schedule": "constant",
                "warmup_steps": 3000,
                "optimizer": "muon",
                "seed": 42,
                "data_revision": "train-revision",
                "eval_data_revision": "eval-revision",
                "encoder_revision": "encoder-revision",
                "tokenizer_revision": "tokenizer-revision",
                "sampling_configs_path": "sampling.yml",
                "num_samples": 500,
                "max_optimizer_steps": 20000,
            }
            specs = (
                (ELF_LABEL, root / "elf", "ELF-B"),
                (WONN_LABEL, root / "wonn", "ELF-WONN-B"),
            )
            for label, run_dir, model in specs:
                run_dir.mkdir()
                model_steps = CHECKPOINT_STEPS[20000][label]
                config = {
                    **common_config,
                    "model": model,
                    "save_optimizer_steps": ",".join(str(step) for step in model_steps),
                }
                if label == ELF_LABEL:
                    config["resume"] = str(run_dir)
                    (run_dir / "training_resumed.json").write_text(
                        json.dumps({"resume_train_step": 10000}), encoding="utf-8",
                    )
                if model == "ELF-WONN-B":
                    config.update({
                        "wonn_num_layers": 12,
                        "wonn_num_oscillators": 768,
                        "wonn_num_inner_steps": 3,
                        "wonn_num_heads": 12,
                    })
                (run_dir / "config.yml").write_text(
                    yaml.safe_dump(config), encoding="utf-8",
                )
                (run_dir / "training_complete.json").write_text(json.dumps({
                    "status": "complete",
                    "completed_optimizer_step": 20000,
                    "effective_batch_size": 24,
                    "samples_seen": 20000 * 24,
                    "world_size": 2,
                    "batch_size_per_device": 12,
                    "learning_rate": 0.0005,
                    "model": model,
                    "model_parameters": 100,
                    "elapsed_training_seconds": 10.0,
                    "peak_allocated_cuda_mib": 200.0,
                }), encoding="utf-8")
                training_rows = []
                for step in model_steps:
                    (run_dir / f"checkpoint_{step}").write_bytes(b"checkpoint")
                    training_rows.append({
                        "optimizer_step": step,
                        "samples_seen": step * 24,
                        "loss": 1.0,
                        "l2_loss": 0.5,
                        "ce_loss": 1.5,
                        "lr": 0.0005,
                        "samples_per_second": 10.0,
                        "elapsed_training_seconds": float(step),
                    })
                    hypotheses = [f"translation {index}" for index in range(4)]
                    if label == WONN_LABEL:
                        hypotheses[-1] = "different"
                    _write_evaluation(run_dir, label, step, hypotheses)
                (run_dir / "train_metrics.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in training_rows),
                    encoding="utf-8",
                )

            output = root / "analysis"
            payload = analyze_pair(root / "elf", root / "wonn", output, 4)
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(
                payload["comparison_contract"]["terminal_optimizer_step"], 20000,
            )
            self.assertEqual(len(payload["metrics_by_checkpoint"]), 7)
            self.assertTrue((output / "comparison.json").is_file())
            self.assertTrue((output / "metrics_by_checkpoint.csv").is_file())
            chart = (output / "quality_curves.svg").read_text(encoding="utf-8")
            self.assertIn("BLEU vs optimizer step", chart)
            self.assertIn("chrF++ vs samples seen", chart)
            self.assertIn(ELF_LABEL, chart)
            self.assertIn(WONN_LABEL, chart)


if __name__ == "__main__":
    unittest.main()

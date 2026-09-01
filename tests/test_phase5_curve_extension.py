import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_phase5_50k import _metric_bundle
from analyze_phase5_curve_extension import build_report, evaluate_convergence
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


def _metric_row(step: int, loss: float) -> dict:
    return {
        "optimizer_step": step,
        "loss": loss,
        "l2_loss": loss,
        "ce_loss": loss,
        "lr": 0.001,
        "samples_per_second": 10.0,
    }


class Phase5CurveConfigTest(unittest.TestCase):
    def test_extension_configs_preserve_contract_and_set_separate_caps(self):
        root = REPO_ROOT / "src/configs/training_configs"
        elf = load_config_from_yaml(str(root / "train_de-en-ELF-B-phase5-90k.yml"))
        wonn = load_config_from_yaml(str(root / "train_de-en-WONN-L6T3-phase5-130k.yml"))
        for field in (
            "data_path", "eval_data_path", "data_revision", "eval_data_revision",
            "max_length", "max_input_length", "encoder_model_name",
            "encoder_revision", "tokenizer_revision", "batch_size", "lr",
            "lr_schedule", "warmup_steps", "optimizer", "seed", "num_workers",
        ):
            self.assertEqual(getattr(elf, field), getattr(wonn, field), field)
        self.assertEqual(elf.max_optimizer_steps, 90000)
        self.assertEqual(wonn.max_optimizer_steps, 130000)
        self.assertEqual(wonn.wonn_num_inner_steps, 3)
        self.assertIsNone(elf.resume)
        self.assertIsNone(wonn.resume)
        self.assertNotEqual(elf.output_dir, wonn.output_dir)
        self.assertEqual(
            elf.output_dir,
            "outputs/phase5/elf_runs/b12_seed42/curve_50_90k",
        )
        self.assertEqual(
            wonn.output_dir,
            "outputs/phase5/wonn_runs/l6t3_seed42_b12/curve_50_130k/"
            "wonn_l6t3_seed42_b12",
        )

    def test_pipeline_retains_source_diagnostics_and_terminal_full_eval(self):
        pipeline = (REPO_ROOT / "scripts/run_phase5_curve_extension_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('train_to_plateau "Transformer ELF-B"', pipeline)
        self.assertIn("outputs/phase5/elf_b_seed42_b12_0_90k", pipeline)
        self.assertIn('"$wonn_base" 130000', pipeline)
        self.assertIn('--elf-extension "$elf_baseline"', pipeline)
        self.assertIn('--wonn-extension "$wonn_dir"', pipeline)
        self.assertIn('step % 20000', pipeline)
        self.assertIn('evaluate_checkpoint "$label" "$config_path" "$output_dir" "$terminal_step" 3000', pipeline)
        self.assertIn('--bootstrap-resamples 1000', pipeline)


class Phase5ConvergenceMonitorTest(unittest.TestCase):
    def _runs(self, root: Path, collapsed: bool = False):
        base = root / "base"
        extension = root / "extension"
        history = extension / "monitoring"
        base.mkdir(parents=True)
        extension.mkdir(parents=True)
        references = [f"translated sentence {index}" for index in range(1000)]
        hypotheses = ["same output" for _ in references] if collapsed else references
        _write_evaluation(base, 50000, hypotheses)
        _write_evaluation(extension, 60000, hypotheses)
        _write_evaluation(extension, 70000, hypotheses)
        (base / "train_metrics.jsonl").write_text(
            json.dumps(_metric_row(50000, 1.0)) + "\n", encoding="utf-8",
        )
        extension_rows = [
            _metric_row(51000, 1.0), _metric_row(55000, 0.997),
            _metric_row(56000, 0.996), _metric_row(60000, 0.994),
            _metric_row(61000, 0.993), _metric_row(65000, 0.991),
            _metric_row(66000, 0.990), _metric_row(70000, 0.988),
        ]
        (extension / "train_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in extension_rows),
            encoding="utf-8",
        )
        return base, extension, history

    def test_two_consecutive_quality_and_loss_plateaus_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, extension, history = self._runs(Path(temporary))
            first = evaluate_convergence(
                "test", base, extension, 60000, history, bootstrap_resamples=10,
            )
            self.assertTrue(first["interval_plateau"])
            self.assertEqual(first["decision"], "continue")
            first_path = history / "checkpoint_60000/convergence.json"
            first_path.parent.mkdir(parents=True)
            first_path.write_text(json.dumps(first), encoding="utf-8")
            second = evaluate_convergence(
                "test", base, extension, 70000, history, bootstrap_resamples=10,
            )
            self.assertEqual(second["consecutive_plateau_intervals"], 2)
            self.assertEqual(second["decision"], "stop_converged")

    def test_output_collapse_is_abnormal_not_convergence(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, extension, history = self._runs(Path(temporary), collapsed=True)
            result = evaluate_convergence(
                "test", base, extension, 60000, history, bootstrap_resamples=5,
            )
            self.assertTrue(result["abnormal_checks"]["unique_rate_below_limit"])
            self.assertEqual(result["decision"], "abort_abnormal")


class Phase5CurveReportTest(unittest.TestCase):
    def test_report_separates_same_step_and_endpoint_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "curve"
            elf_base = Path(temporary) / "elf_base"
            wonn_base = Path(temporary) / "wonn_base"
            elf_extension = root / "elf_b_seed42_b12"
            wonn_extension = root / "wonn_l6t3_seed42_b12"
            for base, extension, model, parameters in (
                (elf_base, elf_extension, "ELF-B", 100),
                (wonn_base, wonn_extension, "ELF-WONN-B", 50),
            ):
                base.mkdir(parents=True)
                extension.mkdir(parents=True)
                _write_evaluation(
                    base, 50000,
                    [f"translated sentence {index}" for index in range(1000)],
                )
                _write_evaluation(
                    extension, 60000,
                    [f"translated sentence {index}" for index in range(3000)],
                )
                (base / "training_complete.json").write_text(json.dumps({
                    "status": "complete",
                    "completed_optimizer_step": 50000,
                    "elapsed_training_seconds": 100.0,
                    "peak_allocated_cuda_mib": 100.0,
                }), encoding="utf-8")
                (extension / "training_complete.json").write_text(json.dumps({
                    "status": "complete",
                    "completed_optimizer_step": 60000,
                    "elapsed_training_seconds": 20.0,
                    "peak_allocated_cuda_mib": 90.0,
                    "model_parameters": parameters,
                    "model": model,
                }), encoding="utf-8")
                (base / "train_metrics.jsonl").write_text(
                    json.dumps(_metric_row(50000, 1.0)) + "\n", encoding="utf-8",
                )
                (extension / "train_metrics.jsonl").write_text(
                    json.dumps(_metric_row(60000, 0.9)) + "\n", encoding="utf-8",
                )
                (extension / "terminal_status.json").write_text(json.dumps({
                    "status": "complete",
                    "terminal_step": 60000,
                    "reason": "budget_cap",
                }), encoding="utf-8")
            output = root / "analysis/final"
            result = build_report(
                root, elf_base, wonn_base, 60000, 60000, output,
                bootstrap_resamples=2,
            )
            self.assertEqual(result["common_step"], 60000)
            self.assertTrue((output / "comparison.json").is_file())
            report = (output / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("Fair same-step checkpoint", report)
            self.assertIn("not a same-budget claim", report)


if __name__ == "__main__":
    unittest.main()

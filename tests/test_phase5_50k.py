import sys
import tempfile
import unittest
from unittest import mock
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_phase5_50k
from analyze_phase5_50k import _evaluate_gate, _metric_bundle, _paired_bootstrap
from phase5_artifacts import (
    checkpoint_path,
    completion_path,
    config_path,
    retained_checkpoint_steps,
    source_commit_path,
    terminal_status_path,
    training_metric_paths,
)
from evaluate_phase5_wonn_gate import (
    STAGES as WONN_GATE_STAGES,
    _evaluate_gate as _evaluate_wonn_gate,
)
from configs.config import load_config_from_yaml


class Phase550KConfigTest(unittest.TestCase):
    def test_paired_configs_share_the_training_contract(self):
        config_root = REPO_ROOT / "src/configs/training_configs"
        elf = load_config_from_yaml(
            str(config_root / "train_de-en_ELF-B-phase5-50k.yml")
        )
        wonn = load_config_from_yaml(
            str(config_root / "train_de-en-WONN-L6T3-phase5-50k.yml")
        )
        shared_fields = (
            "data_path", "eval_data_path", "data_revision", "eval_data_revision",
            "max_length", "max_input_length", "pad_token", "encoder_model_name",
            "encoder_revision", "tokenizer_revision", "latent_mean", "latent_std",
            "bottleneck_dim", "num_time_tokens", "num_self_cond_cfg_tokens",
            "num_model_mode_tokens", "denoiser_p_mean", "denoiser_p_std",
            "denoiser_noise_scale", "t_eps", "time_schedule", "decoder_prob",
            "decoder_noise_scale", "decoder_p_mean", "decoder_p_std",
            "label_drop_prob", "self_cond_prob", "batch_size", "grad_accum_steps",
            "max_optimizer_steps", "lr", "weight_decay", "warmup_steps",
            "optimizer", "compile_train", "gradient_checkpointing", "ema_decay1",
            "sampling_configs_path", "num_samples", "online_eval", "use_compile",
            "log_freq", "save_optimizer_steps", "save_freq", "eval_freq",
            "final_eval", "seed", "num_workers",
        )
        for field in shared_fields:
            self.assertEqual(getattr(elf, field), getattr(wonn, field), field)
        self.assertEqual(elf.model, "ELF-B")
        self.assertEqual(wonn.model, "ELF-WONN-B")
        self.assertEqual(wonn.wonn_num_inner_steps, 3)
        self.assertEqual(elf.batch_size, 12)
        self.assertEqual(elf.max_optimizer_steps, 50000)
        self.assertEqual(elf.num_workers, 2)
        self.assertFalse(elf.final_eval)
        self.assertFalse(wonn.final_eval)
        self.assertIsNone(elf.resume)
        self.assertIsNone(wonn.resume)
        self.assertIsNone(elf.init_from)
        self.assertIsNone(wonn.init_from)
        self.assertEqual(
            elf.output_dir,
            "outputs/phase5/elf_runs/b12_seed42/formal_0_50k",
        )
        self.assertEqual(
            wonn.output_dir,
            "outputs/phase5/wonn_runs/l6t3_seed42_b12/formal_0_50k/"
            "wonn_l6t3_seed42_b12",
        )
        self.assertTrue(elf.data_revision)
        self.assertTrue(elf.eval_data_revision)
        self.assertTrue(elf.encoder_revision)
        self.assertTrue(elf.tokenizer_revision)


class Phase550KAnalysisTest(unittest.TestCase):
    def test_metric_bundle_has_expected_perfect_scores(self):
        metrics = _metric_bundle(
            ["the cat is here", "a small test"],
            ["the cat is here", "a small test"],
        )
        self.assertAlmostEqual(metrics["bleu"], 100.0)
        self.assertAlmostEqual(metrics["chrf2"], 100.0)
        self.assertAlmostEqual(metrics["ter"], 0.0)
        self.assertEqual(metrics["empty_rate_pct"], 0.0)
        self.assertEqual(metrics["length_ratio"], 1.0)

    def test_paired_bootstrap_detects_consistent_wonn_gain(self):
        sources = [f"source {index}" for index in range(12)]
        references = [f"translated sentence number {index}" for index in range(12)]
        elf = {
            "sources": sources,
            "references": references,
            "hypotheses": ["" for _ in references],
        }
        wonn = {
            "sources": sources,
            "references": references,
            "hypotheses": list(references),
        }
        result = _paired_bootstrap(elf, wonn, resamples=50, seed=7)
        self.assertGreater(
            result["bleu_delta_wonn_minus_elf"]["ci95_low"], 0.0
        )
        self.assertGreater(
            result["chrf2_delta_wonn_minus_elf"]["ci95_low"], 0.0
        )

    def test_paired_bootstrap_rejects_different_references(self):
        elf = {
            "sources": ["a"], "references": ["one"], "hypotheses": ["one"],
        }
        wonn = {
            "sources": ["a"], "references": ["two"], "hypotheses": ["two"],
        }
        with self.assertRaisesRegex(ValueError, "references are not paired"):
            _paired_bootstrap(elf, wonn, resamples=2, seed=1)

    def test_20k_gate_requires_quality_noncollapse_and_learning_trend(self):
        def run(
            bleu_10k, bleu_20k, chrf_10k, chrf_20k, empty_20k,
            unique_20k=99.0, length_ratio_20k=1.0,
        ):
            return {"evaluations": {
                10000: {"metrics": {"bleu": bleu_10k, "chrf2": chrf_10k}},
                20000: {"metrics": {
                    "bleu": bleu_20k,
                    "chrf2": chrf_20k,
                    "empty_rate_pct": empty_20k,
                    "unique_rate_pct": unique_20k,
                    "length_ratio": length_ratio_20k,
                }},
            }}

        passing = _evaluate_gate({
            "ELF-B": run(0.1, 2.0, 4.0, 20.0, 2.0),
            "WONN-L6T3": run(0.2, 1.2, 13.0, 16.0, 3.0),
        })
        self.assertTrue(passing["passed"])

        failing = _evaluate_gate({
            "ELF-B": run(0.1, 8.0, 4.0, 35.0, 0.0),
            "WONN-L6T3": run(0.2, 0.3, 13.0, 14.0, 11.0),
        })
        self.assertFalse(failing["passed"])

    def test_pilot_gate_requires_wonn_source_sensitivity(self):
        run = {"evaluations": {
            5000: {"metrics": {"bleu": 0.02, "chrf2": 10.0}},
            10000: {"metrics": {
                "bleu": 0.2,
                "chrf2": 15.0,
                "empty_rate_pct": 0.0,
                "unique_rate_pct": 99.0,
                "length_ratio": 1.0,
            }},
        }}
        diagnostics = {
            "status": "complete", "num_samples": 256, "seed": 42,
            "models": {"WONN-L6T3": {"conditioning": {
            "correct": {"chrf2": 15.0},
            "shuffled": {"chrf2": 14.0},
            "zero": {"chrf2": 13.0},
            "shuffle_construction": {"fixed_target_offsets": True},
        }}},
        }
        self.assertTrue(_evaluate_wonn_gate(
            run, diagnostics, WONN_GATE_STAGES["pilot10k"],
        )["passed"])

        diagnostics["models"]["WONN-L6T3"]["conditioning"]["shuffled"]["chrf2"] = 14.9
        self.assertFalse(_evaluate_wonn_gate(
            run, diagnostics, WONN_GATE_STAGES["pilot10k"],
        )["passed"])

    def test_end_to_end_analysis_writes_comparison_artifacts(self):
        references = ["a translated sentence", "another translated sentence", "final text"]
        sources = ["quelle a", "quelle b", "quelle c"]
        hypotheses = {
            "ELF-B": ["", "another sentence", "text"],
            "WONN-L6T3": list(references),
        }
        model_specs = {
            "ELF-B": {
                "directory": "elf", "config_model": "ELF-B",
                "wonn_inner_steps": None,
            },
            "WONN-L6T3": {
                "directory": "wonn", "config_model": "ELF-WONN-B",
                "wonn_inner_steps": 3,
            },
        }
        stage = {
            "target_step": 50000,
            "checkpoint_steps": (50000,),
            "evaluation_steps": (50000,),
            "expected_samples": {50000: 3},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "runs"
            output = Path(tmpdir) / "analysis"
            root.mkdir(parents=True)
            (root / "experiment_manifest.json").write_text(json.dumps({
                "status": "ready", "source_commit": "abc123",
            }), encoding="utf-8")
            for label, spec in model_specs.items():
                run_dir = root / spec["directory"]
                eval_dir = run_dir / "evaluations/checkpoint_50000/ode-test-cond"
                eval_dir.mkdir(parents=True)
                (run_dir / "source_commit").write_text(
                    "abc123\n", encoding="utf-8",
                )
                (run_dir / "checkpoint_50000").write_bytes(b"checkpoint")
                resolved_config = {
                    "model": spec["config_model"],
                    "batch_size": 12,
                    "seed": 42,
                    "init_from": None,
                    "data_revision": "train-rev",
                    "eval_data_revision": "eval-rev",
                    "encoder_revision": "encoder-rev",
                    "tokenizer_revision": "tokenizer-rev",
                }
                if spec["wonn_inner_steps"] is not None:
                    resolved_config["wonn_num_inner_steps"] = spec["wonn_inner_steps"]
                (run_dir / "config.yml").write_text(
                    json.dumps(resolved_config), encoding="utf-8",
                )
                (run_dir / "training_complete.json").write_text(json.dumps({
                    "status": "complete",
                    "completed_optimizer_step": 50000,
                    "model": spec["config_model"],
                    "effective_batch_size": 12,
                    "samples_seen": 600000,
                    "model_parameters": 10,
                    "elapsed_training_seconds": 20.0,
                    "peak_allocated_cuda_mib": 100.0,
                }), encoding="utf-8")
                (run_dir / "train_metrics.jsonl").write_text(json.dumps({
                    "step": 50000,
                    "optimizer_step": 50000,
                    "loss": 1.0,
                    "l2_loss": 0.5,
                    "ce_loss": 1.5,
                    "lr": 0.001,
                    "samples_seen": 600000,
                    "elapsed_training_seconds": 20.0,
                    "samples_per_second": 30.0,
                }) + "\n", encoding="utf-8")
                generated_rows = [
                    {
                        "id": index,
                        "source": sources[index],
                        "reference": references[index],
                        "generated": hypotheses[label][index],
                    }
                    for index in range(3)
                ]
                (eval_dir / "all_generated_0_50000.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in generated_rows),
                    encoding="utf-8",
                )
                metrics = _metric_bundle(hypotheses[label], references)
                (eval_dir / "metrics.jsonl").write_text(json.dumps({
                    "step": 50000,
                    "num_samples": 3,
                    "generation_seconds": 2.0,
                    "decode_seconds": 0.1,
                    "samples_per_second": 1.4,
                    "timing_scope": "cuda_synchronized_sampler_and_decoder",
                    "peak_allocated_cuda_mib": 50.0,
                    "bleu": metrics["bleu"],
                    "rouge1": 1.0,
                    "rouge2": 1.0,
                    "rougeL": 1.0,
                }) + "\n", encoding="utf-8")

            with mock.patch.object(analyze_phase5_50k, "MODEL_SPECS", model_specs), \
                    mock.patch.object(analyze_phase5_50k, "STAGES", {"final50k": stage}):
                result = analyze_phase5_50k.analyze(
                    root, output, "final50k",
                    verify_checkpoints=False, bootstrap_resamples=10,
                )

            self.assertEqual(result["status"], "complete")
            self.assertTrue((output / "comparison.json").is_file())
            self.assertTrue((output / "comparison.md").is_file())
            self.assertTrue((output / "training_summary.csv").is_file())
            self.assertTrue((output / "evaluation_summary.csv").is_file())
            report = (output / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("## Executive result", report)
            self.assertIn("## Learning curve", report)

    def test_pipeline_records_intermediate_gates_without_blocking_50k(self):
        pipeline = (
            REPO_ROOT / "scripts/run_phase5_50k_pipeline.sh"
        ).read_text(encoding="utf-8")
        pilot_position = pipeline.index('run_gate "pilot10k" 10000')
        gate_position = pipeline.index('run_gate "gate20k" 20000')
        continuation_position = pipeline.index(
            'run_training "WONN-L6T3" "$wonn_config" "$wonn_dir" 50000'
        )
        self.assertLess(pilot_position, gate_position)
        self.assertLess(gate_position, continuation_position)
        self.assertNotIn('run_training "ELF-B"', pipeline)
        self.assertIn('--elf-run-dir "$elf_baseline"', pipeline)
        self.assertIn("outputs/phase5/elf_b_seed42_b12_0_90k", pipeline)
        self.assertIn("observed_failed_continuing", pipeline)
        self.assertNotIn("exit 20", pipeline)


class Phase5ArtifactPathTest(unittest.TestCase):
    def test_consolidated_elf_layout_resolves_without_native_run_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            for directory in ("checkpoints", "training", "provenance"):
                (run_dir / directory).mkdir()
            self.assertEqual(
                checkpoint_path(run_dir, 50000),
                run_dir / "checkpoints/checkpoint_50000",
            )
            self.assertEqual(
                retained_checkpoint_steps(
                    run_dir, (5000, 10000, 15000, 20000),
                ),
                [10000, 20000],
            )
            self.assertEqual(
                completion_path(run_dir, 50000),
                run_dir / "provenance/training_complete_50000.json",
            )
            self.assertEqual(
                config_path(run_dir, 90000),
                run_dir / "provenance/config_50000_90000.yml",
            )
            self.assertEqual(
                source_commit_path(run_dir, 50000),
                run_dir / "provenance/source_commit_00000_50000.txt",
            )
            self.assertEqual(
                terminal_status_path(run_dir, 90000),
                run_dir / "provenance/terminal_status_90000.json",
            )
            self.assertEqual(
                training_metric_paths(run_dir, 90000),
                [
                    run_dir / "training/train_metrics_00000_50000.jsonl",
                    run_dir / "training/train_metrics_50000_90000.jsonl",
                ],
            )


if __name__ == "__main__":
    unittest.main()

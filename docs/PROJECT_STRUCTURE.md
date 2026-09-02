# DLM-WONN 代码与实验目录索引

本文区分当前执行入口与历史复现入口，并说明代码和产物边界。文档总索引见
[`README.md`](README.md)，研究边界见 [`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md)，后续阶段见
[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)。

## 1. 主执行链路

| 路径 | 职责 |
| --- | --- |
| `src/train.py` | 训练循环、optimizer step 预算、resume、checkpoint 与最终状态标记 |
| `src/train_step.py` | denoising/decoding objective、梯度累积和单步更新 |
| `src/eval.py` | checkpoint 采样、翻译/摘要指标与评测输出 |
| `src/generation.py` | ODE/SDE sampler、self-conditioning 和条件位置恢复 |
| `src/utils/` | 数据加载、训练状态、checkpoint 与分布式公共工具 |

## 2. 模型与配置

| 路径 | 职责 |
| --- | --- |
| `src/modules/model.py` | 官方 ELF Transformer backbone 与共享 decoder 契约 |
| `src/modules/wonn_layers.py` | S/I、QKV/O attentive coupling、phase update、OmegaTransition |
| `src/modules/wonn_model.py` | ELF-compatible WONN 完整模型 |
| `src/modules/model_factory.py` | ELF/WONN 公共构造入口 |
| `src/configs/training_configs/` | 官方 ELF、基础 WONN 和当前云端配对配置；分类见目录内 `README.md` |
| `src/configs/sampling_configs/` | 条件与无条件采样配置 |

官方 ELF YAML 保持基线用途；WONN 和云端 YAML 是独立派生配置，不覆盖上游基线。

## 3. 当前脚本入口

| 分组 | 入口 |
| --- | --- |
| 通用启动 | `scripts/launch.sh` |
| 契约验证 | `scripts/verify_phase2.py`、`verify_phase3.py`、`verify_phase4.py` |
| 训练工程公共工具 | `cloud_pipeline_checks.py`、`profile_cloud_training.py` |
| 4 GPU 云端配对 | `bootstrap_cloud_env.sh`、`verify_cloud_env.sh`、`run_cloud_pair_pipeline.sh`、`analyze_cloud_pair.py`；执行口径见 [`CLOUD_PHASE1_RUNBOOK.md`](CLOUD_PHASE1_RUNBOOK.md) |

云端容器应在 `tmux` 中直接运行 `run_*_pipeline.sh`。所有正式流水线都会拒绝 dirty checkout，
并将 commit、resolved config、数据 revision、seed、训练状态和评测完成标记写入输出目录。
云端配对入口默认以 GPU 0–1 训练 ELF、GPU 2–3 训练 WONN，训练完成后四卡并行消费 checkpoint
评测队列；也支持 profile Gate 选出的四卡串行模式，但同一次正式 run 不得切换 world size。

## 4. 测试

`tests/test_model_*` 和 `test_wonn_*` 覆盖模型接口与动力学；`test_data_contract.py`、
`test_sampling_contract.py`、`test_checkpoint_utils.py` 覆盖 ELF 公共边界；`test_training.py` 和
`test_cloud_*` 覆盖训练预算、resume、4-GPU 流水线、产物检查和分析。完整 CPU 回归入口为：

```bash
.venv/bin/python -m unittest discover -v
```

## 5. Phase 5 历史数据

本地实验产物统一位于 Git 忽略的 `outputs/phase5/`，不随代码提交：

| 路径 | 内容 |
| --- | --- |
| `outputs/phase5/elf_b_seed42_b12_0_90k/` | Transformer ELF-B 0–90K 合并归档；保留 10K 间隔 checkpoint、评测、训练指标与 provenance |
| `outputs/phase5/wonn_l12k768t3_seed42_b12_0_90k/` | WONN-L12/K768/T3 0–90K 完整训练、9 个 checkpoint 评测与最终曲线分析 |
| `outputs/phase5/learning_rate_diagnostic_20260825/` | 历史学习率诊断报告 |

这些目录是需要保留的历史证据，但当前代码不再包含对应旧流水线。新运行必须写入独立的
`DLM_WONN_PAIR_RUN_ROOT`，避免覆盖归档。

## 6. 其他目录

| 路径 | 职责 |
| --- | --- |
| `docs/` | 现役计划和运行手册；状态索引见 `docs/README.md` |
| `docs/architecture/` | 与现行实现一致的算法和公式说明 |
| `papers/` | ELF/WONN 论文 |
| `references/` | 只读上游参考实现；运行时代码不得导入 |
| `reports/` | 可提交的静态说明或可视化报告源码 |
| `outputs/` | 不提交的训练和评测产物 |

退役脚本、配置和测试可从 Git 历史追溯，不保留在当前执行目录。

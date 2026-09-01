# DLM-WONN 代码与实验目录索引

本文回答两个问题：代码各部分负责什么，以及 Phase 5 的入口和历史数据在哪里。研究边界见
[`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md)，后续阶段见 [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)。

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
| `src/modules/denoiser_objectives.py` | Phase 5 可选语义辅助 objective；默认权重为 0 |
| `src/configs/training_configs/` | 官方 ELF 配置与独立 WONN/Phase 5 配置 |
| `src/configs/training_configs/phase5_mechanism/` | Phase 5 机制消融配置 |
| `src/configs/sampling_configs/` | 条件/无条件采样及 Phase 5 诊断配置 |

官方 ELF YAML 保持基线用途；文件名含 `phase5` 或 `WONN` 的 YAML 是派生实验配置，不覆盖上游基线。

## 3. 脚本分组

| 分组 | 入口 |
| --- | --- |
| 通用启动 | `scripts/launch.sh` |
| 契约验证 | `scripts/verify_phase2.py`、`verify_phase3.py`、`verify_phase4.py` |
| Phase 5 训练工程 | `capture_phase5_manifest.py`、`phase5_pipeline_checks.py`、`profile_phase5.py` |
| L6/T3 50K 与曲线延伸 | `run_phase5_50k_pipeline.sh`、`run_phase5_curve_extension_pipeline.sh` |
| L12/K768/T3 90K | `run_wonn_l12k768_90k_pipeline.sh`、`analyze_wonn_l12k768_90k.py` |
| 机制实验 | `run_phase5_mechanism_pipeline.sh`、`diagnose_phase5_checkpoint.py`、`summarize_phase5_*` |
| 本地后台封装 | `launch_phase5_*.sh`、`launch_wonn_l12k768_90k.sh`；依赖 user systemd，不用于云容器 |

云端容器应在 `tmux` 中直接运行 `run_*_pipeline.sh`。所有正式流水线都会拒绝 dirty checkout，
并将 commit、resolved config、数据 revision、seed、训练状态和评测完成标记写入输出目录。

## 4. 测试

`tests/test_model_*` 和 `test_wonn_*` 覆盖模型接口与动力学；`test_data_contract.py`、
`test_sampling_contract.py`、`test_checkpoint_utils.py` 覆盖 ELF 公共边界；`test_phase5_*` 覆盖训练预算、
resume、流水线、分析和机制实验。完整 CPU 回归入口为：

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

这些目录是需要保留的历史证据。新运行分别写入 `outputs/phase5/elf_runs/` 与
`outputs/phase5/wonn_runs/`，避免覆盖归档。`scripts/phase5_artifacts.py` 同时解析原生 run 与合并 ELF
归档布局。

## 6. 其他目录

| 路径 | 职责 |
| --- | --- |
| `docs/` | 研究计划、架构公式、环境与验证记录 |
| `papers/` | ELF/WONN 论文 |
| `references/` | 只读上游参考实现；运行时代码不得导入 |
| `reports/` | 可提交的静态说明或可视化报告源码 |
| `outputs/` | 不提交的训练和评测产物 |

为保持已有命令和产物 lineage 可复现，Phase 5 脚本暂不物理移动到子目录；本索引是稳定导航层。

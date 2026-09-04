# 脚本索引

## 当前执行主线

| 脚本 | 用途 |
| --- | --- |
| `launch.sh` | 通用单卡/多卡 train 与 eval 启动器 |
| `bootstrap_cloud_env.sh` | 在 Ubuntu 云实例构建固定 Python 环境 |
| `verify_cloud_env.sh` | 验证 clean checkout、依赖、4 张 GPU、BF16 和 compile |
| `profile_cloud_training.py` | 固定配置 smoke 的单模型显存与吞吐测量工具 |
| `run_cloud_matrix_pipeline.sh` | 四张 GPU 各跑一个模型，完成 100K 训练、28 次评测和分析 |
| `analyze_cloud_matrix.py` | 校验四模型配置与产物，生成指标表和复合折线图 |
| `cloud_pipeline_checks.py` | checkpoint、评测和流水线状态校验 |

当前运行口径只以 `docs/CLOUD_PHASE1_RUNBOOK.md` 为准。

## 回归与契约验证

- `verify_phase2.py`：ELF backbone 契约。
- `verify_phase3.py`：WONN backbone 契约。
- `verify_phase4.py`：可学性和真实 batch smoke。
- `run_phase4_learnability.py`：Phase 4 验收实现。

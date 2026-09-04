# Phase 1：4-GPU、四模型、100K 云端运行手册

> 本文是当前 WMT14 实验的唯一运行口径。旧 pair 和小 batch 流水线只从 Git 历史追溯。

## 1. 完成定义

一次正式运行必须同时满足：

- E0/W0/W1/W2 来自同一 clean commit，并从随机初始化开始；
- GPU 0/1/2/3 各训练一个单进程模型，不使用 DDP；
- 每组训练到 100K optimizer steps，effective batch 512，warmup 5K optimizer steps；
- E0 学习率 `2e-3`，三组 WONN 学习率 `1e-3`，均为 constant schedule；
- 四组均保存 5K、10K、25K、40K、60K、80K、100K；
- 每个 checkpoint 评测相同 1000 条 validation 样本，共 28 次；
- `analysis/` 中生成 comparison JSON/Markdown、CSV 和 `quality_curves.svg`；
- `pipeline_complete.json` 存在且所有上游产物校验通过。

模型与配置：

| GPU | 编号 | 配置 |
| ---: | --- | --- |
| 0 | E0 | `train_de-en-ELF-B-E0.yml` |
| 1 | W0 | `train_de-en-WONN-L12K768T3-W0.yml` |
| 2 | W1 | `train_de-en-WONN-L6K768T3-W1.yml` |
| 3 | W2 | `train_de-en-WONN-L9K768T3-W2.yml` |

## 2. Batch 与 step 语义

训练配置中的 `global_batch_size` 是每次 optimizer update 的 effective batch：

```text
512 = per-device micro-batch 16 × world size 1 × grad_accum_steps 32
```

`warmup_steps`、`log_freq`、`max_optimizer_steps` 和 `save_optimizer_steps` 全部使用 optimizer-step
单位。checkpoint 文件名也使用 optimizer step；checkpoint 内部仍保存精确 micro-step 状态以供恢复。

100K 约为论文 WMT14 完整 880K 预算的 11.4%，用于架构筛选，不作为充分收敛结果。

## 3. 本地提交前验证

```bash
.venv/bin/python -m unittest discover -v
bash -n scripts/bootstrap_cloud_env.sh scripts/verify_cloud_env.sh \
  scripts/run_cloud_matrix_pipeline.sh
.venv/bin/python scripts/analyze_cloud_matrix.py validate-configs \
  --config-dir src/configs/training_configs
git status --short
```

正式上传前必须形成 clean commit；不复制 dirty 工作区。

## 4. 云端环境

```bash
sudo bash scripts/bootstrap_cloud_env.sh
export DLM_WONN_PYTHON=/opt/dlm-wonn-venv/bin/python
export DLM_WONN_EXPECTED_GPU_COUNT=4
bash scripts/verify_cloud_env.sh
```

固定缓存放在持久卷：

```bash
export HF_HOME=/root/shared-nvme/dlm-wonn/cache/huggingface
```

首次预下载可设置 `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0`；正式运行使用离线模式，避免数据和
模型 revision 漂移。

## 5. GPU profile gate

正式训练前，对四个配置分别使用 `scripts/profile_cloud_training.py` 测 micro-batch 16。每组至少在
compile 后稳定测量 30–100 个 micro-steps，记录 seconds/step、samples/second 和峰值显存。
profile 脚本的 warmup 参数只是性能预热，不是正式训练的 5K optimizer-step warmup。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 $DLM_WONN_PYTHON scripts/profile_cloud_training.py \
  --config src/configs/training_configs/train_de-en-ELF-B-E0.yml \
  --model ELF-B --decoder-prob 0.2 --batch-size 16 \
  --warmup-steps 30 --measure-steps 100 --output outputs/profile/e0.json
```

W0/W1/W2 使用各自配置、`--model ELF-WONN-B` 和独立输出文件。任何一组 OOM 都停止：将四份正式
配置统一改为 micro-batch 8 / accumulation 64，重新测试、提交并使用新的 run root；不得只覆盖
单个模型。

四模型并发会共享 CPU 与存储，因此并发吞吐用于估算总墙钟时间；模型间速度结论应另做隔离 profile。

## 6. Preflight

每次正式运行使用新的持久目录：

```bash
export DLM_WONN_PYTHON=/opt/dlm-wonn-venv/bin/python
commit_short="$(git rev-parse --short=7 HEAD)"
export DLM_WONN_RUN_ROOT="/root/shared-nvme/dlm-wonn/runs/wmt14-matrix-${commit_short}-100000"
export DLM_WONN_ALL_GPUS=0,1,2,3
export DLM_WONN_EVAL_BATCH_SIZE=16
export HF_HOME=/root/shared-nvme/dlm-wonn/cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DLM_WONN_PREFLIGHT_ONLY=1
bash scripts/run_cloud_matrix_pipeline.sh
```

Preflight 校验 clean checkout、四张可见 GPU、固定矩阵、batch/LR/warmup/checkpoint、输出目录名和
provenance。它不会开始训练。

## 7. 正式启动与恢复

```bash
unset DLM_WONN_PREFLIGHT_ONLY
tmux new -s dlm-wonn-matrix
bash scripts/run_cloud_matrix_pipeline.sh
```

监控：

```bash
nvidia-smi
tail -F "$DLM_WONN_RUN_ROOT/e0/pipeline.log"
tail -F "$DLM_WONN_RUN_ROOT/w0/pipeline.log"
tail -F "$DLM_WONN_RUN_ROOT/w1/pipeline.log"
tail -F "$DLM_WONN_RUN_ROOT/w2/pipeline.log"
```

实例重启后，只在确认没有存活训练进程时，以相同 commit、环境变量和 run root 重新执行 pipeline。
已完成模型会跳过，未完成模型从自己目录的最新 checkpoint 恢复。以下情况必须新建 run root：

- commit、模型、batch、LR、warmup、目标 step、checkpoint 或评测样本数变化；
- checkpoint 来自其他模型或其他输出目录；
- provenance 缺失，或目录曾被手工混合。

## 8. 产物

```text
<run-root>/
├── provenance/
├── e0/checkpoint_{step} + evaluations/checkpoint_{step}/
├── w0/checkpoint_{step} + evaluations/checkpoint_{step}/
├── w1/checkpoint_{step} + evaluations/checkpoint_{step}/
├── w2/checkpoint_{step} + evaluations/checkpoint_{step}/
├── analysis/comparison.json
├── analysis/metrics_by_checkpoint.csv
├── analysis/comparison.md
├── analysis/quality_curves.svg
└── pipeline_complete.json
```

结果回传本地时复制整个 run root 到新的 Git 忽略目录，不覆盖 `outputs/phase5/` 历史归档。100K 后
按 `RESEARCH_PLAN.md` 的 gate 选择候选；完整 3000 条 validation 复评另行执行。

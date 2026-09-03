# Phase 1：4-GPU、20K 云端配对实验运行手册

> 状态：代码已实现，尚未在目标 4-GPU 云主机完成 CUDA smoke 或正式训练。
> 本文是当前 WMT14 工程实验的唯一运行口径；历史 50K/90K/130K 流水线不属于本轮输入。

## 1. 完成定义

一次正式运行必须同时满足：

- 从同一 clean commit、同一 seed 和同一数据 revision 随机初始化 ELF-B 与
  WONN-L12/K768/T3；
- 两个模型均训练到 `20000` optimizer steps；
- 两边固定 global batch 24（每卡 12）、学习率 `5e-4`、warmup 3000、seed 42；
- WONN 保存 `5K, 10K, 15K, 20K`；ELF 本次运行已越过 5K，因此保留
  `10K, 15K, 20K`；
- 每个现存 checkpoint 使用同一批 500 个验证样本，共 7 次评测；
- 生成 `comparison.json`、`metrics_by_checkpoint.csv`、`comparison.md` 和一张
  `quality_curves.svg` 复合折线图；
- `pipeline_complete.json` 存在且内容通过产物校验。

WMT14 在本阶段用于比较两种模型的早期学习速度和生成能力，不追求充分收敛。新运行与旧小 batch
的 90K 并不等价；报告必须同时给 optimizer
steps 与 samples seen。

## 2. 不变边界

- 不使用本地历史 checkpoint、warm start、训练指标或生成结果。
- 不修改 ELF encoder、Flow Matching、conditioning、sampler 或 shared decoder。
- 不覆盖官方 ELF 配置或历史输出目录。
- 除 2026-09-03 用户明确将运行从 30K 缩短到 20K 的一次性迁移外，同一次 run 恢复时不得改变
  commit、GPU 布局、world size、有效 batch、学习率、warmup、目标 step、checkpoint 列表或评测口径。
- 正式训练只允许写入持久卷中的新目录。

当前配置：

- `src/configs/training_configs/train_de-en-ELF-B-cloud-60k.yml`
- `src/configs/training_configs/train_de-en-WONN-L12K768T3-cloud-60k.yml`

## 3. GPU 统筹

默认 `2+2`：

```text
GPU 0-1  -> ELF-B，2-rank DDP
GPU 2-3  -> WONN，2-rank DDP
两边并发训练；训练完成后 GPU 0-3 静态消费 8 个 checkpoint 评测任务
```

本轮不测试 `4-serial`，固定 `grad_accum_steps=1`：

```text
effective batch = per-device batch × world size
```

配置中的 `global_batch_size=24` 在 2+2 时解析为每 rank 12。不要因 GPU 利用率未达到 100%
而扩大 batch；以端到端用时和稳定性为准。

## 4. clean commit gate

在本地完成后提交代码，并记录 commit：

```bash
git status --short
git rev-parse HEAD
.venv/bin/python -m unittest discover -v
bash -n scripts/bootstrap_cloud_env.sh scripts/verify_cloud_env.sh \
  scripts/run_cloud_pair_pipeline.sh
```

`git status --short` 必须为空。不要复制 dirty worktree 作为正式实验版本。

## 5. 构建和验证云环境

在 Ubuntu x86_64 云实例中：

```bash
sudo bash scripts/bootstrap_cloud_env.sh
export DLM_WONN_PYTHON=/opt/dlm-wonn-venv/bin/python
export DLM_WONN_EXPECTED_GPU_COUNT=4
bash scripts/verify_cloud_env.sh
```

验证脚本会检查固定 Python/依赖、clean checkout、4 张可见 GPU、原生 BF16、CUDA
`torch.compile` 和项目模块导入。系统镜像只有在这一步完整通过后才可保存。
锁定环境使用 PyTorch 2.6.0 的 CUDA 12.4 wheel，以匹配目标主机的 NVIDIA 550 驱动；不要改用
CUDA 13 wheel，也不要跳过驱动兼容性检查。

数据、encoder 和 tokenizer 必须按两份 YAML 固定的 revision 预下载到持久缓存。首次联网准备时可
设置 `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0`；正式运行默认离线，避免中途网络漂移。

固定缓存位置：

```bash
export HF_HOME=/root/shared-nvme/dlm-wonn/cache/huggingface
```

## 6. 唯一 smoke 与固定预算

只执行一次 2+2 并发 smoke：ELF 使用 GPU 0、1，WONN 使用 GPU 2、3；每卡 batch 12、global
batch 24、lr `5e-4`、warmup 3000。先运行到约 100 step 并保存 checkpoint，再从同一目录恢复到
约 200 step。记录两边稳定后的 seconds/step、samples/second、峰值显存，并验证 loss/梯度/参数
有限、DDP 正常退出、无 NCCL hang、checkpoint 可加载且恢复后的 step/metrics 连续。

不搜索其他 batch、学习率、scheduler 或 GPU 布局。正式预算固定为 20K；smoke 吞吐仅用于报告
预计墙钟时间，不再决定 50K/60K 分支。运行配置检查：

```bash
$DLM_WONN_PYTHON scripts/analyze_cloud_pair.py validate-configs \
  --elf-config src/configs/training_configs/train_de-en-ELF-B-cloud-60k.yml \
  --wonn-config src/configs/training_configs/train_de-en-WONN-L12K768T3-cloud-60k.yml \
  --effective-batch 24 --world-size 2 --target-steps 20000
```
两份配置必须使用相同的固定 `lr=0.0005`。

## 7. preflight 与正式启动

每次运行使用全新持久目录：

```bash
export DLM_WONN_PYTHON=/opt/dlm-wonn-venv/bin/python
export DLM_WONN_TARGET_STEPS=20000
commit_short="$(git rev-parse --short=7 HEAD)"
export DLM_WONN_PAIR_RUN_ROOT="/root/shared-nvme/dlm-wonn/runs/wmt14-pair-${commit_short}-20000"
export DLM_WONN_GPU_LAYOUT=2+2
export DLM_WONN_GLOBAL_BATCH_SIZE=24
export DLM_WONN_EVAL_SAMPLES=500
export DLM_WONN_ALL_GPUS=0,1,2,3
export DLM_WONN_ELF_GPUS=0,1
export DLM_WONN_WONN_GPUS=2,3
export HF_HOME=/root/shared-nvme/dlm-wonn/cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DLM_WONN_PREFLIGHT_ONLY=1
bash scripts/run_cloud_pair_pipeline.sh
```

目录名中的短 commit 和目标数字必须与当前 clean checkout 及 `DLM_WONN_TARGET_STEPS` 一致。
preflight 会验证 clean commit、正好四卡、GPU 分组、batch、固定学习率、
warmup、目标 step、checkpoint、评测契约和输出目录名。

preflight 通过后，在同一 clean commit 上取消该变量并在 `tmux` 中启动：

```bash
unset DLM_WONN_PREFLIGHT_ONLY
tmux new -s dlm-wonn-cloud-pair
bash scripts/run_cloud_pair_pipeline.sh
```

不要并行启动第二个相同 run root。pipeline 使用顶层锁和两个训练锁阻止重复进程。

## 8. 恢复、监控与失败处理

安全监控：

```bash
nvidia-smi
tail -F "$DLM_WONN_PAIR_RUN_ROOT/elf/pipeline.log"
tail -F "$DLM_WONN_PAIR_RUN_ROOT/wonn/pipeline.log"
```

自动监控建议约每 10 分钟检查一次 tmux、两个 optimizer step、四卡进程归属、显存、pipeline 状态
和日志尾部；发现进程退出、NaN/Inf、NCCL 或 OOM 时立即处理，不必等满 10 分钟。

实例重启后，只在确认没有存活训练进程时，用完全相同的 commit、环境变量和 run root 重新执行
pipeline。已完成的一侧会跳过，未完成的一侧从自身 checkpoint 恢复。以下情况必须新建 run root：

- 需要更换 commit、布局、world size、batch/LR、warmup、目标 step、checkpoint 或评测样本数；
- 现有 checkpoint 缺少 pair provenance；
- 输出目录来源不明确或曾被手工混合。

任一训练或评测失败都会阻止顶层完成标记。不要手工伪造 `training_complete.json`、
`evaluation_complete.json` 或 `pipeline_complete.json`。

## 9. 产物验收

```text
<run-root>/
├── provenance/
├── elf/checkpoint_{step}/
├── elf/evaluations/checkpoint_{step}/
├── wonn/checkpoint_{step}/
├── wonn/evaluations/checkpoint_{step}/
├── analysis/comparison.json
├── analysis/metrics_by_checkpoint.csv
├── analysis/comparison.md
├── analysis/quality_curves.svg
└── pipeline_complete.json
```

最终报告至少包含 BLEU、chrF++、TER、空输出率、唯一输出率、长度比、loss、samples seen、训练/
评测吞吐、训练耗时、sampler latency 和峰值评测显存。图中同时按 optimizer step 和 samples seen
展示 BLEU 与 chrF++。

将结果回传本地时复制整个 run root 到新的 Git 忽略目录，不覆盖 `outputs/phase5/` 中的历史归档。

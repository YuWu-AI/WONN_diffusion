# DLM-WONN

DLM-WONN 研究能否用 Winfree Oscillatory Neural Network（WONN）替换
[Embedded Language Flows（ELF）](https://arxiv.org/abs/2605.10938) 的 Transformer denoiser，
同时保持 encoder、Flow Matching、self-conditioning、conditioning mask、sampler 和 shared
decoder 不变。

当前唯一执行主线是 Phase 1 的 WMT14 云端工程闭环：在一台 4-GPU 主机上，从同一 clean commit
分别训练 ELF-B 和 WONN-L12/K768/T3 到 60K optimizer steps，再对 9 个 checkpoint 做统一评测并
生成一张复合折线图。历史 50K/90K/130K 实验只作为产物和 Git 历史证据，不是当前启动入口。

## 快速导航

| 入口 | 用途 |
| --- | --- |
| [`docs/README.md`](docs/README.md) | 文档状态与阅读顺序 |
| [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) | 已确认的架构边界和技术基线 |
| [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) | Phase 1–3 研究顺序与验收标准 |
| [`docs/CLOUD_PHASE1_RUNBOOK.md`](docs/CLOUD_PHASE1_RUNBOOK.md) | 4-GPU、60K 云端运行手册 |
| [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) | 代码、脚本、配置和产物索引 |

## 目录

```text
.
├── src/                         # ELF/WONN 模型、训练、采样和评测
│   ├── modules/                 # 正式模型实现；WONN 只在这里实现
│   └── configs/                 # 官方基线与独立实验配置
├── scripts/                     # 当前入口、验证工具和历史实验入口
├── tests/                       # 公共契约、模型和流水线测试
├── docs/
│   ├── architecture/            # 现行算法与公式说明
│   └── CLOUD_PHASE1_RUNBOOK.md  # 当前云端运行口径
├── outputs/                     # Git 忽略的 checkpoint、日志和评测产物
├── papers/                      # 论文材料
├── references/                  # 只读上游参考实现
└── PROJECT_HANDOFF.md           # 稳定技术基线
```

## 环境与验证

```bash
uv venv --python 3.10 .venv
uv pip sync --python .venv/bin/python requirements-lock.txt
.venv/bin/python -m unittest discover -v
```

本机环境快照见 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)；官方 ELF checkpoint、训练和评测
命令见 [`docs/ELF_UPSTREAM_README.md`](docs/ELF_UPSTREAM_README.md)。CUDA 严格验证入口为：

```bash
.venv/bin/python scripts/verify_phase2.py
.venv/bin/python scripts/verify_phase3.py
.venv/bin/python scripts/verify_phase4.py
```

## 当前云端入口

正式运行必须来自 clean commit，并将输出放在持久卷的新目录：

```bash
export DLM_WONN_PYTHON=/opt/dlm-wonn-venv/bin/python
export DLM_WONN_PAIR_RUN_ROOT=/persistent/dlm-wonn/runs/wmt14-pair-20260902-a
export DLM_WONN_GPU_LAYOUT=2+2
export DLM_WONN_GLOBAL_BATCH_SIZE=48
bash scripts/run_cloud_pair_pipeline.sh
```

`48` 只是待云端 profile 复核的初始候选，不应在未测显存和吞吐时视为最终 batch。完整 gate、
`4-serial` 备选布局、恢复约束和产物定义见云端运行手册。

## 不变量

- `main` 以 ELF 官方 PyTorch `pytorch_elf` 分支为执行基线；`elf-upstream` 只用于同步。
- 官方 ELF YAML 保留不动；WONN 和实验配置独立维护。
- 运行时代码不得从 `references/` 导入。
- checkpoint、日志、缓存、样本和数据不得提交。
- “测试通过”“流水线完整”“模型效果更好”是不同结论，必须分别验证。

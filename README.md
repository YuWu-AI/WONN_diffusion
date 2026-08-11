# DLM-WONN

本项目研究使用 Winfree Oscillatory Neural Network（WONN）完整替换
[Embedded Language Flows（ELF）](https://arxiv.org/abs/2605.10938) 的 Transformer denoiser，
同时保持 ELF 的 contextual embedding、Flow Matching、self-conditioning、conditioning mask、
sampler 和 shared decoder 不变。

旧开发计划的 Phase 1–4 已完成：官方 ELF-B 基线已复现，backbone 契约已固定，WONN 已完整
接入训练与采样主链路，并通过固定 batch 可学性和条件敏感性验收。详细技术基线见
[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)。

项目现已重新编号：

1. Phase 1：整理原 Phase 5 训练基础设施，并在云端从头重跑端到端工程实验；
2. Phase 2：OpenWebText 无条件生成主实验；
3. Phase 3：XSum 条件摘要实验。

当前代码进度、任务差异和验收 gate 见 [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)。
本地旧 Phase 5 实验数据不作为新计划结论，后续数据统一在云端重新产生。

## 项目结构

```text
.
├── src/                         # ELF 训练、模型、采样和评测代码
│   ├── modules/                 # ELF/WONN backbone 与公共 factory
│   └── configs/                 # 官方基线配置与独立 WONN 配置
├── scripts/                     # 启动、评测和严格验收脚本
├── tests/                       # 模型、数据、采样和训练契约测试
├── docs/
│   ├── RESEARCH_PLAN.md         # 新 Phase 1–3 与云端执行计划
│   ├── ENVIRONMENT.md           # 已验证本机环境和依赖
│   └── ELF_UPSTREAM_README.md   # ELF 官方 PyTorch 使用说明
├── papers/                      # ELF/WONN 论文
├── references/                  # 只读上游参考实现
├── PROJECT_HANDOFF.md           # 项目总览、架构边界和已完成基础
├── CLAUDE.md                    # 项目级 AI 协作规则（AGENTS.md 同源）
├── requirements.txt
└── requirements-lock.txt
```

正式 WONN 实现位于 `src/modules/wonn_layers.py` 和 `src/modules/wonn_model.py`；运行时代码不得
从 `references/` 导入。官方 ELF YAML 保留不动，WONN 与消融实验使用独立配置。

## 环境

```bash
uv venv --python 3.10 .venv
uv pip sync --python .venv/bin/python requirements-lock.txt
```

已验证环境见 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)，官方 ELF checkpoint、训练和评测
命令见 [`docs/ELF_UPSTREAM_README.md`](docs/ELF_UPSTREAM_README.md)。

## 常用入口

```bash
# CPU 回归检查
.venv/bin/python -m unittest discover -v

# CUDA 严格契约与可学性验收
.venv/bin/python scripts/verify_phase2.py
.venv/bin/python scripts/verify_phase3.py
.venv/bin/python scripts/verify_phase4.py

# 官方单卡训练入口
bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
```

训练产物统一写入 `outputs/` 等 Git 忽略目录，不提交 checkpoint、日志、生成样本、缓存或本地
数据。云端实验必须从 clean commit 启动并使用持久卷保存产物。

## 上游关系

`main` 以 ELF 官方 `pytorch_elf` 分支为执行基线；`elf-upstream` 只用于同步官方代码：

```bash
git fetch elf-upstream
```

WONN 参考实现以 submodule 固定在 `references/WONN/`。当前派生对象 `af3f468` 尚未发布到
`.gitmodules` 指向的官方远程；对外共享全新 clone 前，需要先推送到可访问 fork 并更新 URL。

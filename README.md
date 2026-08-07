# DLM-WONN

本项目研究使用 Winfree Oscillatory Neural Network（WONN）完整替换
[Embedded Language Flows（ELF）](https://arxiv.org/abs/2605.10938) 的 Transformer
backbone，同时保持 ELF 的连续表征空间、Flow Matching 目标、条件接口、采样器和共享
decoder 不变。

当前状态（2026-08-07）：Phase 4 已完成。工程基线固定在 ELF 官方 PyTorch `b29d883`；
Phase 1 的 WMT14 De→En ELF-B validation BLEU 为26.55，第一版30.5M参数的 WONN-ELF 已接入
训练器和采样器，并通过固定 WMT14 batch 的 MSE、CE、混合 objective 过拟合和 synthetic
conditional sensitivity 验收。下一步是在批准预算后进入 Phase 5 WMT14 端到端训练。实测结果见
[docs/PHASE1_BASELINE.md](docs/PHASE1_BASELINE.md) 和
[docs/PHASE2_CONTRACTS.md](docs/PHASE2_CONTRACTS.md)、
[docs/PHASE3_WONN_ELF.md](docs/PHASE3_WONN_ELF.md)、
[docs/PHASE4_LEARNABILITY.md](docs/PHASE4_LEARNABILITY.md)，研究设计见
[PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。

## 项目结构

```text
.
├── src/                         # ELF PyTorch 训练、模型、采样和评测代码
├── scripts/                     # 官方启动、PPL 评测与 Phase 2/3/4 严格验收脚本
├── docs/
│   ├── ELF_UPSTREAM_README.md   # 官方 PyTorch ELF 使用说明
│   ├── ENVIRONMENT.md           # 已验证硬件、依赖和复现命令
│   ├── PHASE1_BASELINE.md       # ELF checkpoint、validation 和训练 smoke 结果
│   ├── PHASE2_CONTRACTS.md      # ELF backbone 替换契约及测试结果
│   ├── PHASE3_WONN_ELF.md       # WONN-ELF 实现、诊断和 smoke 结果
│   └── PHASE4_LEARNABILITY.md   # 固定 batch 可学性和条件敏感性结果
├── papers/
│   ├── ELF.pdf
│   └── WONN.pdf
├── references/
│   ├── README.md
│   └── WONN/                    # 固定派生 commit 的 WONN submodule（见下方限制）
├── PROJECT_HANDOFF.md           # ELF-WONN 架构与开发计划
├── CLAUDE.md                    # 项目级 AI 协作规则（AGENTS.md 同源）
├── requirements.txt             # ELF PyTorch 依赖
├── requirements-lock.txt        # 2026-08-06 已验证的精确依赖快照
└── LICENSE                      # ELF 的 MIT License
```

`src/` 的训练、Flow Matching、conditioning、sampler 和 decoder 仍以 ELF PyTorch 为基线；
`src/modules/wonn_layers.py` 和 `wonn_model.py` 提供独立 WONN backbone，运行时代码不从
`references/WONN/` 导入。训练产物统一写到 `outputs/`，该目录不进入版本控制。

## ELF 代码导览

- `src/train.py`：加载配置、数据、冻结 T5 encoder，创建模型并管理完整训练循环。
- `src/train_step.py`：构造 flow 噪声，执行 denoising/decoding 两种训练分支并计算损失。
- `src/eval.py`：加载 checkpoint 并启动评测。
- `src/generation.py`：组织无条件生成、翻译和摘要评测。
- `src/modules/model.py`：当前 ELF Transformer backbone、flow 输出头和共享 token decoder。
- `src/modules/layers.py`：attention、RoPE、RMSNorm、SwiGLU 和投影层。
- `src/modules/wonn_layers.py`：Winfree coupling、phase recurrence 和 frequency transition。
- `src/modules/wonn_model.py`：ELF-compatible WONN backbone、readout 和动力学诊断。
- `src/modules/model_factory.py`：ELF/WONN 公共模型构造入口。
- `src/modules/t5_encoder.py`：冻结的 T5 continuous embedding encoder。
- `src/utils/data_utils.py`：数据加载、padding、source/target 拼接与 mask。
- `src/utils/sampling_utils.py`：加噪、self-conditioning、CFG、ODE/SDE 更新。
- `src/utils/generation_utils.py`：多步采样及多 GPU 生成辅助逻辑。
- `src/configs/`：训练任务配置与采样配置。

本机环境见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)，Phase 1 实测结果见
[docs/PHASE1_BASELINE.md](docs/PHASE1_BASELINE.md)。ELF 官方命令保存在
[docs/ELF_UPSTREAM_README.md](docs/ELF_UPSTREAM_README.md)。

## 环境复现

```bash
uv venv --python 3.10 .venv
uv pip sync --python .venv/bin/python requirements-lock.txt
```

`.venv` 只属于本机环境，不进入版本控制。若要重新解析允许范围内的最新依赖，使用
`requirements.txt`；需要复现 Phase 0 环境时使用 `requirements-lock.txt`。

## 下一阶段

1. 保留 Phase 4 固定任务作为后续模型改动的可学性 gate。
2. 为 ELF/WONN 建立实际 wall-clock、显存和 FLOPs 对照，确定 compute-matched 配置。
3. 确认训练预算、随机种子和调参规则后，再启动 Phase 5 WMT14 端到端比较。

## 上游关系

官方 ELF 仓库配置为只读语义的远程 `elf-upstream`：

```bash
git fetch elf-upstream
```

本项目 `main` 基于官方 `pytorch_elf` 分支；官方 JAX 主分支保留在本地
`elf-jax-main`。创建本项目自己的远程仓库后，应将其命名为 `origin`。

WONN 通过 Git submodule 固定在派生快照 `af3f468`（上游基点 `62d7ac5`）。该派生提交尚未发布到
`.gitmodules` 指向的官方远程，因此全新 clone 暂时无法初始化此 submodule；在对外共享前需要将
提交推送到可访问的 fork 并更新 `.gitmodules`。发布问题解决后，新 clone 使用：

```bash
git submodule update --init --recursive
```

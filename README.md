# DLM-WONN

本项目研究使用 Winfree Oscillatory Neural Network（WONN）完整替换
[Embedded Language Flows（ELF）](https://arxiv.org/abs/2605.10938) 的 Transformer
backbone，同时保持 ELF 的连续表征空间、Flow Matching 目标、条件接口、采样器和共享
decoder 不变。

当前状态（2026-08-07）：Phase 1 已完成。工程基线固定在 ELF 官方 PyTorch `b29d883`；
官方 WMT14 De→En ELF-B checkpoint 的 3000 条 validation evaluation 得到 BLEU 26.55，单 batch
训练和 checkpoint save/load 已验证。下一步是 Phase 2 模型契约测试，尚未实现 WONN backbone。
详细结果见 [docs/PHASE1_BASELINE.md](docs/PHASE1_BASELINE.md)，研究设计见
[PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。

## 项目结构

```text
.
├── src/                         # ELF PyTorch 训练、模型、采样和评测代码
├── scripts/                     # 官方启动与 PPL 评测脚本
├── docs/
│   ├── ELF_UPSTREAM_README.md   # 官方 PyTorch ELF 使用说明
│   ├── ENVIRONMENT.md           # 已验证硬件、依赖和复现命令
│   └── PHASE1_BASELINE.md       # ELF checkpoint、validation 和训练 smoke 结果
├── papers/
│   ├── ELF.pdf
│   └── WONN.pdf
├── references/
│   ├── README.md
│   └── WONN/                    # 固定 commit 的官方 WONN submodule
├── PROJECT_HANDOFF.md           # ELF-WONN 架构与开发计划
├── CLAUDE.md                    # 项目级 AI 协作规则（AGENTS.md 同源）
├── requirements.txt             # ELF PyTorch 依赖
├── requirements-lock.txt        # 2026-08-06 已验证的精确依赖快照
└── LICENSE                      # ELF 的 MIT License
```

`src/` 当前仍是未修改的 ELF PyTorch 基线。后续 WONN 实现应新增在
`src/modules/`，而不是从 `references/WONN/` 直接导入。训练产物统一写到
`outputs/`，该目录不进入版本控制。

## ELF 代码导览

- `src/train.py`：加载配置、数据、冻结 T5 encoder，创建模型并管理完整训练循环。
- `src/train_step.py`：构造 flow 噪声，执行 denoising/decoding 两种训练分支并计算损失。
- `src/eval.py`：加载 checkpoint 并启动评测。
- `src/generation.py`：组织无条件生成、翻译和摘要评测。
- `src/modules/model.py`：当前 ELF Transformer backbone、flow 输出头和共享 token decoder。
- `src/modules/layers.py`：attention、RoPE、RMSNorm、SwiGLU 和投影层。
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

## 推荐的开发顺序

1. 为现有 ELF 写模型契约测试。
2. 新增 `wonn_layers.py` 和 `wonn_model.py`，保持原模型输入输出契约。
3. 先完成单元测试和单 batch 过拟合，再开始端到端比较。

## 上游关系

官方 ELF 仓库配置为只读语义的远程 `elf-upstream`：

```bash
git fetch elf-upstream
```

本项目 `main` 基于官方 `pytorch_elf` 分支；官方 JAX 主分支保留在本地
`elf-jax-main`。创建本项目自己的远程仓库后，应将其命名为 `origin`。

WONN 通过 Git submodule 固定。新克隆项目后执行：

```bash
git submodule update --init --recursive
```

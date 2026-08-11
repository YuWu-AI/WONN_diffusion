# ELF-WONN 新阶段研究计划

> 计划重置日期：2026-08-11
> Phase 1–4 的基础实现与验收已经完成，统一沉淀在
> [`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md)。本文从新的 Phase 1 重新编号。

## 1. 新旧阶段映射

| 新计划 | 对应旧进度 | 定位 |
| --- | --- | --- |
| Phase 1 | 原 Phase 5 | 整理训练基础设施，并在云端从头重跑端到端工程实验 |
| Phase 2 | 原拟定 Phase 6 | OpenWebText 无条件生成主实验 |
| Phase 3 | 原拟定 Phase 7 | XSum 条件摘要实验 |

WMT14 翻译不再承担主要研究结论。新 Phase 1 只用它验证代码、云端环境、训练恢复和统一评测
链路；所有实验数据将在云端重新产生，本地旧 pilot 指标不纳入新计划的结果判断。

## 2. Phase 1 当前代码进度

原 Phase 5 的实现位于独立分支和 worktree：

```text
branch:   phase5-wmt14
commit:   e33842c
worktree: .worktrees/phase5-wmt14
```

该分支当前是干净的，已经实现：

- 按 optimizer step 精确限制训练预算；
- 在指定 step 保存 checkpoint；
- resume 后恢复 step、optimizer、EMA、scheduler 和 RNG；
- reconcile `train_metrics.jsonl`，避免恢复训练后指标重复或越界；
- 从兼容 checkpoint 只加载 model/EMA 的 warm start；
- 区分纯训练完成标记与评测完成标记；
- checkpoint 结构、finite values、生成工件和指标的汇总校验；
- WMT14 ELF/WONN 独立实验 YAML、profiling、后台流水线和 13 项训练工程测试（测试文件当前仍名为
  `tests/test_phase5_training.py`）；
- 单机单卡与单机多卡 `torchrun` 启动入口。

这些代码尚未成为新的可执行主线，原因是：

1. 通用训练能力仍在 `phase5-wmt14`，尚未与 `main` 整合；
2. `main` 工作树当前包含尚未提交的“最后一层移除无效 frequency transition”修正，与分支内
   WONN 文件有重叠，需要人工整合而不是直接覆盖；
3. Phase 1 shell 脚本仍含本机绝对 Python/include 路径、强制 Hugging Face offline 和
   `notify-send`，不能原样部署云端；
4. WMT14 专用 YAML 和 pipeline 应与任务无关的训练基础设施分开；
5. 云端运行必须固定 clean commit，当前 dirty `main` 不能直接作为实验版本。

因此当前准确状态是：**训练基础设施已经在隔离分支实现并通过代码级测试，但仍需整合和云端化，
之后才从头产生新的实验结果。**

## 3. Phase 1：云端工程基线

### 目标

得到一个可追溯、可恢复、可重复的 ELF/WONN 云端训练与评测入口，为 OWT 和 XSum 提供公共
基础设施。WMT14 只作为低成本端到端验证任务。

### 代码工作

1. 以 `main` 的正式 WONN 实现为架构真身，逐项引入 `phase5-wmt14` 的通用训练能力。
2. 保留最后一层不创建 frequency transition 的当前修正，并补兼容 checkpoint 的明确策略。
3. 将脚本改为相对项目路径或可配置环境变量；移除本机 include、桌面通知和强制 offline 假设。
4. 保留官方 ELF YAML；新实验使用独立 YAML，不覆盖 upstream 配置。
5. 把 WMT14 专用 pipeline 与公共 train/eval/checkpoint 工具分开。
6. 为云端运行记录 commit、配置、依赖、GPU、数据 revision、随机种子和输出目录。

### 执行顺序

1. CPU unit tests。
2. 单卡 BF16 forward/backward 和单 batch train smoke。
3. checkpoint save、故意中断、resume、metrics 连续性检查。
4. 单机多 GPU 100-step smoke，确认无 rank hang。
5. 云端从随机初始化分别运行 ELF 与 WONN，不使用本地 warm-start 结果。
6. 使用同一数据、seed、训练 token/step 预算和采样配置评测。
7. 独立检查 checkpoint、日志、生成样本和 completion manifest。

### 验收门槛

- 实验来自 clean commit，配置和依赖可追溯；
- ELF 与 WONN 都能从头训练、保存、恢复并完成评测；
- resume 前后 optimizer step 和 metrics 单调且无重复；
- 多 GPU 启动和退出正常；
- 产物写入持久卷，训练结束后可独立加载 checkpoint；
- 报告资源与工程结果，但不因 WMT14 质量高低改变后续 OWT 方向。

Phase 1 完成后冻结 WMT14，不再追加翻译任务预算。

## 4. Phase 2：OpenWebText 无条件生成

### 研究问题

在相同训练和采样预算下，WONN 是否能改善 ELF 的生成质量—多样性折中。主指标是
Gen. PPL 与 unigram entropy 的联合曲线，而不是单独优化其中一个数字。

### 必要代码改动

1. 从官方 `train_owt_ELF-B.yml` 派生独立 `train_owt_ELF-WONN-B.yml`，只改变 backbone、
   WONN 超参数、资源 batch/accumulation 和输出目录。
2. 将 coupling 默认路径改为 SDPA，避免显式物化 `[B,H,L,L]`；
   `forward_with_diagnostics()` 才计算完整 attention weights。
3. 增加 1024-token BF16 forward/backward、gradient checkpointing、DDP、resume 和生成测试。
4. 固定 GPT-2 Large PPL evaluator、tokenizer、数据 revision 和采样 sweep。

训练器、Flow Matching、self-conditioning、sampler 和 shared decoder 不需要为 OWT 复制或改写。
现有 `eval_data_path=null` 路径已经表达无条件生成。

### 执行顺序与 gate

1. 复评官方 ELF-B OWT checkpoint，验证 PPL/entropy evaluator。
2. profile 1024-token WONN 的 per-device micro-batch，再用 gradient accumulation 达到目标
   effective batch：

   ```text
   effective batch = per-device batch × world size × grad_accum_steps
   ```

3. 先跑 2k/5k/10k optimizer-step ELF/WONN pilot。
4. 对相同 sampling steps 和 guidance 设置报告 PPL–entropy 曲线、空输出率、吞吐、显存、
   wall-clock 和 sampler latency。
5. pilot 有可信信号后，再决定完整 compute-matched 和 parameter-matched 训练。

若 PPL 改善只来自 entropy collapse，或二次 attention 导致不可接受的长序列成本，则停在 pilot，
先修正结构或实验设计。

## 5. Phase 3：XSum 摘要

### 研究问题

在长条件上下文中，WONN 是否能正确使用 source，并在同等预算下获得有竞争力的 ROUGE 和生成
稳定性。

### 必要代码改动

1. 复用 Phase 2 已验证的长序列 SDPA 路径。
2. 派生独立 `train_xsum_ELF-WONN-B.yml`，保留官方 ELF 配置。
3. 增加 1088-token source+target 拼接、padding、source restore、target-only loss 和 label-drop
   契约测试。
4. 统一 ROUGE-1/2/L 汇总；正式结论阶段再加入实体和数字一致性检查。

### 执行顺序与 gate

1. 复评官方 ELF-B XSum checkpoint，确认数据和 evaluator。
2. 通过 1088-token 单卡、DDP、resume 和 conditional mask smoke。
3. 跑 ELF/WONN 5k/10k pilot。
4. 根据 ROUGE、事实一致性、吞吐和显存决定是否完整训练。

Phase 3 不早于 Phase 2 的长序列 gate；否则无法区分 backbone 长序列问题与 conditional masking
问题。

## 6. 不同任务的改动边界

| 范围 | Phase 1 WMT14 | Phase 2 OWT | Phase 3 XSum |
| --- | --- | --- | --- |
| 目的 | 云端工程闭环 | 无条件生成主实验 | 条件摘要验证 |
| 序列长度 | 64 source + 64 target | 1024 tokens | 1024 source + 64 target |
| conditioning | 保留 source mask | 无 source condition | 保留 source mask |
| loss mask | target only | 全文本有效 token | target only |
| 新模型代码 | 不新增结构 | SDPA 长序列快路径 | 复用 SDPA |
| 新配置 | WMT14 独立 YAML | OWT WONN YAML | XSum WONN YAML |
| 主指标 | 工件完整性；BLEU 辅助 | Gen. PPL + entropy | ROUGE-1/2/L |

## 7. 公共实验纪律

- baseline 与 WONN 使用相同数据 revision、seed、有效 batch、训练 token/step 和采样配置。
- 同时报告模型参数、wall-clock、峰值显存、吞吐和 sampler latency。
- checkpoint、日志、样本、缓存和数据只写入 Git 忽略目录或云端持久卷。
- 每个阶段先 unit/smoke，再 pilot，最后才允许扩大预算。
- “工件完整”“代码可运行”“模型质量更好”是三个独立结论，不得互相替代。
- 云端命令必须从 clean commit 启动；禁止把 dirty worktree 临时复制为实验版本。

## 8. 下一步唯一主线

1. 整合 `phase5-wmt14` 的通用训练代码与 `main` 的 WONN frequency 修正。
2. 完成脚本云端可移植化和 Phase 1 的本地 smoke。
3. 固定 clean commit，在云端从头跑新 Phase 1。
4. Phase 1 工程验收后立即进入 Phase 2 OWT，不继续扩展 WMT14。

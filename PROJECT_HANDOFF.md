# ELF-WONN 项目总览与技术基线

> 状态基线：2026-09-01
> 本文只记录已经确定的项目事实、架构边界和已完成基础；后续阶段与执行顺序见
> [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)。

## 1. 研究目标

本项目研究能否用 Winfree Oscillatory Neural Network（WONN）完整替换 Embedded Language
Flows（ELF）的 Transformer denoiser backbone，并在保持 ELF 生成框架不变的条件下，改善生成
质量、参数效率或同等计算预算下的效果。

ELF 仍在预训练语言模型的 contextual embedding space 中执行 Flow Matching。WONN 的相位
`theta` 和频率 `omega` 仅是一次 denoiser 前向调用内部的 hidden states，不替代 ELF latent，
也不跨采样 step 延续。

核心比较对象是：

```text
相同数据、encoder、Flow Matching、conditioning、sampler、decoder 和训练预算
                         │
                 仅替换 denoiser backbone
                         │
              ELF Transformer vs. ELF-WONN
```

## 2. 不可破坏的 ELF 边界

- 保留 frozen T5 contextual encoder 和 tokenizer。
- 保留 linear flow path、x-prediction 参数化和 velocity supervision。
- 保留 self-conditioning、time/CFG/mode control tokens。
- 保留 conditional source prefix、padding mask、target-only loss 和 sampler restore。
- 保留 denoise/decode shared backbone 以及最终 unembedding decoder。
- 保持 `src/modules/model.py` 的输入输出 shape、dtype、device 和 mode 契约。
- 每次 sampler 调用都从当前 `z_t` 重新确定性初始化 WONN hidden state。

除非实验明确要研究上述部件，否则后续任务只能替换 backbone，不应同时改变生成框架。

## 3. 已完成基础

原开发计划的 Phase 1–4 已全部执行成功。它们现在作为项目已知事实，不再作为新计划中的活动
阶段，也不再分别维护阶段文档。

### 3.1 ELF 官方基线已复现

- 项目以 ELF 官方 `pytorch_elf` 分支为执行基线，固定上游基点为 `b29d883`。
- 官方 ELF-B WMT14 De→En checkpoint 能完成加载、生成和 validation evaluation。
- 3000 条 validation 样本的 BLEU 为 `26.5521`，与官方 PyTorch validation 参考约 `26.7`
  一致。
- 单 batch 训练已覆盖 loss、backward、optimizer、EMA、checkpoint save/load 和恢复后继续训练。
- 正式 ELF-B（T5 vocab 32100）参数量为 `104,579,940`。

以上事实证明本项目的 ELF 数据、训练、采样和评测主链路可用；它们不是 WONN 效果结论。

### 3.2 Backbone 契约已固定

替换模型必须保持以下行为：

- 输入支持 `(B,S,C)` 与 self-conditioned `(B,S,2C)`；
- continuous output 为 `(B,S,C)`，decode logits 为 `(B,S,V)`；
- denoise/decode mode、固定 `max_length`、RoPE 和 control-token 语义不变；
- padding mask、source condition restore 和 target-only supervision 不变；
- eval forward 与 sampler rollout 在固定输入下确定；
- CPU/CUDA、BF16、backward、gradient checkpointing 和 sampler shape 可验证。

严格契约验收已在 CUDA 环境完成，后续 WONN 也通过了同一组检查。测试入口位于
`tests/`，严格脚本位于 `scripts/verify_phase2.py`、`scripts/verify_phase3.py` 和
`scripts/verify_phase4.py`。

### 3.3 WONN backbone 已完整接入

正式实现位于：

- `src/modules/wonn_layers.py`：phase wrapping、sensitivity/influence、attentive Winfree
  coupling、recurrent phase update 和 layer 间 frequency transition；
- `src/modules/wonn_model.py`：ELF-compatible conditioning、确定性 phase/frequency adapter、
  WONN backbone、phase readout 和 shared decoder；
- `src/modules/model_factory.py`：ELF/WONN 公共构造入口；
- `src/configs/training_configs/`：官方 ELF YAML 与独立 WONN YAML。

当前正式 `ELF-WONN-B` 为 384 oscillators、6 layers、每层 2 个 inner steps、12 个 coupling
heads。每个 attention head 的维度固定为 `384 / 12 = 32`。独立 `OmegaTransition` 只存在于
相邻层之间，最后一层后不创建无输出作用的参数；当前参数量为 `26,252,399`。

每层先用逐振子 S/I MLP 计算 sensitivity 与 influence，再由 influence 经过完整
`W_qkv -> attention -> head merge -> W_o -> RMSNorm -> ReLU` 产生 coupling field。
普通路径使用 SDPA，只有 diagnostics 显式计算 attention weights；field 不含 influence residual、
phase residual 或 Transformer FFN。运行时代码不从只读的 `references/WONN/` 导入。

### 3.4 最小可学性已验证

旧交互架构曾在固定真实 T5/WMT14 batch 上完成 denoising、decoding 和 mixed objective 过拟合；
该结果不能转移为新交互架构的质量证据。新架构已通过完整 coupling/transition 单元测试、正式
`384x6x3` BF16 CUDA forward/backward，以及真实 WMT14 `batch=2` mixed 单步 smoke；S/I、
QKV/O、theta embedding、omega FFN 和 alpha 均有直接梯度或参数更新检查。

在 target latent 完全相同、仅 source 不同的 synthetic conditional task 中，交换 source 会使
target prediction 随之翻转，说明模型能够使用条件信息。

旧 synthetic source-swap 结果只排除了旧架构“完全忽略 source”的实现失败。新架构尚未完成
短程 overfit、source-swap 复验或翻译质量实验；单 batch 有限梯度不能证明泛化能力或优于 ELF。

## 4. 端到端数据流

训练路径：

```text
tokens
  -> frozen T5 encoder
  -> clean contextual embeddings x
  -> ELF corruption and self-conditioning
  -> current flow state z_t
  -> ELF Transformer or WONN shared backbone
  -> predicted clean embeddings x_hat
  -> denoising MSE or decoding CE
```

采样路径：

```text
Gaussian noise z_0
  -> repeated ELF ODE/SDE steps
  -> each step calls a fresh deterministic backbone
  -> final decode-mode call at t=1
  -> unembedding
  -> discrete tokens
```

conditional task 的 source 和 target 在同一序列中。source positions 保持 clean，loss 只监督
target，采样时由现有 `restore_cond()` 固定 source。

## 5. WONN 内部设计

对每个 token，WONN 使用：

\[
\theta,\omega\in\mathbb{R}^{K},\qquad
P(\theta)=[\sin\theta,\cos\theta].
\]

输入通过确定性 projection 初始化 `theta` 和 `omega`。每层内部重复 Winfree dynamics：

\[
\theta^{l,s+1}=\operatorname{wrap}
\left(\theta^{l,s}+\gamma_l\left[\omega^l+S(\theta^{l,s})\odot m^{l,s}\right]\right),
\]

其中完整 attentive coupling 先从 $I(\theta)$ 生成 Q/K/V，再合并 heads 并通过输出投影：

\[
F=\operatorname{ReLU}\left(\operatorname{RMSNorm}
\left(W_o\operatorname{MergeHeads}
\left[\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{K/H}}\right)V\right]
\right)\right).
\]

`theta` 在 layer 间原样传递；`omega` 只在相邻 layer 边界由
`RMSNorm([ThetaEmbedding(theta), omega]) -> K-width FFN` 做有界 residual update。最终输出只从
phase features 读取，以避免绕过 oscillator dynamics。

这里必须区分三种时间：ELF flow time `t`、WONN depth `l` 和 layer 内 recurrent step `s`。

## 6. 当前已知约束

- 普通 coupling 已使用 SDPA；diagnostics 仍会显式物化 `[B,H,L,L]` attention weights，因此
  长序列诊断必须控制 batch 和采样范围。
- 26.25M 参数少于 ELF-B 的 105M，但 recurrent coupling 的真实 FLOPs、吞吐和采样开销必须实测；
  不能仅凭参数量宣称效率优势。
- `references/WONN` 当前固定派生提交 `af3f468`，该对象尚未发布到 `.gitmodules` 指向的官方
  远程；对外提供全新 clone 前需要推送到可访问 fork 并更新 submodule URL。
- checkpoint、日志、缓存、生成样本和本地数据必须留在 Git 忽略目录。

## 7. 项目入口

| 文件或目录 | 用途 |
| --- | --- |
| `README.md` | 快速开始与代码导航 |
| `docs/README.md` | 文档状态、分类与阅读顺序 |
| `docs/RESEARCH_PLAN.md` | 新 Phase 1–3、当前代码进度和云端执行门槛 |
| `docs/PROJECT_STRUCTURE.md` | 代码分层、Phase 5 脚本/配置和本地数据索引 |
| `docs/CLOUD_PHASE1_RUNBOOK.md` | 当前 4-GPU、60K 云端配对运行口径 |
| `docs/ENVIRONMENT.md` | 已验证本机环境与依赖快照 |
| `docs/ELF_UPSTREAM_README.md` | ELF 官方 PyTorch 命令与参考指标 |
| `src/modules/` | ELF 与 WONN 正式模型实现 |
| `tests/`、`scripts/verify_phase*.py` | 契约和回归验收 |
| `references/` | 只读上游参考实现 |

当前 `main` 保留 optimizer-step 训练预算、checkpoint/resume、独立评测和产物校验等公共能力，
并以 4-GPU、60K 配对流水线作为 Phase 1 的唯一 WMT14 入口。旧 20K/50K/90K/130K、机制消融和
本地后台脚本已从当前目录移除，可从 Git 历史追溯；旧 Phase 5 产物仍只是历史诊断数据，不自动
升级为新研究计划的正式结论。

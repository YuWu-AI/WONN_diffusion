# ELF-WONN：架构设计与开发计划

> 状态快照：2026-08-07
>
> 项目阶段：Phase 4 已完成；固定 batch 可学性和 synthetic conditional sensitivity 已通过
>
> 目标读者：接手实现的 coworker、在新对话中继续工作的 AI agent

## 1. 五分钟摘要

本项目研究能否用 Winfree Oscillatory Neural Network（WONN）完整替换 Embedded Language Flows（ELF）中的 Diffusion Transformer（DiT）backbone，从而得到质量更好、参数效率更高，或在相同计算量下更有效的连续语言 flow denoiser。

核心边界已经确定：

- 不把 ELF 的 continuous latent space 改成相位空间；ELF 仍在 pretrained contextual embedding space 中做 Flow Matching。
- WONN 的相位 \(\theta\) 和频率 \(\omega\) 只是一次 denoiser 前向调用内部的 hidden states。
- 完整替换 ELF 的 `net(z, t, mode)`；不保留 Transformer block 或标准 attention value residual。
- 保留 ELF 的 encoder、flow path、self-conditioning、in-context control tokens、source prefix、shared denoising/decoding、sampler 和 unembedding。
- 第一个端到端任务优先候选为 WMT14 De→En；验证成立后再做 OpenWebText（OWT）主实验，最后用 XSum 做长上下文压力测试。任务选择和完整训练预算仍待项目负责人确认。

建议下一位接手者准备 Phase 5：

1. 先确定 compute-matched 配置、完整训练预算、随机种子和公平调参规则。
2. 获得批准后再启动 WMT14 ELF/WONN 端到端训练；Phase 4 不能替代泛化评测。

## 2. 研究问题

### 2.1 主假设

ELF 用 Transformer 参数化从 noisy contextual embedding 到 clean contextual embedding 的向量场。我们的主假设是：

> 带有 state-dependent attentive coupling、周期相位状态和快慢双状态演化的 Winfree dynamics，能够成为比标准 DiT 更有效的语言 flow denoiser。

ELF denoiser 的外部函数签名保持不变：

\[
D_\phi(z_t,t,c,m)\rightarrow \hat{x}_0,
\]

其中 \(z_t\) 是当前 continuous flow state，\(t\) 是 ELF flow time，\(c\) 包括 self-conditioning 和条件序列，\(m\) 是 denoise/decode mode。内部实现从 Transformer 改为 WONN。

### 2.2 不研究什么

第一阶段明确不做以下改动：

- 不让语言 token 或 ELF latent 永久生活在 \(\theta\) 空间。
- 不修改 ELF 的 rectified-flow interpolant、x-prediction 目标或 ODE/SDE sampler。
- 不同时引入新的 encoder、tokenizer、decoder 或 conditioning 方案。
- 不跨 ELF sampling step 传递 WONN hidden state。
- 不先做 progressive distillation、few-step distillation 或新的 noise schedule。

这些边界是为了把结果尽量归因于 denoiser backbone。

## 3. 依据与基线

### 3.1 WONN

本地论文：[WONN.pdf](./papers/WONN.pdf)

官方参考实现的上游基点为 `62d7ac52dee8b864cb77faac019a3d7ea1c2f7ae`；项目当前在
`references/WONN/` 固定派生快照 `af3f468d631d8b5a7f3730ccca5b7e686c458da1`。

原始 WONN 的离散动力学为：

\[
\theta_i^{(l,s+1)}
=
\operatorname{wrap}
\left[
\theta_i^{(l,s)}
+\gamma
\left(
\omega_i^{(l)}
+S_l(\theta_i^{(l,s)})
\sum_j c_{ij}I_l(\theta_j^{(l,s)})
\right)
\right].
\]

原模型具有以下特征：

- 输入主要编码进 frequency state \(\Omega_{\mathrm{init}}\)，phase 随机初始化。
- 每个 layer 内执行若干参数共享的 recurrent dynamics steps。
- \(\theta\) 是快速状态，\(\omega\) 在 layer 边界慢速更新。
- 原视觉模型使用 phase-aware convolution 做 layer transition。
- coupling 可以是 local convolution，也可以是 global attention；论文实验使用 attentive coupling。
- \(S\) 和 \(I\) 可以是固定三角函数或 learnable neural mappings。
- 输出只从最终 phase representation 读取。

### 3.2 ELF

- 本地论文：[ELF.pdf](./papers/ELF.pdf)
- 论文：[ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938)
- 官方代码：[lillian039/ELF](https://github.com/lillian039/ELF)
- 官方主实现为 JAX/TPU；仓库提供 `pytorch_elf` 分支。

与本项目直接相关的 ELF-B 配置：

- frozen pretrained T5-small contextual encoder；
- ELF embedding dimension 512，模型 hidden width 768；
- 12 layers、12 attention heads、约 105M parameters；
- continuous-time Flow Matching，\(t=0\) 为 noise，\(t=1\) 为 data；
- linear interpolant：

\[
z_t=t x+(1-t)\epsilon;
\]

- 采用 x-prediction，由 \(\hat{x}_0\) 转换为 velocity；
- denoising branch 使用 MSE，decoding branch 使用 token CE；
- 默认约 80% denoising updates、20% decoding updates；
- denoiser 和 final decoder 共享 backbone，通过 mode tokens 区分；
- final decoder 输出 contextual embedding，再经 learnable unembedding 得到 token logits；
- 使用 self-conditioning、in-context time/CFG/mode tokens、RoPE、RMSNorm 和 qk-norm；
- 官方 OWT ELF-B 使用 32-step SDE 时报告约 24.1 Gen. PPL 和 5.15 unigram entropy；这些数字只能作为原实现复现目标，不能直接当作本项目已复现结果。

## 4. 已确定的总体数据流

### 4.1 ELF 外部流程保持不变

训练时：

```text
tokens
  -> frozen T5 contextual encoder
  -> clean contextual embeddings x
  -> ELF corruption / self-conditioning
  -> current flow state z_t
  -> shared WONN backbone in denoise or decode mode
  -> predicted clean contextual embeddings x_hat
  -> MSE (denoise) or unembedding + CE (decode)
```

采样时：

```text
Gaussian noise z_0
  -> repeated ELF ODE/SDE steps
  -> each step calls a fresh deterministic WONN denoiser
  -> final decode-mode WONN call at t=1
  -> unembedding
  -> discrete target tokens
```

### 4.2 保留 ELF in-context conditioning

一次 network call 的完整序列为：

\[
H_{\mathrm{all}}
=
[C_t,C_w,C_m,H_{\mathrm{condition}},H_{\mathrm{target}}].
\]

第一版保留：

- 4 个 time tokens；
- 4 个 CFG tokens；
- 4 个 mode tokens；
- self-conditioning 的 channel concatenation 和 projection；
- conditional task 的 clean source prefix embeddings；
- ELF 原有 padding/condition attention mask；
- RoPE 和 qk-norm；
- denoising/decoding shared weights。

control、source 和 target positions 都转换为 oscillator states，并共同参与 attention coupling 和 phase evolution。
模型移除 control positions 后，必须像原 ELF 一样对完整 source+target positions 输出 continuous
prediction；现有 `loss_mask` 只监督 target positions，sampler 的 `restore_cond()` 负责固定
source positions。第一版不能为了内部 target-only readout 而改变模型外部 shape 或额外传入
source/target boundary。

第一版不增加显式 conditioning bias：

\[
B_t e_t+B_m e_m.
\]

原因是 time/mode/CFG 已通过 control tokens 注入。重复注入会同时改变 backbone 和 conditioning interface，也会使收益难以归因。

## 5. WONN hidden-state 设计

### 5.1 每个 token 是一组 oscillator

不能用单个 scalar phase 表示一个 token。对序列长度 \(N\)，定义：

\[
\theta,\omega\in\mathbb{R}^{N\times K}.
\]

第一版取：

\[
d_{\mathrm{model}}=768,\qquad K=384.
\]

因此：

\[
P(\theta)=[\sin\theta,\cos\theta]
\in\mathbb{R}^{N\times768}.
\]

这样 phase embedding 与 ELF-B hidden width 完全一致。每个 token 有384个 phase degrees of freedom 和384个 frequency degrees of freedom，总动态状态自由度与768维 Transformer hidden state 大致对齐。

### 5.2 确定性初始化

原始 WONN 的随机 phase initialization 不适用于 ELF ODE vector field。同一个 \((z_t,t)\) 必须产生确定性的输出；额外随机性只能由 ELF sampler 显式控制。

推荐 adapter：

\[
H=\operatorname{RMSNorm}(H_{\mathrm{all}}),
\]

\[
(a,b)=W_\theta H,
\qquad a,b\in\mathbb{R}^{N\times384},
\]

\[
\theta^{0,0}=\operatorname{atan2}(b,a),
\]

\[
\omega^0=W_\omega H
\in\mathbb{R}^{N\times384}.
\]

实现时应使用带 \(\epsilon\) 的 pair normalization，避免 \((a,b)\) 同时接近零造成不稳定：

\[
(\bar a,\bar b)
=
\frac{(a,b)}{\sqrt{a^2+b^2+\epsilon}}.
\]

普通 neural operations 使用周期安全的 \([\sin\theta,\cos\theta]\)，不直接把裸角度当作 Euclidean feature。

### 5.3 三种时间必须区分

- \(t\)：ELF flow/diffusion time；
- \(l\)：WONN layer/depth；
- \(s\)：一个 WONN layer 内的 recurrent dynamics step。

WONN hidden state只在一次 `net(z_t, t, mode)` 调用内部持续存在。下一次 ELF sampler 调用必须由新的 \(z_{t+\Delta t}\) 重新初始化：

\[
\theta_{\mathrm{init}}^{\,t+\Delta t}
=E_\theta(z_{t+\Delta t}).
\]

禁止把上一个 flow step 的最终 phase 直接作为下一个 flow step 的 hidden state；否则 denoiser 将依赖求解历史，不再是标准 Markov vector field，原 ELF training objective 和 solver 假设都需要重新推导。

## 6. WONN layer

### 6.1 Phase 在 layer 间连续传递

每个 layer 内，phase 由 Winfree dynamics 更新：

\[
\theta^{l,s+1}
=
\operatorname{wrap}
\left[
\theta^{l,s}
+\gamma_l\Delta\theta^{l,s}
\right].
\]

layer 边界不再使用原视觉 WONN 的 phase-aware convolution，也不重新从 ELF input 编码 phase：

\[
\boxed{
\theta^{l+1,0}=\theta^{l,T}
}
\]

也就是说，\(\theta\) 只能被 Winfree dynamics 改变，不被独立 layer-transition network 换坐标系。

### 6.2 Frequency 是慢速状态

\(\omega^l\) 在一个 layer 的全部 inner steps 内固定，在 layer 边界做 gated residual update：

\[
\omega^{l+1}
=
\omega^l
+\alpha_l
G_l\left(
[\sin\theta^{l,T},\cos\theta^{l,T},\omega^l]
\right).
\]

建议 \(\alpha_l\) 近零初始化。这样训练初期接近 fixed intrinsic frequency，随后模型可以学习跨层慢速改变动力学。

不采用两个极端：

- 整个 denoiser 完全固定 \(\omega\)：解释简单，但完整 backbone 的表达能力可能不足；
- 每个 inner step 都更新 \(\omega\)：会退化成复杂双状态 RNN，破坏快慢时间尺度和稳定性。

## 7. Attentive Winfree coupling

### 7.1 Attention 的职责

attention 只生成 state-dependent token coupling weights，不保留标准 Transformer 的 \(A V(P)\) residual update。

把384个 oscillator channels 分成12组：

\[
K=12\times32.
\]

对第 \(h\) 个 head：

\[
A_{ij,h}^{l,s}
=
\operatorname{softmax}_j
\left(
\frac{
\operatorname{RoPE}(Q_{i,h}^{l,s})
\operatorname{RoPE}(K_{j,h}^{l,s})^\top
}{\sqrt{64}}
+M_{ij}
\right).
\]

Winfree influence message：

\[
m_{i,h}^{l,s}
=
\sum_j A_{ij,h}^{l,s}
I_{l,h}(\theta_{j,h}^{l,s}).
\]

Phase velocity：

\[
\Delta\theta_{i,h}^{l,s}
=
\omega_{i,h}^l
+S_{l,h}(\theta_{i,h}^{l,s})
\odot m_{i,h}^{l,s}.
\]

概念上的职责划分是：

- \(A_{ij,h}\)：决定哪个 token 影响哪个 token；
- \(I(\theta_j)\)：发送端产生什么影响；
- \(S(\theta_i)\)：接收端对该影响有多敏感；
- \(\omega_i\)：该 oscillator 的内在漂移。

### 7.2 第一版的实现

第一版已采用以下实现：

- 12 coupling heads，每个 head 管理32个 oscillator channels；
- Q/K head dimension 64，与 ELF-B 对齐；
- 保留 RoPE、qk-norm、padding/condition mask；
- 不建立384张独立 attention maps；
- 不让全部384个 channels 只共享一张 attention map；
- \(S\) 和 \(I\) 使用 phase-safe learnable mappings：输入 \([\sin\theta_h,\cos\theta_h]\)，输出32维；
- 使用按 head 分组的轻量线性映射，并用 `tanh` 限制输出幅度，避免 phase velocity 无界；
- 同时实现固定三角函数 \(S(\theta)=\cos\theta, I(\theta)=\sin\theta\) 作为机制消融，而不是主模型；
- \(\gamma_l\) 使用可学习的正值标量，初始为0.1，并通过 sigmoid 限制在0.25以下。

注意：真正的参数量和 FLOPs 必须从实现后的 profiler 得到，不能仅凭上述维度估算。

## 8. Output 与 ELF shared decoder

ELF 的 decoder 不是独立 autoregressive decoder，而是同一 backbone 在 \(t=1\)、`mode=decode` 下进行一次 contextual recovery，再通过 unembedding 输出 token logits。

第一版继续共享完整 WONN backbone，并对移除 control tokens 后的全部文本位置读取 phase：

\[
\hat x_{\mathrm{text}}
=
W_{\mathrm{out}}
\operatorname{RMSNorm}
\left(
[\sin\theta_{\mathrm{text}}^{L,T},
 \cos\theta_{\mathrm{text}}^{L,T}]
\right).
\]

第一版只从最终 phase state 读取：

- 不直接把 \(\omega^{L}\) 接到 output head，避免模型绕过 oscillator dynamics；
- 不增加从 noisy input 到输出的直接 residual shortcut，避免高 \(t\) 区域退化成 identity mapping；
- control positions 被丢弃，source+target positions 保持与 ELF 相同的输出 shape；
- source positions 不计算 target loss，并由现有 sampler 恢复为 clean condition；
- denoise mode 对 target clean embeddings 计算 MSE；
- decode mode 对 target token logits 计算 CE。

这是第一版实现选择，尚未得到可学性或任务质量验证。后续应加入 `phase-only`、
`phase+omega` 和 `phase+input-residual` 消融。

## 9. 第一版模型规模

为了同时控制参数和 recurrent compute，第一版已实现：

| 配置 | 值 |
|---|---:|
| ELF model width | 768 |
| Oscillators per token \(K\) | 384 |
| Coupling heads | 12 |
| Oscillators per head | 32 |
| Q/K head dimension | 64 |
| WONN layers \(L\) | 6 |
| Inner steps per layer \(T\) | 2 |
| Coupling evaluations per call | 12 |
| Initial phase | Deterministic `atan2` projection |
| Phase transition | Identity carry |
| Frequency update | Once per layer, gated residual |
| Output | Phase-only；完整 source+target shape，target-only supervision |

选择 \(L=6,T=2\) 是为了让每次 denoiser call 的 attention/coupling evaluations 数量与12-layer ELF-B 大致相同。它不是严格 FLOP matched：\(S/I\)、frequency update、adapter 和 recurrent implementation 都会影响实际成本，必须 profiler 后再调整。

后续至少报告两条比较线：

1. **Compute-matched**：匹配实际 forward FLOPs 或 wall-clock，比较质量和参数量。
2. **Parameter-matched**：匹配 trainable parameters，单独报告额外 recurrent compute。

不能只用“参数更少”掩盖更高的采样计算量。

## 10. 开发计划

### Phase 0：建立代码基线

已完成：

1. ELF PyTorch `pytorch_elf` 基线固定为 `b29d883`，官方远程为 `elf-upstream`。
2. WONN 项目参考快照为 `af3f468d631d8b5a7f3730ccca5b7e686c458da1`，基于官方 `62d7ac5` 增加本地可视化工具；远程发布问题见 `docs/PHASE1_BASELINE.md`。
3. 建立 Python 3.10.12 `.venv`，保存 `requirements-lock.txt` 精确依赖快照。
4. 验证 RTX 5060 Laptop 8GB、PyTorch 2.13.0+cu130、float32/bfloat16 CUDA 运算。
5. 验证未修改 ELF-B denoise/decode 双模式随机前向；当时手工使用 32128 词表，参数量为104,594,304；真实 T5/checkpoint 使用 32100 词表和104,579,940个参数。
6. 记录环境、输入固定长度契约和未验证事项到 `docs/ENVIRONMENT.md`。

Phase 0 不包含官方 checkpoint 指标或训练复现；这些是 Phase 1 的验收内容。

### Phase 1：复现未修改 ELF baseline

已于 2026-08-07 完成，完整命令、指标、profiling 和失败记录见
[`docs/PHASE1_BASELINE.md`](docs/PHASE1_BASELINE.md)。WMT14 De→En 3000 条 validation
BLEU 为 26.55；单 batch 训练、EMA 和 checkpoint save/load 均已验证。

按成本从低到高完成：

1. 下载官方 WMT14 De→En ELF-B PyTorch checkpoint 和 validation 数据。
2. 先生成少量样本，核对 tokenizer、T5 encoder、condition mask、sampler 和 decoder 路径。
3. 运行官方 validation evaluation，记录 BLEU 和采样配置；不要求与论文 test score 完全相同。
4. 用小数据完成单 batch 训练 smoke，验证 loss、backward、optimizer、EMA 和 checkpoint save/load。
5. 记录参数量、单步前向、峰值显存、tokens/s 和 sampler latency。

完成门槛：官方 checkpoint generation/evaluation 和最小训练路径都能运行；所有失败和8GB显存下的
配置调整必须记录，不能在 baseline 未通过时进入 WONN 实现。

### Phase 2：建立模型契约测试

已于 2026-08-07 完成。15项测试在 CPU/CUDA 上全部通过，覆盖范围、运行命令和固定事实见
[`docs/PHASE2_CONTRACTS.md`](docs/PHASE2_CONTRACTS.md)。

先用原始 ELF 定义替换 backbone 后必须继续满足的契约：

- 输入/输出 shape、dtype 和 device；
- `max_length` 固定长度及 RoPE 行为；
- self-conditioning on/off；
- denoise/decode 两个 mode；
- padding mask 和 source condition 恢复；
- eval mode 确定性；
- backward 和 mixed precision；
- sampler 对完整 source+target output shape 的要求。

该接入门槛已满足；同一组契约测试已经对 ELF-WONN 通过，并纳入 Phase 3 严格验收。

### Phase 3：实现独立 WONN backbone

已于 2026-08-07 完成。实现、参数量、动力学诊断、GPU smoke 和已知限制见
[`docs/PHASE3_WONN_ELF.md`](docs/PHASE3_WONN_ELF.md)。第一版使用384 oscillators、6 layers、
每层2个 inner steps和12个 coupling heads；公共模型名为 `ELF-WONN-B`。

实际代码结构：

```text
WONNELF
  ELF control/self-conditioning preparation
  phase_projection + frequency_projection
  [WONNLayer] x L
    AttentiveWinfreeCoupling
      q/k token coupling
      GroupedPhaseMap sensitivity + influence
    recurrent phase update x T
    frequency_transition
  phase_features + FinalLayer + shared token decoder
```

`WONNELF.forward(...)` 已保持 ELF 原模型的输入输出 shape 和 mode 行为。
`src/modules/wonn_layers.py`、`src/modules/wonn_model.py` 和独立 WONN YAML 已落地；官方 ELF
配置保持不动。

Phase 3 已验证 phase wrapping、确定性初始化、mask、梯度、mixed precision、\(L=1,T=1\) 和正式
配置，并记录 phase update、\(\omega\) norm、attention entropy 与 finite fraction。训练后的动力学
稳定性和可学性属于 Phase 4。

### Phase 4：最小可学性验证

已于 2026-08-07 完成，严格命令、阈值和完整结果见
[`docs/PHASE4_LEARNABILITY.md`](docs/PHASE4_LEARNABILITY.md)。验收按顺序完成：

1. 正式 WONN-B 在固定真实 T5/WMT14 batch 上过拟合 denoising MSE；
2. 正式 WONN-B 在同一输入路径上过拟合 decoding CE；
3. batch 5 按 ELF 的4:1比例同时过拟合两种 objective；
4. target 输入相同时，仅交换 source latent 即使预测100%翻转；
5. 正式 ELF-B 和 ELF-WONN-B 在相同5样本、40步设置下，L2与CE均下降。

Phase 4 还确认 frequency gate 和 transition weights 均实际变化。它排除了“backbone 完全不可训练”
和“conditional 输出完全忽略 source”两类失败，但固定 batch 过拟合不能证明泛化、BLEU 或模型优劣。

### Phase 5：第一个端到端任务候选——WMT14 De→En

WMT14 source 64、target 64、总长度128，适合验证 control tokens、source prefix、target-only
supervision 和 shared decoding。固定 encoder、optimizer、flow schedule、self-conditioning、
sampler 和训练 tokens，依次训练：

1. 原 ELF-Transformer baseline；
2. ELF-WONN compute-matched；
3. 必要时增加 parameter-matched。

比较 BLEU、denoising MSE、decode CE、参数、FLOPs、wall-clock、峰值显存和 sampler latency。
该任务仍待项目负责人正式批准完整训练预算。

### Phase 6：主实验——OpenWebText

WMT14 出现可靠信号后再做 OWT unconditional generation，比较 Transformer/WONN 与
shared/separate decoder。主要报告 Gen. PPL–entropy frontier、sampling steps、参数、FLOPs、
wall-clock、peak memory 和 throughput，不能用 entropy collapse 换取表面 PPL 改善。

### Phase 7：XSum 长上下文压力测试

XSum source 约1024、target 64，只用于回答长 context 下 recurrence 成本、token oversmoothing 和
phase synchronization。只有 WMT14 或 OWT 至少一个出现可靠正结果后再投入。

## 11. 公平比较与消融

### 11.1 必须保持不变

- tokenizer 和 frozen T5 encoder；
- train/eval split；
- embedding bottleneck 和外部 input/output dimensions；
- Flow Matching interpolant 和 x-prediction conversion；
- time sampling/noise schedule；
- denoise/decode branch probability；
- self-conditioning 和 CFG protocol；
- ODE/SDE sampler；
- training tokens、batch size和优化器设置，除非稳定性要求改变；
- evaluation scripts 和生成样本数。

如因稳定性必须改变 optimizer、learning rate 或 gradient clipping，需要同时给 Transformer baseline 相同调参预算，并明确报告。

### 11.2 最小消融矩阵

为区分收益来自 phase geometry、recurrence、weight sharing 还是额外计算，至少需要：

1. 原 ELF Transformer；
2. ELF-WONN；
3. 与 WONN 相同重复次数的 tied-weight recurrent Transformer；
4. 去掉 wrap/sin/cos 的 Euclidean recurrent block；
5. phase features 但 \(T=1\)；
6. fixed trig \(S/I\) 与 learnable \(S/I\)；
7. fixed \(\omega\) 与 layer-wise slow \(\omega\) update；
8. shared decoder 与 frozen separate decoder；
9. phase-only readout 与 phase+frequency readout。

不需要在第一轮同时完成所有消融。优先级是1、2、3、6、8。

## 12. 需要记录的动力学诊断

普通 loss 无法判断 oscillator 是否真的被使用。训练和采样至少记录：

- 每层/每 inner step 的 phase update RMS：\(\|\Delta\theta\|\)；
- frequency norm 和 frequency residual-update norm；
- Kuramoto order parameter：

\[
r=\left|\frac{1}{NK}\sum_{i,k}e^{\mathrm{i}\theta_{i,k}}\right|;
\]

- 按 token、head 和 channel group 分解的局部 order parameter；
- phase histogram/entropy；
- attention entropy；
- control→target、source→target coupling mass；
- phase initialization 与 final phase 的变化；
- gradient norm、NaN/Inf count；
- output 对 \(\theta\)、\(\omega\)、control tokens 的敏感性。

全局 \(r\to1\) 不一定是好事。视觉分类中的强同步可能对应语言中的 token oversmoothing。语言更可能需要局部同步、多簇 phase organization，而不是所有位置同相。

## 13. 主要风险

### 13.1 Phase collapse

所有 token/oscillator 快速同步会丢失词序和 token identity。应通过 order parameter、phase entropy、head-wise diagnostics 和 BLEU/entropy 联合判断。

### 13.2 Nested compute

ELF 已经在多个 flow steps 中调用 denoiser，WONN 又在每个 layer 内 recurrent。即使参数更少，wall-clock 也可能更差。因此 coupling evaluations、FLOPs 和 sampler latency 必须单独匹配。

### 13.3 Shared decoder 冲突

WONN 可能擅长 embedding denoising，却不擅长最终 contextual token recovery。separate-decoder 诊断是研究计划的一部分，不是可选的事后补丁。

### 13.4 Attention 退化成普通 Transformer

如果引入标准 \(V\)-projection residual、FFN 和多条 Euclidean shortcuts，模型可能只是使用 sin/cos activation 的 Transformer。第一版必须让 attention 只参数化 coupling，让 phase update由 Winfree equation 完成。

### 13.5 动力学不稳定

Learnable \(S/I\)、\(\omega\) 和 \(\gamma\) 都可能导致过大的 winding。需要 bounded \(S/I\)、受控 step size、frequency normalization、mixed-precision tests 和逐步增加 \(T\)。

### 13.6 过度解释 energy

原 WONN 的简单 interaction energy 只在 zero-frequency、特定三角 interaction 和合适 coupling 条件下具有清晰 Lyapunov 意义。本项目包含非零 \(\omega\)、state-dependent attention 和 learnable \(S/I\)，不能把该 energy 当作全局 log-density，也不能预期它沿网络单调下降。

## 14. 成功标准

### 第一阶段：工程可行

- 官方 ELF baseline 可运行；
- WONN 两种 mode 均稳定训练；
- 单 batch 和固定小批次能够过拟合；
- control/source 信息确实影响 target；
- 不出现系统性 NaN、phase collapse 或 padding leakage；
- profiler 能给出可信的参数、FLOPs、显存和吞吐数据。

### 第二阶段：研究信号

满足以下任一条件都可视为值得继续：

- matched compute 下，WONN 获得更好的 BLEU 或 Gen. PPL–entropy frontier；
- matched quality 下，WONN 使用更少参数；
- matched quality 下，WONN 需要更少 flow sampling steps；
- WONN 在长序列或高噪声区间表现出更稳定的恢复能力；
- separate-decoder 诊断显示 WONN denoiser 有明确收益，即使 shared decoder 尚未解决。

只有训练 loss 更低、参数更少但 FLOPs 大幅增加，或 Gen. PPL 降低同时 entropy collapse，都不能算成功。

## 15. 已确认、默认与未决事项

### 已确认

- ELF 是主基线。
- 完整替换 ELF DiT backbone，而不是在 Transformer 中插入插件。
- ELF latent 不改成 \(\theta\) latent。
- \(\theta\) 由当前 ELF state 确定性初始化。
- \(\theta\) 只由 dynamics 改变，layer 边界 identity carry。
- \(\omega\) layer 内固定、layer 间慢速 residual update。
- 完整 in-context sequence 统一初始化为 \((\theta,\omega)\)。
- control/source/target 都是 oscillator nodes，并共同演化。
- 不额外加入 \(B_t e_t+B_m e_m\)。
- 每个 token 采用384个 oscillator，phase sin/cos feature width 为768。
- shared-WONN 是主模型，separate decoder 是 denoiser 机制诊断。
- ELF PyTorch 基线固定为 `b29d883`，WONN 项目参考快照固定为 `af3f468`（上游基点 `62d7ac5`）。
- 当前开发机为单张 RTX 5060 Laptop 8GB，Phase 0 环境见 `docs/ENVIRONMENT.md`。

### 第一版实现默认值

- 12 coupling heads，每个管理32个 oscillators；
- 6 WONN layers，每层2个 inner steps；
- learnable bounded \(S/I\)，fixed trig 作为消融；
- learnable bounded positive step size，约0.1初始化；
- phase-only readout，保持完整 source+target output shape；
- PyTorch ELF 分支作为第一工程基底。

这些默认值可以在 profiler、稳定性测试或小规模实验提供证据后调整，不需要重新讨论研究边界。

### 尚未确认

- 是否正式批准 WMT14 De→En 作为第一个端到端任务；
- 完整训练可用的额外 GPU/TPU 资源和训练时间预算；
- 第一轮使用完整 ELF-B 还是先增加一个更小的 debug scale；
- 完整训练的随机种子数量和统计显著性要求；
- 最终论文更强调 matched-compute quality、parameter efficiency，还是 few-step sampling。

## 16. 新对话接手指令

新对话或新 coworker 应先阅读本文件和 [WONN.pdf](./papers/WONN.pdf)，然后：

1. 不要重新讨论“是否把 ELF latent 改成 \(\theta\)”；该方向已明确否决。
2. 不要在第一版额外添加 timestep/mode bias；conditioning 只走 ELF control tokens。
3. 不要跨 flow sampling step carry \(\theta/\omega\)。
4. Phase 4 已完成；不要把固定 batch 结果解释为泛化或质量结论。
5. 正式批准 WMT14 训练预算并确定公平比较规则后，再进入 Phase 5 端到端训练。
6. 实现时保持 `net(...)` 外部接口和训练器不变。
7. 每个里程碑都同时检查质量、动力学诊断和实际计算成本。

如果实现发现这里的公式与 pinned upstream code 不一致，应先记录具体代码证据，再修改本文件；不要静默改变架构边界。

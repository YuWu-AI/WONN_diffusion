# WONN 文本交互机制重构计划

> 历史设计记录：该计划已经完成并整合到 `main`。其中的 `phase5-wmt14`、worktree 和旧输出路径
> 仅用于说明当时的实施过程；当前代码与实验入口以 [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)
> 为准。

## 1. 目标与结论

当前 ELF-WONN 的 attentive coupling 不是原始 WONN `StandardAttention` 的等价文本实现：它只用
attention 生成 token 权重，再直接聚合按 head 分组的 `influence(theta)`，缺少原始 WONN 的完整
`W_qkv`、value projection、head 合并和 `W_o`。这会限制振子通道之间、尤其是不同 head 之间的
信息交换。

本次重构的目标是：

1. 将 layer 内 coupling 改为原始 WONN 的完整交互顺序；
2. `S(theta)` 和 `I(theta)` 都使用原始 WONN 风格的逐振子 MLP；
3. 只做文本模态必需的适配：1D RoPE、padding key mask、逐 token 状态布局；
4. 保持 layer 间 `theta` 不做额外 transition，并将 `omega` 更新改成适合文本的独立 FFN；
5. 先直接替换 `phase5-wmt14` 分支上的旧实现，验证后把同一实现整合到 `main`；
6. 不兼容、不迁移旧 WONN checkpoint。旧 Phase 5 代码和结果只保留在 Git 历史中。

这不是新增一个可选 coupling，也不保留旧实现作为 legacy mode。完成后仓库中只有新的正式
WONN 文本交互路径。

## 2. 不变边界

以下接口和训练语义不在本次修改范围内：

- frozen T5 encoder 与 contextual latent；
- ELF Flow Matching、x-prediction、velocity 转换和现有 loss；
- self-conditioning、CFG、mode/time tokens；
- source/target 数据拼接、loss mask、sampler `restore_cond()`；
- shared continuous decoder 与 token unembedding；
- `WONNELF.forward()` 的输入输出 shape 和公开契约；
- layer 内 Euler phase update 及当前 bounded learnable phase step；
- 最终仅从 phase features `[sin(theta), cos(theta)]` 读取输出。

attention 仍然是全局双向交互。control、source、target 的所有有效位置进入同一个 coupling；
attention mask 只屏蔽 padding key，不引入 causal、source-only 或 directional mask。

## 3. 新的 layer 内 coupling

### 3.1 状态与维度

设：

- `theta, omega`: `[B, N, K]`；
- `B`: batch size；
- `N`: control + source + target 的序列长度；
- `K`: 每个 token 的振子数；
- `H`: attention head 数；
- `C = K / H`: 每个 head 的振子数和 Q/K/V head dimension。

必须检查 `K % H == 0`。head dimension 只能由 `K / H` 派生，不再允许独立的
`wonn_qk_head_dim` 改变它。正式 WMT 配置为 `K=384`、`H=12`、`C=32`。

### 3.2 周期安全的 S/I MLP

每个振子只读取自身相位的周期表示：

```text
p[b,n,k] = [sin(theta[b,n,k]), cos(theta[b,n,k])]
```

不同 token 共享参数；不同振子使用独立参数。S/I MLP 不在进入 attention 前混合 token，也不在
不同振子之间混合通道。

按照 `references/WONN/image_recognition/wlayer.py` 在 `group_size=1`、`hidden_ratio=2` 时的真实
结构实现：

```text
S: per oscillator Linear(2, 2) -> ReLU -> Linear(2, 1)
I: per oscillator Linear(2, 4) -> ReLU -> Linear(4, 1)
```

因此：

```text
S(theta), I(theta): [B, N, K]
```

输出不加 `tanh`。删除当前按 head 联合处理 32 个振子的 `GroupedPhaseMap`，也删除
`fixed_trig` 正式/消融路径。

实现时可使用一个参数化的 `PerOscillatorMLP`，权重显式带 `K` 维并通过 `einsum` 计算；不要为
384 个振子创建 Python `ModuleList` 循环。

### 3.3 完整 multi-head attention field

Q/K/V 必须全部由 `I(theta)` 生成：

```text
qkv = W_qkv(I)                         # [B, N, 3K]
q, k, v -> [B, H, N, C]
q, k = RoPE_1D(q), RoPE_1D(k)
A = softmax((q @ k^T) / sqrt(C), dim=-1)
M = A @ v                              # [B, H, N, C]
M = merge_heads(M)                     # [B, N, K]
F = ReLU(RMSNorm(W_o(M)))              # [B, N, K]
```

具体要求：

- `W_qkv: Linear(K, 3K, bias=True)`；
- `W_o: Linear(K, K, bias=True)`；
- 使用现有 1D text RoPE，只作用于 Q/K；
- 删除当前 coupling 输入前的 `phase_norm` 和 Q/K 后的 `q_norm`、`k_norm`；
- `RMSNorm + ReLU` 只放在 `W_o` 之后；
- field 不加 `I` residual、phase residual 或 Transformer FFN；
- attention dropout 保留现有 deterministic 语义；输出 projection dropout 可复用现有
  `proj_drop`，正式配置仍为 0；
- 普通训练路径优先使用 PyTorch SDPA；只有 diagnostics 路径显式计算 attention weights 和
  entropy。两条路径必须有数值等价测试（dropout=0）。

padding mask 只作用于 key 维 `j`。必须保留测试：修改 masked key 的 `theta` 不得改变任一有效
query 的 field 或 phase velocity。若一个样本不存在任何有效 key，应 fail fast，而不是对全掩码
logits 做 softmax 后产生伪造的均匀 coupling。

### 3.4 Winfree dynamics

完整 layer 内更新为：

```text
dtheta = omega + S(theta) * F
theta = wrap(theta + gamma_l * dtheta)
```

同一 layer 的 `T` 次 recurrent step 共享同一套：

- S-MLP；
- I-MLP；
- `W_qkv`；
- `W_o`；
- coupling output RMSNorm；
- phase step `gamma_l`。

每一步使用更新后的 `theta` 重新计算 S、I、Q、K、V 和 field。不同 layer 不共享这些参数。

### 3.5 参数初始化

交互结构一致还不够，coupling 的初始尺度也应对齐参考实现：

- `W_qkv` 和 `W_o` 使用 `nn.Linear` 的 PyTorch 默认初始化，与原始 `StandardAttention` 一致；
- S/I 的自定义 grouped 参数按等价 `groups=K, kernel_size=1` 的 `nn.Conv2d` 默认规则初始化；
  即每个逐振子线性层按自身 `fan_in` 使用 Kaiming-uniform 默认界限，bias 使用相同
  `fan_in` 对应的 uniform 界限；
- coupling output `RMSNorm` scale 初始化为 1；
- 不把 QKV、O 或 S/I 的输出层零初始化，确保初始 coupling field 和全部路径都有梯度。

`OmegaTransition` 是文本适配而非参考模块的逐行复制，其两层 dense FFN 使用项目现有
`_make_linear` 初始化；`ThetaEmbedding` 则按原始 grouped `1x1` projection 的默认初始化。

## 4. 新的 layer 间 omega transition

### 4.1 设计原则

原始图像 WONN 的 transition 使用二维卷积，不能直接搬到文本。新的文本 transition 应满足：

- 同时读取上一层最终的 `theta` 和 `omega`；
- 对相位使用周期安全表示；
- 不再增加第二套 token attention；
- 不使用假设局部图像结构的 1D 卷积；
- 逐 token 混合全部振子通道；
- residual 保留慢频率状态；
- 从第一个训练 step 起 FFN 主体就能获得梯度。

### 4.2 计算公式

先用逐振子独立的仿射映射压缩相位：

```text
E_theta = ThetaEmbedding([sin(theta), cos(theta)])  # [B, N, K]
```

`ThetaEmbedding` 对每个振子执行独立 `Linear(2, 1, bias=True)`，没有激活函数，不混合 token 或
振子通道。

然后更新频率：

```text
U = RMSNorm(concat(E_theta, omega))       # [B, N, 2K]
h = ReLU(Linear(2K, K)(U))               # [B, N, K]
delta_omega = Linear(K, K)(h)            # [B, N, K]
alpha_l = alpha_max * sigmoid(raw_alpha_l)
omega_next = omega + alpha_l * delta_omega
theta_next = theta
```

固定设计值：

- FFN hidden width 为 `K`；
- `alpha_init = 0.1`；
- `alpha_max = 0.25`；
- `raw_alpha_l` 根据上述初值反算；
- 不在 `delta_omega` 末端使用 `tanh`；
- 不使用零初始化 gate。

每个 layer boundary 使用独立参数，不共享 `ThetaEmbedding`、RMSNorm、FFN 或 `alpha_l`。`L` 个
coupling layer 只创建 `L-1` 个 `OmegaTransition`；最后一层后不创建无效 transition。

为了让结构边界清晰，建议把 `OmegaTransition` 从 `WONNLayer` 中拆出：

```text
for l in range(L):
    theta = wonn_layers[l](theta, omega, ...)
    if l < L - 1:
        omega = omega_transitions[l](theta, omega)
```

`theta` 在 boundary 上原样传入下一层。

## 5. 代码修改范围

### 5.1 `phase5-wmt14` 先行修改

主要文件：

- `src/modules/wonn_layers.py`
  - 删除 `GroupedPhaseMap`；
  - 用 `PerOscillatorMLP` 实现 S/I；
  - 重写 attentive coupling 为完整 QKV/O 路径；
  - 新增 `ThetaEmbedding` 和 `OmegaTransition`；
  - 让 `WONNLayer` 只负责 recurrent phase dynamics。
- `src/modules/wonn_model.py`
  - head dimension 改为 `K/H`；
  - 分别创建 `L` 个 layer 和 `L-1` 个 omega transition；
  - 调整 gradient checkpointing 与 diagnostics 聚合；
  - 保持公开 forward 契约不变。
- `src/modules/model_factory.py`
  - 不再传递 `qk_head_dim` 和 `coupling_mode`。
- `src/configs/config.py`
  - 删除 `wonn_qk_head_dim`、`wonn_coupling_mode`；
  - 只有确需调节时才增加 omega transition step 配置，否则使用上述固定设计值，避免制造无用超参。
- `src/configs/training_configs/**/*.yml`
  - 删除所有旧 QK dimension 和 coupling mode 字段；
  - 所有 WONN 配置统一指向新实现；
  - 正式 50K 配置继续使用 `384/6/3/12`；
  - 更新正式输出目录，避免自动发现并加载旧结构 checkpoint。
- `scripts/run_phase4_learnability.py`
  - 删除对旧 `frequency_gate`、`frequency_transition` 名称的假设；
  - 改为检查新 omega FFN、theta embedding 和 alpha 都收到更新。
- Phase 5 pipeline/diagnostic scripts
  - 更新旧输出目录、字段名和 diagnostics key；
  - 删除只服务于旧 WONN checkpoint warm-start 的入口；
  - 不保留旧架构兼容分支。
- `tests/`
  - 重写 WONN layer/model/factory/config/checkpoint 相关断言。

`references/WONN/` 保持只读，运行时代码不得从中导入。

### 5.2 整合回 `main`

Phase 5 验证通过并形成 clean commit 后，将同一实现整合回 `main`：

1. 只迁移 WONN 模块、配置和对应测试，不覆盖 `main` 上无关的新工作；
2. 保留 `main` 已有的 ELF baseline、外部接口和最终层无效参数修正；
3. 更新 `PROJECT_HANDOFF.md`、`docs/RESEARCH_PLAN.md`、
   `docs/WONN_ARCHITECTURE_FORMULA_CN.md` 中旧的“attention 只生成权重、无 V/O”描述；
4. 在两个分支分别运行验证，不能仅以 cherry-pick 成功代替测试；
5. 最终确认两个分支的正式 WONN 核心文件不存在语义漂移。

## 6. checkpoint 与旧实验处理

新架构与旧 WONN checkpoint 明确不兼容，原因包括：

- `GroupedPhaseMap` 被 S/I per-oscillator MLP 取代；
- Q/K 投影被完整 `W_qkv` 取代；
- 新增 V、`W_o` 和 coupling output norm；
- head dimension 从 64 改为 32；
- omega transition 参数结构完全改变。

处理规则：

- 不写参数名转换或部分迁移脚本；
- 不使用 `strict=False` 静默加载；
- `resume`、`init_from` 或 eval 加载旧 checkpoint 时必须明确失败；
- 新正式训练从随机初始化开始；
- 新配置使用新的 output directory，防止 auto-resume 命中旧文件；
- 不删除本地旧 artifacts 作为实现步骤，也不继续维护其 pipeline；需要追溯时使用 Git 历史和已有
  artifacts。

## 7. 测试与验收

### 7.1 coupling 单元测试

至少覆盖：

1. S/I 输出 shape 正确、对 `theta + 2pi` 周期等价；
2. 每个振子参数独立，不存在意外的 head 内 S/I 混合；
3. `head_dim == K/H`，不可整除时 fail fast；
4. 小张量上与直接写出的 `W_qkv -> attention -> W_o -> RMSNorm -> ReLU` 参考计算一致；
5. SDPA 普通路径与显式 diagnostics 路径在 dropout=0 时一致；
6. masked key 扰动不影响有效 query；
7. `W_o` 可以实现跨 head 的 value 重组；
8. 将 `W_o` 输出置零时 field 为零，证明没有隐藏 residual；
9. phase velocity 精确满足 `omega + S * field`；
10. recurrent steps 共享参数，并在每一步重新计算当前 phase 的 coupling；
11. forward/backward 梯度有限，S、I、QKV、O 均能收到梯度。

### 7.2 omega transition 单元测试

至少覆盖：

1. `theta` 在 layer boundary 前后逐元素不变；
2. `theta` 和 `omega` 分别变化时都能改变 `delta_omega`；
3. alpha 初值为 0.1，始终在 `(0, 0.25)`；
4. 第一个 backward 即可为 theta embedding、两层 FFN 和 alpha 产生有限梯度；
5. 6-layer 模型恰好有 5 个 transition；
6. 不同 boundary 参数对象独立；
7. 最后一层后不存在无效 transition 参数。

### 7.3 模型与训练回归

按顺序执行：

1. WONN focused CPU unittest；
2. 全仓库 unittest discovery；
3. 配置加载与 Phase 5 pipeline dry-run；
4. tiny model forward/backward、gradient checkpointing 和 deterministic 测试；
5. 正式 `384x6x3` BF16 CUDA forward/backward；
6. mixed denoising/decoding 单 batch smoke；
7. source prefix restore、sampling 和 shared decoder contract；
8. 小 batch overfit，确认 coupling 与 omega transition 的所有 trainable path 实际更新；
9. profiler 记录参数量、显存、step time，并与旧实现差异分开报告。

测试通过只证明实现与契约正确，不证明翻译质量。不得从一次有限梯度或单 batch 下降直接进入正式
50K 结论。

### 7.4 重新训练门槛

只有以下条件全部满足后才启动新的 WMT 训练：

- Phase 5 worktree 为 clean commit；
- 配置、数据 revision、encoder/tokenizer revision、seed 和硬件被 manifest 记录；
- 新输出目录为空且不能 auto-resume 旧 checkpoint；
- 先完成短程 from-scratch pilot，确认 loss、phase update、attention entropy、coupling field RMS、
  omega delta RMS、alpha 和梯度均有限且非退化；
- pilot 通过后再运行既定 50K pipeline 和独立 checkpoint evaluation。

正式比较仍使用相同数据、seed、有效 batch、optimizer-step、sampling 和评测预算。旧 Phase 5 WONN
结果不作为新实现的 warm-start 或实现正确性证据。

## 8. 完成标准

只有同时满足以下条件，重构才算完成：

- layer 内代码顺序与本文第 3 节一致；
- S/I 为原始 WONN 风格 per-oscillator MLP；
- Q/K/V 全部来自 I，且存在完整 `W_qkv`、head merge 和 `W_o`；
- coupling 后只有 `RMSNorm + ReLU`，没有 field residual；
- `theta` layer 间不变，`omega` 使用本文第 4 节的独立 residual FFN；
- 所有旧配置字段、旧参数假设和静默 checkpoint 兼容路径被移除；
- Phase 5 与 main 的实现和测试均通过；
- 实际验证命令、结果、未运行项和残余风险被记录；
- 正式训练从新架构的 clean commit、随机初始化和新输出目录开始。

## 9. 下一次对话的执行顺序

1. 在 `.worktrees/phase5-wmt14` 阅读本计划和项目规则；
2. 再次核对 `references/WONN/common/modules.py` 与
   `references/WONN/image_recognition/wlayer.py`，以源码为准；
3. 先写/改 focused tests，使其明确描述新公式；
4. 实现 `PerOscillatorMLP`、完整 attentive coupling 和 `OmegaTransition`；
5. 更新 model/factory/config/YAML/pipeline；
6. 运行从 focused 到 full、CPU 到 CUDA 的分层验证；
7. 做一次对抗式检查：公式、索引、mask、参数共享、旧 checkpoint 和无关改动；
8. 在 Phase 5 形成 clean commit 后，再将相同实现整合到 `main` 并重新验证；
9. 验证完成前不上传、不启动正式训练。

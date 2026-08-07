# Phase 3 WONN-ELF backbone

本文件记录 2026-08-07 完成的第一版 WONN-ELF backbone。Phase 3 的结论仅是：WONN 已作为
独立 backbone 接入 ELF，保持训练器和采样器接口，并通过 forward、backward、mixed objective、
diagnostics 和 sampler smoke；它不代表模型已经具备可学性或达到 ELF baseline 质量。

## 实现边界

正式实现位于：

- `src/modules/wonn_layers.py`：phase wrapping、grouped sensitivity/influence、attentive Winfree
  coupling、recurrent phase step 和 layer-wise frequency transition。
- `src/modules/wonn_model.py`：ELF-compatible conditioning、确定性 phase/frequency adapter、
  6-layer WONN backbone、phase-only readout 和 shared decoder。
- `src/modules/model_factory.py`：ELF 与 ELF-WONN 的公共构造入口。
- `src/configs/training_configs/train_de-en_ELF-WONN-B.yml`：独立 WMT14 WONN 配置；官方 ELF
  YAML 未修改。当前8GB环境显式使用 gradient checkpointing，并关闭已知缺少 Python 3.10
  headers 时不可用的 `torch.compile`。

运行时代码没有从 `references/WONN` 导入。WONN hidden state 只存在于一次 forward 内，每次
ELF flow/sampling step 都由当前 latent 重新确定性初始化。

## 第一版结构

| Item | Value |
| --- | ---: |
| Model name | `ELF-WONN-B` |
| Oscillators per token | 384 |
| Phase feature width | 768 (`sin` + `cos`) |
| WONN layers | 6 |
| Inner steps per layer | 2 |
| Coupling evaluations per forward | 12 |
| Coupling heads | 12 |
| Oscillators per head | 32 |
| Q/K head dimension | 64 |
| Step size | learnable positive scalar，0.1 init，0.25 upper bound |
| Frequency transition | once per layer，zero-initialized learnable gate |
| Readout | final phase only |
| Parameters with T5 vocab 32100 | 30,472,176 |

Attention 只生成 token coupling weights，并计算
`message = attention @ influence(theta)`；没有标准 Transformer V projection residual、FFN 或 noisy
input→output shortcut。Learned sensitivity/influence 使用按 head 分组的 phase-safe mapping 和 `tanh`。
`fixed_trig` 作为机制消融入口保留。

## 模型接口

`WONNELF.forward(...)` 与原 ELF 保持同一输入输出契约：

```python
output, decoder_logits = model(
    x,
    t,
    attention_mask=attention_mask,
    deterministic=True,
    self_cond_cfg_scale=scale,
    decoder_step_active=mode,
)
```

- `x` 支持 `(B,S,C)` 和 self-conditioned `(B,S,2C)`；
- continuous output 保持 `(B,S,C)`、float32；
- decode mode logits 保持 `(B,S,V)`、float32；
- time、SC-CFG、mode tokens、padding mask、source prefix 和 sampler 接口不变；
- `forward_with_diagnostics()` 是独立观测接口，不改变正常 forward 返回值。

## 验证

严格验收需要 CUDA，且不允许 skip：

```bash
.venv/bin/python scripts/verify_phase3.py
```

2026-08-07 在 RTX 5060 Laptop 8GB 上运行结果：40 tests，0 failures，0 skips。覆盖：

- 原 ELF 全部 Phase 2 契约；
- WONN phase wrapping、周期等价性、learned/fixed coupling 和 mask；
- ELF/WONN 公共 factory；
- WONN self-conditioning、mode、RoPE、确定性、gradient checkpointing 和 sampler；
- tiny CUDA BF16 backward；
- 正式 `384×6×2` WONN-B CUDA BF16 forward/backward 和 diagnostics；
- mixed CE/L2 `train_step` 更新。

正式配置实测：

| Smoke | Result |
| --- | --- |
| BF16 forward/backward | `(1,128,1024)` → output `(1,128,512)` + logits `(1,128,32100)`；梯度有限 |
| Forward/backward peak allocated | 322.6 MiB |
| Mixed CE/L2 Muon train step | loss 16.0375；L2 21.6939；CE 10.3810；两分支均有限 |
| Mixed train-step peak allocated | 993.8 MiB |
| Two-step conditional ODE + decode | source prefix 精确保持；latent 有限；IDs `(1,128)` |

这些时间和显存来自单次 synthetic-input smoke，不是稳定性能基线。Mixed train step 使用确定性
synthetic encoder 输出，以验证正式 shape 下的完整 `train_step`、optimizer 和 EMA 路径；Phase 4
仍需使用真实 T5 embedding 做单 batch 过拟合。

## 初始动力学诊断

正式随机初始化 forward 的6层结果：

- phase update RMS：0.1154–0.1155；
- frequency norm：1.1536；
- frequency update RMS：0（frequency gate 按设计零初始化）；
- attention entropy：4.4613–4.4700；
- coupling message RMS：0.0855–0.0872；
- final Kuramoto order parameter：0.0055；
- phase/frequency finite fraction：1.0。

这些数值只证明 dynamics 被执行且初始化未发生全局 phase collapse；不能证明训练后仍稳定。frequency
transition weights 在 gate 离开零点后才开始获得有效更新，Phase 4 必须确认 gate 和 transition 均有梯度并实际变化。

## 已知限制与下一步

- 当前 coupling 显式物化 attention weights，以便直接计算 entropy。WMT14 长度128已验证；在 OWT
  长度1024前必须重新测量显存和速度，必要时改为 SDPA 主路径加可选 diagnostic 路径。
- Phase 3 没有做 denoising/decoding 单 batch 过拟合，没有生成 BLEU，也没有证明 WONN 优于 ELF。
- 30.5M 参数少于 ELF-B 的104.6M，但12次 recurrent coupling 的真实计算量尚未与 ELF-B匹配；
  不能把参数更少直接解释为效率更高。

后续 Phase 4 已完成 denoising MSE、decoding CE、混合 objective 过拟合和 synthetic conditional
sensitivity；命令、阈值和结果见 `docs/PHASE4_LEARNABILITY.md`。这些结果只证明固定任务可学，完整
WMT14 训练仍需独立预算批准。

# Phase 2 ELF model contracts

本文件记录 2026-08-07 基于 ELF backbone 建立并验证的模型契约。目标是让后续
ELF-WONN 替换保持训练器和采样器无感知；WONN 接入后必须通过同一组行为检查。

## 运行方式

完整验收需要 CUDA，并且不允许跳过测试：

```bash
.venv/bin/python scripts/verify_phase2.py
```

CPU-only 开发检查可使用 `.venv/bin/python -m unittest discover -v`，它会运行13个 CPU 测试并
跳过2个 CUDA 测试；这不能代替完整验收。严格脚本在 CUDA 不可见、测试少于15项或出现 skip 时失败。
2026-08-07 在 RTX 5060 Laptop GPU 上实际运行结果为：15 tests，0 failures，0 skips。

## 契约覆盖

| 契约 | 测试证据 |
| --- | --- |
| 输入/输出 shape、dtype、device | 微型 ELF CPU 测试和真实 ELF-B CUDA 工厂测试 |
| 固定 `max_length` 与 RoPE | 精确长度成功，过短/过长输入均触发现有 RuntimeError |
| self-conditioning on/off | `(B,S,C)` 与 `(B,S,2C)` 均保持输出契约 |
| denoise/decode mode | `None` 不产生 logits；标量和 per-example mode gate 语义一致 |
| padding mask | 修改 masked key token 不影响 valid token logits |
| source condition mask | add-noise、restore helpers 和完整 ODE rollout 均保持 source prefix |
| eval 确定性 | 固定输入连续两次 forward 和 ODE rollout bitwise 一致 |
| backward | CPU gradient checkpointing 与 CUDA BF16 backward 均产生有限梯度 |
| sampler shape | latent 保持完整 source+target shape，decoder 返回完整 token shape |

测试文件：

- `tests/test_data_contract.py`：source/target/padding mask 和固定长度 dataloader。
- `tests/test_model_contract.py`：模型 forward、mode、self-conditioning、RoPE、确定性、backward、
  CUDA BF16，以及官方 ELF-B 的参数量和真实 shape。
- `tests/test_sampling_contract.py`：conditioning 恢复、ODE rollout、decoder shape 和确定性。

## 固定的 baseline 事实

- 官方 ELF-B 参数量：104,579,940（T5/decoder vocab 32100）。
- self-conditioned input：`(B, 128, 1024)`。
- continuous output：`(B, 128, 512)`，float32。
- decoder logits：`(B, 128, 32100)`，float32。
- 真实 ELF-B 测试使用 CUDA BF16 autocast，output head 保持 float32。
- `self_cond_cfg_scale` 保持 optional；省略时仍插入基础 self-conditioning tokens，传值时再叠加
  scale embedding，使 prefix 数量始终与预计算 RoPE 长度一致。
- 当前 RoPE 契约要求实际序列长度严格等于 `max_length`；padding/truncation 由 dataloader 负责。

## 边界与下一步

Phase 2 没有修改 Flow Matching、mask、sampler 或 decoder，也没有接入 WONN。契约审核发现
`self_cond_cfg_scale=None` 时缺少基础 self-conditioning tokens，导致 prefix 数量与 RoPE 长度不一致；
`src/modules/model.py` 已做最小修复，使 optional 参数语义与现有接口一致，传入 scale 的正式路径不变。
测试使用标准库 `unittest`，未新增依赖。Phase 2 验收不依赖
`references/WONN` 是否在当前 worktree 中初始化，运行时代码也未从中导入。

Phase 3 已使用 `src/modules/wonn_layers.py`、`src/modules/wonn_model.py` 和独立 WONN YAML 完成
接入。模型和 sampler 契约通过可复用 mixin 同时验证 ELF 与 ELF-WONN；没有为新模型放宽 shape、
mask、确定性或 sampler 断言。实现和严格验收结果见 `docs/PHASE3_WONN_ELF.md`。

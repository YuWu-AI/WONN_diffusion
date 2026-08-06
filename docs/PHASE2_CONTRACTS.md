# Phase 2 ELF model contracts

本文件记录 2026-08-07 在未修改 ELF backbone 上建立并验证的模型契约。目标是让后续
ELF-WONN 替换保持训练器和采样器无感知；WONN 接入后必须通过同一组行为检查。

## 运行方式

完整验收需要 CUDA：

```bash
.venv/bin/python -m unittest discover -v
```

在不可见 GPU 的沙箱内，同一命令运行 13 个 CPU 测试并跳过 2 个 CUDA 测试；这不能代替完整验收。
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
- 模型配置 `num_self_cond_cfg_tokens > 0` 时，调用者必须传入对应 scale token，使 prefix 数量与
  预计算 RoPE 长度一致。
- 当前 RoPE 契约要求实际序列长度严格等于 `max_length`；padding/truncation 由 dataloader 负责。

## 边界与下一步

Phase 2 没有修改 ELF 模型、Flow Matching、mask、sampler 或 decoder 实现，也没有接入 WONN。
测试使用标准库 `unittest`，未新增依赖。`references/WONN` 未初始化且运行时代码未从中导入。

Phase 3 应先独立实现 `src/modules/wonn_layers.py`，再组装 `src/modules/wonn_model.py` 和独立
WONN YAML。接入公共 backbone factory 后，应把 WONN 工厂加入本契约套件，并要求 ELF 与
ELF-WONN 同时通过；不能通过放宽 shape、mask、确定性或 sampler 断言来迁就新模型。

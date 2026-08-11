# 已验证本机环境

本文件记录 2026-08-06 在当前工作站完成验证的 ELF PyTorch 环境。它描述已验证事实，
不代表论文指标已经复现。

## Hardware

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU memory | 8151 MiB |
| Compute capability | 12.0 |
| NVIDIA driver | 580.173.02 |
| System CUDA toolkit | 未安装（无 `nvcc`） |
| Disk available after setup | 817 GiB |

PyTorch wheel 自带 CUDA runtime，因此运行 ELF 不依赖系统 `nvcc`。只有编译自定义 CUDA
extension 时才需要额外安装系统 toolkit。

## Python environment

项目环境位于被 Git 忽略的 `.venv/`：

```bash
uv venv --python 3.10 .venv
uv pip sync --python .venv/bin/python requirements-lock.txt
```

已验证的核心版本：

| Package | Version |
| --- | --- |
| Python | 3.10.12 |
| uv | 0.11.29 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| Transformers | 4.44.2 |
| NumPy | 1.26.4 |
| datasets | 5.0.1 |

`requirements.txt` 保留 ELF upstream 的允许范围；`requirements-lock.txt` 是本次环境的
完整精确快照。前者用于重新解析，后者用于复现。

## Verification performed

以下检查已在 2026-08-06 实际运行：

- `uv pip check`：89 个包依赖一致；
- PyTorch 可识别一张 RTX 5060；
- PyTorch arch list 包含 `sm_120`；
- CUDA float32 和 bfloat16 matrix multiplication 均产生有限值；
- ELF-B denoise/decode 双模式随机前向成功；
- 随机前向手工使用 32128 词表时，ELF-B 参数量为104,594,304；
- 输入固定为 `(1, 128, 1024)`（包含 self-conditioning channel concat）；
- continuous output：`(1, 128, 512)`；
- decoder logits：`(1, 128, 32128)`；真实 T5 tokenizer/checkpoint 使用 32100，正式 baseline
  参数量为104,579,940，详见 [`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md)；
- 两次顺序前向约 0.134 秒，峰值 allocated memory 约 493.3 MiB。

时间和显存数字来自单次未预热随机前向，只用于环境 sanity check，不能作为正式性能基线。

## 输入契约说明

ELF 的 RoPE 按配置 `max_length` 预计算。直接传入短于 `max_length` 的未 padding 序列会发生
位置维度不匹配；正式 dataloader 会把序列 pad/truncate 到配置长度。模型契约测试必须使用该真实
输入约束。

本环境快照当时只完成随机前向 sanity check；此后官方 ELF checkpoint、training smoke、sampler
和 validation evaluation 已完成，结论统一见 [`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md)。

# Phase 1 ELF baseline reproduction

本文件记录 2026-08-07 在单张 RTX 5060 Laptop 8GB 上完成的 WMT14 De→En ELF-B
PyTorch baseline 验收。结论是：官方 checkpoint generation/validation 和最小训练路径均已跑通，
可以进入 Phase 2 模型契约测试；在 Phase 2 完成前不接入 WONN backbone。

## 固定输入

| Item | Value |
| --- | --- |
| ELF code baseline | `b29d883`（`pytorch_elf`） |
| Official checkpoint | `embedded-language-flows/ELF-B-de-en-torch` |
| Hugging Face snapshot | `93ce98315a2dec985cdefb8d8e62ab618e896ec4` |
| Checkpoint state | step 880600, epoch 100 |
| Validation dataset | `embedded-language-flows/wmt14_de-en_validation_t5`，3000 条 |
| Encoder/tokenizer | `t5-small` |
| Tokenizer / decoder vocab | 32100 |
| ELF-B parameters | 104,579,940 |
| Device/runtime | RTX 5060 Laptop 8GB, PyTorch 2.13.0+cu130 |

Phase 0 的随机前向曾手工传入 `vocab_size=32128`，因此记录了 104,594,304 个参数和
`(1, 128, 32128)` logits。真实 T5 tokenizer 和官方 checkpoint 均使用 32100，正式 baseline
应以本文件的 104,579,940 个参数和 `(B, 128, 32100)` logits 为准。

## Validation evaluation

运行命令：

```bash
.venv/bin/python src/eval.py \
  --config src/configs/training_configs/train_de-en_ELF-B.yml \
  --checkpoint_path embedded-language-flows/ELF-B-de-en-torch \
  --config_override global_batch_size=16 \
  --config_override num_samples=3000 \
  --config_override use_bf16=true \
  --config_override use_compile=false \
  --config_override use_wandb=false \
  --config_override output_dir=outputs/phase1-elf-b-wmt-de-en-validation
```

采样配置保持官方 YAML：ODE、64 steps、CFG 2、self-conditioning CFG 1、
logit-normal time schedule、seed 42。

| Metric | Result |
| --- | ---: |
| BLEU | 26.5521 |
| ROUGE-1 / ROUGE-2 / ROUGE-L | 60.0050 / 34.4043 / 55.2751 |
| Generation time | 756.22 s / 3000 samples |
| Decode time | 9.41 s / 3000 samples |
| End-to-end wall time | 13:08.54 |
| Average generation batch latency | 4.02 s / batch of 16 |
| Generation throughput | 3.97 samples/s；507.8 padded latent positions/s |
| Batch-16 peak allocated CUDA memory | 2078.6 MiB |
| Full-run `nvidia-smi` snapshot | 2370 MiB used，98% utilization |

官方说明给出的 PyTorch validation BLEU 约为 26.7；本次 26.55 与其一致。峰值 CUDA 数字来自
单独复跑的相同 batch-16 eager BF16 预检；`nvidia-smi` 是完整运行中的单次快照，不是峰值。

## Single-batch training smoke

从 validation cache 选取 1 条样本并 `save_to_disk` 到被忽略的
`data/phase1_wmt14_smoke/`，只用于验证训练机制，不作为训练结果：

```bash
.venv/bin/python -c \
  'from datasets import load_dataset; d=load_dataset("embedded-language-flows/wmt14_de-en_validation_t5")["train"]; d.select(range(1)).save_to_disk("data/phase1_wmt14_smoke")'
```

训练使用官方 ELF-B 配置并通过 CLI 覆盖：batch 1、1 epoch、BF16、gradient checkpointing、
Muon、无 W&B、无 evaluation、`compile_train=false`。第一次运行完成 loss、backward、optimizer、
EMA 和 `checkpoint_1` 保存；第二次成功恢复 step/epoch、EMA、optimizer、scheduler 和 RNG，随后继续到
step 2 并保存 `checkpoint_2`。

| Item | Result |
| --- | ---: |
| Step-2 loss / L2 / CE | 0.2534 / 0.2534 / 0.0000 |
| Synchronized training-step time | 0.396 s |
| Throughput | 2.49 steps/s；约 323 padded token positions/s |
| Peak allocated CUDA memory | 2285.4 MiB |
| Checkpoint size | about 1.2 GiB each |
| Checkpoint restore | step 1, epoch 1 successfully restored and continued |

单样本随机分支选择了 denoiser，所以本次 CE 指标为 0；decoder forward 已由 checkpoint evaluation
覆盖，两个训练分支的强制覆盖属于 Phase 2 契约测试。

## 环境调整与失败记录

- `torch.compile` evaluation 预检失败：Triton 编译需要
  `/usr/include/python3.10/Python.h`，但 Ubuntu 24.04 软件源只提供已安装的 Python 3.12 headers。
- 没有混用 Python 3.12 headers，也没有改变模型；validation 改用 eager BF16。
- 新增 `compile_train` 配置，默认 `true`，保持官方训练默认行为；8GB smoke 显式设为 `false`。
- `eval.py` 和 `train.py` 只增加 peak allocated CUDA memory/首 step 耗时日志，不改变模型接口或算法。
- 官方 ELF YAML 未修改，所有 8GB smoke 调整均通过 CLI override 提供。
- checkpoint、数据、生成结果和缓存位于 `.gitignore` 覆盖的目录或用户级 Hugging Face cache。

## 尚存的仓库可复现性问题

主项目当前记录 WONN submodule `af3f468`，它基于官方 `62d7ac5` 增加本地可视化工具，但该提交
尚未发布到 `.gitmodules` 指向的官方远程。当前 worktree 通过主工作区的本地对象完成检出；全新 clone
仍无法获得 `af3f468`。在对外共享仓库前，需要将该提交推送到可访问的 fork 并更新 `.gitmodules`。

## Phase 1 验收

- [x] 官方 checkpoint 下载和恢复
- [x] 少量 WMT14 generation
- [x] 3000 条 validation evaluation
- [x] loss、backward、optimizer 和 EMA
- [x] checkpoint save/load 及恢复后继续训练
- [x] 参数量、训练 step、显存、吞吐和 sampler latency
- [x] 记录 8GB 环境调整和失败路径

下一步是 Phase 2：先为未修改 ELF 建立输入输出、mask、两种 mode、self-conditioning、确定性、
mixed precision、backward 和 sampler shape 契约测试，再开始 WONN 实现。

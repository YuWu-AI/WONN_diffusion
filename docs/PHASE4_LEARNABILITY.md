# Phase 4 fixed-batch learnability

本文件记录 2026-08-07 在单张 RTX 5060 Laptop 8GB 上完成的 WONN-ELF 最小可学性验收。
结论是：正式30.5M参数 WONN-B 能在固定真实 T5/WMT14 batch 上分别和联合优化 ELF 的两种
objective，并且在排除 target 输入差异后会使用 source 信息。它不代表模型已经泛化、达到 BLEU
baseline，或优于 ELF Transformer。

## 严格运行方式

```bash
.venv/bin/python scripts/verify_phase4.py
```

严格入口先运行全部回归测试，要求至少41项、0 failures、0 skips；随后执行：

```bash
.venv/bin/python scripts/run_phase4_learnability.py \
  --output outputs/phase4/learnability_metrics.json
```

输出 JSON 位于被 `.gitignore` 覆盖的 `outputs/`，不作为源码提交。默认实验使用正式
`ELF-WONN-B`、BF16、gradient checkpointing、AdamW、学习率 `3e-4`。三个固定 objective 和
ELF/WONN 对照仍调用项目的 `train_step`、gradient clipping 和 EMA；synthetic conditional task
使用直接 CE、gradient clipping 和独立 optimizer，不更新 EMA。AdamW 是固定任务的 debug
optimizer，不是 Phase 5 完整训练的优化器结论。

## 固定任务设计

- 从官方 WMT14 De→En validation cache 取前5条样本，并用冻结的真实 `t5-small` encoder 计算
  contextual embeddings；不是随机 encoder 输出。
- encoder 输出只预计算一次，`FrozenBatchEncoder` 仅接受完全相同的 token batch。
- 每个 optimizer step 前重置 flow time、noise 和 branch RNG，使 loss 变化对应同一个有限任务。
- denoising 和 decoding 分别使用1条样本训练80步。
- mixed objective 使用5条样本训练80步；固定 branch assignment 为4条 denoising、1条 decoding，
  即 ELF 配置的 `decoder_prob=0.2`。
- 对照曲线使用同5条样本、相同 AdamW、学习率、随机种子和40步，分别训练正式 ELF-B 与
  ELF-WONN-B。

这种固定随机性的设置适合回答“能否过拟合”，不能估计随机 batch 的训练方差或 validation
generalization。

## 实测结果

2026-08-07 严格运行共发现41项回归测试，结果为0 failures、0 skips；随后15项 learnability
gate 全部通过。

| Experiment | Initial | Final | Final / initial |
| --- | ---: | ---: | ---: |
| WONN denoising L2，80 steps | 0.840325 | 0.000123 | 0.000146 |
| WONN decoding CE，80 steps | 10.459511 | 0.000372 | 0.000036 |
| WONN mixed total，80 steps | 2.584379 | 0.001195 | 0.000462 |
| WONN mixed L2 | 0.374681 | 0.001129 | — |
| WONN mixed CE | 10.380861 | 0.001427 | — |

Mixed run 中第一层 frequency gate 的绝对变化为0.009692，frequency transition weight 的 L2
变化为3.366105；因此 zero-initialized gate 离开零点后，transition 参数也获得了有效更新。

相同5样本、40步对照：

| Model | Parameters | Initial total | Final total | Ratio | Final L2 | Final CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ELF-WONN-B | 30,472,176 | 2.577899 | 0.034080 | 0.013220 | 0.038501 | 0.018481 |
| ELF-B | 104,579,940 | 2.607436 | 0.078806 | 0.030224 | 0.101117 | 0.000087 |

两者都能拟合该固定小批次。不能从一条固定 batch 曲线推断 WONN 优于 ELF；正式比较需要相同训练
tokens、多个随机种子、独立 validation 和实际 compute profiling。

## Synthetic conditional sensitivity

条件任务使用 mixed run 后的 WONN：两个样本的 target latent 输入完全相同，只有前8个 source
positions 使用不同的真实 T5 embeddings；两条样本分别要求 target 输出 token id 10和20。训练120步后：

| Metric | Result |
| --- | ---: |
| CE | 17.800270 → 0.000008 |
| Original-source accuracy | 1.0 |
| Swapped-source accuracy | 0.0 |
| Prediction change fraction | 1.0 |
| Correct-vs-alternative margin | 12.024333 |
| Margin after source swap | -12.024333 |

source swap 使预测和 margin 完整反转，证明该固定任务的 target 输出实际依赖 source，而不是仅靠
target input 或样本顺序完成分类。

## 资源与边界

完整 learnability 实验耗时28.46秒，峰值 allocated CUDA memory 约为3217.5 MiB；数字来自单次严格
运行，只用于本机资源规划。Phase 4 没有生成 BLEU、没有训练随机 mini-batch 序列，也没有保存
checkpoint。进入 Phase 5 前仍需确认完整 WMT14 预算、随机种子、调参规则和 compute-matched 配置。

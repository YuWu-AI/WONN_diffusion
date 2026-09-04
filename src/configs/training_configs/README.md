# 训练配置索引

## 官方 ELF 基线

以下配置来自或保持兼容 ELF PyTorch 基线，不由 WONN 实验覆盖：

- `train_owt_ELF-B.yml`、`train_owt_ELF-M.yml`、`train_owt_ELF-L.yml`
- `train_xsum_ELF-B.yml`
- `train_de-en_ELF-B.yml`

## 当前 WMT14 100K 架构筛选

- `train_de-en-ELF-B-E0.yml`：E0，ELF-B，学习率 `2e-3`。
- `train_de-en-WONN-L12K768T3-W0.yml`：W0，L12/K768/T3，学习率 `1e-3`。
- `train_de-en-WONN-L6K768T3-W1.yml`：W1，L6/K768/T3，学习率 `1e-3`。
- `train_de-en-WONN-L9K768T3-W2.yml`：W2，L9/K768/T3，学习率 `1e-3`。

四份配置均从随机初始化训练到 100K optimizer steps，保存
5K/10K/25K/40K/60K/80K/100K，并用相同 1000 条 validation 样本评测。每个模型独占一张 GPU，
不使用 DDP。

`global_batch_size` 表示每次 optimizer update 的 effective batch。当前配置为：

```text
512 = per-device micro-batch 16 × world size 1 × grad_accum_steps 32
```

退役的 WMT14 pilot、旧云端 pair、50K/90K/130K 和机制消融配置只保留在 Git 历史中。

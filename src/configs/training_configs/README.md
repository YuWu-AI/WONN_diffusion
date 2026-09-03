# 训练配置索引

## 官方 ELF 基线

以下配置来自或保持兼容 ELF PyTorch 基线，不由 WONN 实验覆盖：

- `train_owt_ELF-B.yml`、`train_owt_ELF-M.yml`、`train_owt_ELF-L.yml`
- `train_xsum_ELF-B.yml`
- `train_de-en_ELF-B.yml`

## 当前 Phase 1 云端配对

- `train_de-en-ELF-B-cloud-60k.yml`
- `train_de-en-WONN-L12K768T3-cloud-60k.yml`

两份配置都从随机初始化训练到 30K，使用相同数据 revision、seed、有效 batch、优化器和 checkpoint
步点。文件名中的 `60k` 仅为已发布路径的历史名称；当前 YAML 和 pipeline 的权威预算是 30K。
GPU 布局固定为 2+2，每个模型使用 2 ranks。

配置中的 `global_batch_size: 24` 与正式运行口径一致，并满足：

```text
effective batch = per-device batch × world size × grad_accum_steps
```

退役的 WMT14 pilot、20K/50K/90K/130K、wall-clock matching 和机制消融配置只保留在 Git 历史
中，不属于当前配置面。

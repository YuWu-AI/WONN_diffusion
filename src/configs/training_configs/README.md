# 训练配置索引

## 官方 ELF 基线

以下配置来自或保持兼容 ELF PyTorch 基线，不由 WONN 实验覆盖：

- `train_owt_ELF-B.yml`、`train_owt_ELF-M.yml`、`train_owt_ELF-L.yml`
- `train_xsum_ELF-B.yml`
- `train_de-en_ELF-B.yml`

## 当前 Phase 1 云端配对

- `train_de-en-ELF-B-cloud-60k.yml`
- `train_de-en-WONN-L12K768T3-cloud-60k.yml`

两份配置都从随机初始化训练到 60K，使用相同数据 revision、seed、有效 batch、优化器和 checkpoint
步点。文件名不绑定 world size；实际使用 2+2（每个模型 2 ranks）还是 4 卡串行（每个模型 4
ranks）由 `scripts/run_cloud_pair_pipeline.sh` 决定。

配置中的 `global_batch_size: 48` 是 profile 前的初始候选。正式运行由
`DLM_WONN_GLOBAL_BATCH_SIZE` 统一覆盖两边，且满足：

```text
effective batch = per-device batch × world size × grad_accum_steps
```

退役的 WMT14 pilot、20K/50K/90K/130K、wall-clock matching 和机制消融配置只保留在 Git 历史
中，不属于当前配置面。

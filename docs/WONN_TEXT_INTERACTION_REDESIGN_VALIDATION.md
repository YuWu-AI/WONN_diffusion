# WONN 文本交互重构验证记录

## 1. 当前范围

- Phase 5 先行实现：`phase5-wmt14` commit `3e9f016`。
- `main`：已人工迁移 WONN 模块、正式配置、checkpoint fail-fast 和对应测试；不迁移 Phase 5
  专用训练流水线。
- 新旧 WONN checkpoint 不兼容，不做参数转换、部分加载或 warm-start。

## 2. 已运行验证

### Phase 5 分支

```bash
PYTHONPATH=src ../../.venv/bin/python -m unittest discover -v tests
```

结果：100 tests passed；沙箱内 4 项 CUDA tests 因设备不可见而 skip。

```bash
PYTHONPATH=src ../../.venv/bin/python scripts/verify_phase3.py
```

结果：沙箱外 GPU 严格验证 100 tests passed、0 skip，包括 tiny BF16、正式 `384x6x3`
forward/backward、gradient checkpointing、diagnostics、sampler 和 shared decoder contract。

```bash
PYTHONPATH=src ../../.venv/bin/python scripts/profile_phase5.py \
  --config src/configs/training_configs/train_de-en-WONN-L6T3-phase5-50k.yml \
  --model ELF-WONN-B --decoder-prob 0.5 --batch-size 2 \
  --warmup-steps 0 --measure-steps 1 \
  --output outputs/phase5/redesign_v2/profile/wonn_l6t3_b2_mixed_1step.json \
  --config-override compile_train=false \
  --config-override gradient_checkpointing=true
```

结果：真实缓存 WMT14 batch 同时包含 95 个 L2 token 与 77 个 CE token；loss 有限；
参数量 `26,252,399`；RTX 5060 Laptop GPU 峰值 allocated/reserved 约
`1120/1402 MiB`；单个未预热测量 step 为 `0.134 s`。该时间只能作为 smoke 记录，不能作为稳定
吞吐结论。

```bash
scripts/launch_phase5_50k.sh --dry-run
```

结果：成功生成指向 `outputs/phase5/redesign_v2/formal50k` 的启动命令；未启动训练。

### main

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -v tests
```

结果：55 tests passed；沙箱内 4 项 CUDA tests 因设备不可见而 skip。

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_phase3.py
```

结果：沙箱外 GPU 严格验证 55 tests passed、0 skip。正式参数量、BF16 forward/backward、
gradient checkpointing、sampler、shared decoder、checkpoint fail-fast 和新结构 focused tests 均通过。

逐文件 `cmp` 确认两个分支的 `wonn_layers.py`、`wonn_model.py` 和 `model_factory.py` 完全一致。

## 3. 结构性验收证据

- coupling tests 覆盖逐振子 S/I 周期性与独立性、`K/H` head dimension、直接公式等价、
  SDPA/diagnostics 等价、padding key 隔离、全 mask fail-fast、跨 head `W_o`、无隐藏 field
  residual、recurrent 重算和 S/I/QKV/O 梯度。
- transition tests 覆盖 boundary 上 theta 不变、theta/omega 都影响 delta、alpha 初值与范围、
  第一次 backward 的 theta embedding/两层 FFN/alpha 梯度，以及 6-layer 恰有 5 个独立 transition。
- 配置测试拒绝旧 `wonn_qk_head_dim` 与 `wonn_coupling_mode`；checkpoint 测试确认旧参数名在
  resume/eval 加载阶段明确失败。

## 4. 尚未运行与残余风险

- 未启动新的 50K 或其他正式训练。
- 未完成新架构的短程 overfit、source-swap、自由采样和独立 checkpoint evaluation。
- 未验证 1024/1088-token、DDP、故意中断/resume 或云端容器重建。
- diagnostics 仍显式生成 `[B,H,N,N]` attention weights，长序列诊断存在显存风险。
- 单 batch 有限 loss/梯度只证明实现路径可运行，不证明翻译、生成或条件语义质量。

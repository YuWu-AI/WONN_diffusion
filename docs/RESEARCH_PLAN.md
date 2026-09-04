# ELF-WONN 研究计划

> 当前口径更新：2026-09-04。Phase 1–4 的基础实现与验收见
> [`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md)。

## 1. 当前研究顺序

| 阶段 | 任务 | 目的 |
| --- | --- | --- |
| Phase 1 | WMT14 100K 四模型架构筛选 | 分离 WONN 深度效应，确认后期质量走势 |
| Phase 2 | OpenWebText 无条件生成 | 主实验，比较 PPL–entropy 与资源效率 |
| Phase 3 | XSum 条件摘要 | 验证长条件上下文和事实一致性 |

WMT14 仍是诊断任务，不承担最终通用语言建模结论。Phase 1 只允许执行下述固定矩阵；结束后根据
预先定义的 gate 决定是否把少量模型延长到 200K，然后进入 Phase 2。

## 2. Phase 1：WMT14 100K 架构筛选

### 2.1 研究问题

在数据、生成框架、宽度、inner steps、优化器和有效 batch 相同的条件下，WONN 的后期质量下降
是否随深度增加而增强。L6、L9、L12 构成单一深度变量；E0 提供官方 ELF-B 学习率基线。

### 2.2 固定矩阵

| 编号 | 模型 | 参数量 | 峰值学习率 | GPU |
| --- | --- | ---: | ---: | ---: |
| E0 | ELF-B | 104,579,940 | `2e-3` | 0 |
| W0 | WONN-L12/K768/T3 | 76,024,187 | `1e-3` | 1 |
| W1 | WONN-L6/K768/T3 | 51,076,463 | `1e-3` | 2 |
| W2 | WONN-L9/K768/T3 | 63,550,325 | `1e-3` | 3 |

共同训练口径：

- 从随机初始化开始，seed 42；
- 100K optimizer steps，constant LR，前 5K optimizer steps linear warmup；
- effective/global batch 512；单卡 micro-batch 16，gradient accumulation 32；
- Muon、EMA `0.9999`、相同 WMT14/encoder/tokenizer revision；
- checkpoint：5K、10K、25K、40K、60K、80K、100K；
- 每个 checkpoint 使用相同前 1000 条 validation 样本，共 28 次评测；
- 指标：BLEU、chrF++、TER、空输出率、唯一输出率、长度比、loss、吞吐、耗时和峰值显存。

E0 使用 ELF 论文/官方配置在 effective batch 512 下对应的 `2e-3`；三组 WONN 固定 `1e-3`。
这一轮不同时更改 WONN 的 step bound、alpha bound、sampler、loss 或 inner steps。

### 2.3 执行 gate

1. CPU unit tests。
2. 在目标 GPU 上分别 profile 四个模型的 micro-batch 16；任何一组 OOM 都必须修改正式配置并
   重新形成 clean commit，不能只在命令行临时降低 batch。
3. 四组各做单 batch BF16 forward/backward；验证 loss、梯度和参数有限。
4. 做一次单卡 checkpoint/resume smoke，确认 optimizer-step 文件名、scheduler、EMA、RNG 和
   metrics 连续。
5. 从同一 clean commit 启动四个单卡进程，不使用 DDP、warm start 或历史 checkpoint。
6. 训练完成后统一评测并由 pipeline 校验全部产物。

### 2.4 100K 后的决策

重点比较 40K→60K→80K→100K 的斜率，而不是只看单个终点：

- L9 明显最好：延长 E0 与 L9 到 200K；
- L6 与 L9 接近且仍上升：可同时延长二者，L12 停止；
- 三种 WONN 都出现相似下降：下一轮只改变 dynamics step scale，不再用深度解释；
- 只有更深模型随训练恶化：优先检查深度累积的 phase/frequency dynamics，再决定 step/alpha 消融；
- 100K 仍明显欠拟合但走势稳定：只延长 E0 与最佳 WONN，不自动把四组全部扩到论文完整预算。

完整 3000 条 validation 复评只对 E0 和入选 WONN 执行，不混入本轮统一 1000 样本曲线。

## 3. Phase 2：OpenWebText

### 研究问题

WONN 在 1024-token 无条件生成中，能否在匹配训练预算下获得有竞争力的 generative perplexity，
同时保持合理 entropy、吞吐、显存和 sampler latency。

### 执行顺序

1. 复评官方 ELF-B OWT checkpoint，验证 PPL/entropy evaluator。
2. profile 1024-token per-device micro-batch，再用 accumulation 达到目标 effective batch。
3. 先跑 2K/5K/10K ELF/WONN pilot。
4. 对相同 sampling steps 和 guidance 报告 PPL–entropy 曲线、空输出率和资源指标。
5. pilot 有可信信号后，再决定完整 compute-matched 与 parameter-matched 训练。

若 PPL 改善只来自 entropy collapse，或长序列成本不可接受，则先修正结构，不直接扩大预算。

## 4. Phase 3：XSum

### 研究问题

WONN 在 1088-token 条件上下文中能否正确使用 source，并取得有竞争力的 ROUGE 与事实一致性。

### 执行顺序

1. 复评官方 ELF-B XSum checkpoint。
2. 验证 source+target 拼接、padding、source restore、target-only loss 和 label-drop 契约。
3. 通过单卡、DDP、resume 和 conditional mask smoke。
4. 跑 ELF/WONN 5K/10K pilot，再根据 ROUGE、事实一致性和资源指标决定完整训练。

Phase 3 不早于 Phase 2 的长序列 gate。

## 5. 公共实验纪律

- baseline 与 WONN 使用相同数据 revision、seed、有效 batch、训练预算和采样配置；明确列出的学习率
  对照除外。
- checkpoint、日志、样本、缓存和数据只写入 Git 忽略目录或云端持久卷。
- 正式运行必须来自 clean commit，并记录配置、依赖、数据 revision、硬件和随机种子。
- 每阶段先 unit/smoke，再 pilot，最后才允许扩大预算。
- “工件完整”“代码可运行”“模型质量更好”是三个独立结论，不得互相替代。

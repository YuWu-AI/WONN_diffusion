# 文档导航

本文是 `docs/` 的状态索引。先按用途选择文档，不要把历史计划当作现役命令。

## 现役文档

| 文档 | 状态 | 用途 |
| --- | --- | --- |
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | 权威计划 | Phase 1–3 的研究问题、顺序和 gate |
| [`CLOUD_PHASE1_RUNBOOK.md`](CLOUD_PHASE1_RUNBOOK.md) | 当前运行手册 | 4-GPU、四模型、100K 架构筛选 |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | 当前索引 | 代码、配置、脚本和产物边界 |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | 历史环境快照 | 2026-08-06 本机已验证环境；硬件状态需实时复核 |
| [`ELF_UPSTREAM_README.md`](ELF_UPSTREAM_README.md) | 固定上游快照 | ELF `b29d883` 的官方 PyTorch 用法 |

## 架构参考

`architecture/` 描述现行代码所实现的算法，不包含运行计划：

- [`architecture/ELF_TRANSFORMER_DENOISING_FORMULA_CN.md`](architecture/ELF_TRANSFORMER_DENOISING_FORMULA_CN.md)
- [`architecture/WONN_ALGORITHM_CONCISE_CN.md`](architecture/WONN_ALGORITHM_CONCISE_CN.md)
- [`architecture/WONN_ARCHITECTURE_FORMULA_CN.md`](architecture/WONN_ARCHITECTURE_FORMULA_CN.md)

## 阅读顺序

1. 新接手项目：根目录 `README.md` → `PROJECT_HANDOFF.md` → `RESEARCH_PLAN.md`。
2. 执行当前云实验：`CLOUD_PHASE1_RUNBOOK.md` → `PROJECT_STRUCTURE.md`。
3. 修改模型：先读 `PROJECT_HANDOFF.md` 的接口边界，再读 `architecture/`。
4. 追溯旧实验：使用 Git 历史和 `outputs/` 中保留的产物，不从当前主线复制已退役命令。

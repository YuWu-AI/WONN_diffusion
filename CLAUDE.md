# DLM-WONN project rules

- `main` 以 ELF 官方 PyTorch `pytorch_elf` 分支为执行基线；`elf-upstream` 仅用于同步官方代码。
- 保持 ELF 的 encoder、Flow Matching、self-conditioning、conditioning mask、sampler 和 shared decoder 接口不变，除非任务明确要求修改。
- WONN 正式实现放在 `src/modules/`；`references/` 只读参考，运行时代码不得从中导入。
- 保留官方 ELF 配置作为基线；WONN 和消融实验使用独立 YAML，避免覆盖基线配置。
- 模型替换必须保持 `src/modules/model.py` 现有的输入输出契约，使训练器和采样器不感知 backbone 类型。
- 先验证官方 baseline，再实现 WONN；先做单元测试和单 batch smoke test，再启动完整训练。
- checkpoint、日志、生成样本和本地数据不得提交，统一放入 `.gitignore` 覆盖的目录。
- 云端实验必须从 clean commit 启动，并记录配置、依赖、数据 revision、硬件和随机种子。

文档状态和阅读顺序见 `docs/README.md`；技术基线见 `PROJECT_HANDOFF.md`，Phase 1–3 计划见
`docs/RESEARCH_PLAN.md`，当前 4-GPU 运行口径见 `docs/CLOUD_PHASE1_RUNBOOK.md`。

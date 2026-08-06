# Upstream references

该目录用于保存本研究依赖的上游参考实现。参考代码不属于运行时模块，`src/` 中的正式实现不得从这里直接导入。

WONN 官方仓库以 Git submodule 放在 `references/WONN/`，固定 commit 为
`62d7ac52dee8b864cb77faac019a3d7ea1c2f7ae`。父项目只记录该 upstream commit；
本地数据、checkpoint、缓存和未提交分析脚本不属于可复现基线，也不会随父项目提交。

新克隆项目后执行：

```bash
git submodule update --init --recursive
```

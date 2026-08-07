# Upstream references

该目录用于保存本研究依赖的上游参考实现。参考代码不属于运行时模块，`src/` 中的正式实现不得从这里直接导入。

WONN 仓库以 Git submodule 放在 `references/WONN/`。官方上游基点为
`62d7ac52dee8b864cb77faac019a3d7ea1c2f7ae`；父项目当前固定派生快照
`af3f468d631d8b5a7f3730ccca5b7e686c458da1`，其中增加了本地可视化工具。本地数据、checkpoint、
缓存和未提交分析脚本不属于可复现基线，也不会随父项目提交。

`af3f468` 尚未发布到 `.gitmodules` 指向的官方远程，因此全新 clone 当前无法获取该对象。在对外
共享仓库前，需要把派生提交推送到可访问的 fork 并更新 `.gitmodules`；完成后再执行：

```bash
git submodule update --init --recursive
```

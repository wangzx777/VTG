# VTG Research Workspace

本仓库保存完整的 VTG 研究工作区，包括论文原文、OCR 缓存、阅读产物、
Codex skills、实验项目和参考代码。`.venv`、训练数据、模型权重、checkpoint
和实验输出不进入 Git。

## 双机同步

Git 以提交为同步单位，不能在同一个普通 checkout 中只 pull
`project/` 和 `repos/`。Mac 与 4090 都使用完整仓库，日常通过职责范围减少冲突：

- Mac 主要维护 `literature/`、`.agents/` 和研究索引。
- 4090 主要维护 `project/`、`repos/` 和对应周日志。
- 两台机器每次开始修改前都执行 `git pull --rebase origin main`，完成验证后
  commit 并 push；另一台机器再 pull。
- PDF 与 `literature/extracted/` 会随 Git 同步，但未变化的大文件不会在每次
  pull 时重复下载。

4090 首次部署：

```bash
cd /home/jia/usr_wangzx
git clone --recurse-submodules git@github.com:wangzx777/VTG.git VTG
cd VTG
```

后续每次开始工作：

```bash
git status --short
git pull --rebase origin main
git submodule update --init --recursive
```

服务器完成代码任务后，只暂存本次实际改动。例如：

```bash
git add project repos Log
git commit -m "说明本次实验变更"
git push origin main
```

`repos/` 中的外部官方仓库优先作为 Git submodule 管理。若需要修改 submodule，
先把内部仓库提交并推送到有写权限的远程或 fork，再在本仓库提交新的 submodule
revision。用户自有实现与实验代码放在 `project/`。

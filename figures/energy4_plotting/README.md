# Energy4 / ExoMV-D 实验图绘制代码（p.u. 版本）

这里是三张实验图（24 小时概率预测区间、边缘 PDF、Pearson 相关矩阵）的绘图代码，
需要放回 `NsDiff-TimeXer-Exog` 代码库中使用：

- `energy_figures.py` → `src/experiments/energy_figures.py`（共享绘图模块）
- `replot_figures.py` → `scripts/Energy4Exog/replot_figures.py`（从 `figure_arrays.npz` 重绘，不用重跑模型）
- `NsDiff_energy_pu_figures.patch` → 对 `src/experiments/NsDiff_energy.py` 的修改（调用新模块、保存 `figure_arrays.npz` 与 `pu_base.json`）

图的格式：
- 四个子图标题：Electrical load / PV Power / Cooling load / Heating load
- 纵坐标：Value (p.u.)；横坐标：Forecast horizon (hour)
- p.u. 基准值 = 各变量在训练集上的峰值，可用 `--base Electricity=... PV=...` 覆盖

重跑实验（`bash scripts/Energy4Exog/run_exomvd_3seeds.sh`）后，每个 `seed_k/` 目录下会直接生成新格式的三张图；
也可以 `python3 scripts/Energy4Exog/replot_figures.py --npz .../seed_1/figure_arrays.npz --out figs --dpi 300`。

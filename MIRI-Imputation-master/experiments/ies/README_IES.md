# MIRI-IES 多变量缺失填补实验

基于 MIRI（Missing data Imputation by Reducing mutual Information with rectified
flows, NeurIPS 2025）的 rectified-flow 填补器，构建了一个面向综合能源系统
（Integrated Energy System, IES）的 8 变量小时级多变量缺失填补 benchmark。

参考 MIRI 原文的 UCI 表格实验设计：**先删除真实缺失段，再在完整数据上人为模拟
MCAR / MAR / MNAR 三类缺失**，比较 MIRI 与 GAIN、HyperImpute、MICE、MissForest、
KNN、Mean 等方法的填补性能。

## 8 个变量

```
[Electricity] Electricity load (kW)          (idx 0)
[Electricity] Cooling load (kW)              (idx 1)
[Electricity] Heating load (kW)              (idx 2)
[Power] Solar energy generation (kW)         (idx 3)
[Weather] Horizontal solar irradition (W)    (idx 4)
[Weather] Outdoor air temperature (℃)        (idx 5)
[Weather] Outdoor air humidity (%)           (idx 6)
[Weather] Wind speed (m/s)                   (idx 7)
```

`Date` 列仅用于排序、删除 2011-03 真实缺失段、以及构造 (month, hour) 分层标签，
不作为模型输入变量。

## 目录结构

```
MIRI-Imputation-master/
  envs/miri_ies.yml                    # conda 环境
  src/imputer.py                       # 已做兼容性修改（见下）
  experiments/ies/
    configs/ies_default.yaml           # 实验配置
    prepare_ies.py                     # 读取/清洗/分层划分/标准化
    masks.py                           # MCAR / MAR / MNAR 掩码
    models.py                          # 8 维 IES 专用 MLP 速度场
    baselines.py                       # mean/knn/mice/missforest/gain/hyperimpute/miri
    metrics.py                         # MAE/RMSE/MMD/joint MMD/corr error
    run_ies.py                         # 主运行脚本（单次 / --run-all）
    aggregate_results.py               # 汇总为 mean ± std 表
    plot_results.py                    # 生成 4 类报告图
  results/ies/{raw,imputations,summaries,checkpoints,figures}/
```

## 对 MIRI 源码的改动

只对 `src/imputer.py` 的 `rectified_impute` 做了最小兼容性修改：

- 新增参数 `lr`（学习率）、`estimate_mi`（是否每轮估计 MINE）、
  `checkpoint_path`（checkpoint 保存路径，`None` 时不保存）。
- `estimate_mi=False` 时跳过 MINE，`mi` 记为 `nan`（MINE 不是本实验主指标，且会
  显著拖慢多方法/多机制/多种子的完整扫描）。
- checkpoint 仅在显式提供 `checkpoint_path` 时保存，避免多任务互相覆盖工作目录。

原始 demo（`examples/demo_UCI.ipynb` 等）默认参数不变，仍可运行。

## 实验协议：train complete + test masked

MIRI 本质是 transductive 填补，因此采用拼接协议保证各方法公平比较：

- 训练集完整观测（`M = 1`）；
- 测试集按 MCAR/MAR/MNAR 人为遮蔽（`M` 生成）；
- 拼接后对整个矩阵做 `fit_transform` / MIRI impute；
- **只在测试集人为缺失位置**评价。

所有方法都能从完整训练集学习 8 变量联合分布，但测试集被遮蔽位置的真值不会作为
观测暴露给填补器。掩码约定 `M = 1` observed，`M = 0` missing。

## 环境安装

```bash
cd MIRI-Imputation-master
conda env create -f envs/miri_ies.yml
conda activate miri-ies
export CUDA_VISIBLE_DEVICES=0        # 固定使用物理 GPU 0
```

数据文件已放在 `data/processed_data.xlsx`（配置默认路径）。如需自定义可用
`--data` 覆盖。

## 运行

Smoke test（Mean baseline）：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ies/run_ies.py \
  --config experiments/ies/configs/ies_default.yaml \
  --method mean --mechanism mcar --missing-rate 0.2 --seed 0
```

MIRI 小规模测试：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ies/run_ies.py \
  --config experiments/ies/configs/ies_default.yaml \
  --method miri --mechanism mcar --missing-rate 0.2 --seed 0 \
  --miri-max-rounds 2 --miri-max-epochs 5 --miri-ode-steps 10
```

完整扫描（config 中的全部 methods × mechanisms × rates × seeds，已完成的自动跳过）：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ies/run_ies.py \
  --config experiments/ies/configs/ies_default.yaml --run-all
```

汇总与绘图：

```bash
python experiments/ies/aggregate_results.py \
  --input results/ies/raw --output results/ies/summaries

python experiments/ies/plot_results.py \
  --summary results/ies/summaries/summary_mean_std.csv \
  --raw-dir results/ies/raw \
  --imputation-dir results/ies/imputations \
  --output-dir results/ies/figures
```

## 评价指标

| 指标 | 含义 | 方向 |
|------|------|------|
| `mae_std`, `rmse_std` | 标准化尺度点误差（仅缺失位置） | 越低越好 |
| `masked_mmd` | 逐变量缺失值分布 MMD 的均值 | 越低越好 |
| `joint_mmd` | 8 维联合分布 RBF-MMD | 越低越好 |
| `corr_error` | 相关矩阵 Frobenius 误差 | 越低越好 |
| `*_corr_error` | 关键物理对（PV-辐照、温度-冷/热负荷等）相关系数误差 | 越低越好 |

同时保存 per-variable MAE/RMSE，可反标准化回物理单位。

## 输出

- `results/ies/raw/{method}_{mechanism}_r{rate}_seed{seed}.json` —— 每次实验结果；
  baseline 因依赖缺失而失败时记录 `status="failed"` + 错误信息，不中断整个扫描。
- `results/ies/imputations/{...}.npz` —— `X_test_true / X_test_imp / M_test /
  mean / std / feature_cols / ...`，供绘图脚本直接读取。
- `results/ies/summaries/*.csv` —— `mean ± std` 汇总与排名表。
- `results/ies/figures/*.png` —— 4 类报告图（见下）。

## 报告图

- **Figure 1** `bar_mmd_mcar.png` —— MCAR 20/40/60 下各方法 joint MMD 柱状图。
- **Figure 2** `corr_matrix_triptych.png` —— 真实 / MIRI / 最佳 baseline 相关矩阵三联图
  （默认 MCAR 40%, seed 0；最佳 baseline = joint_mmd 最低的非 MIRI 方法）。
- **Figure 3** `scatter_pv_irradiation_compare.png` —— 光伏 vs 水平辐照散点对比。
- **Figure 4** `scatter_temp_load_compare.png` —— 温度 vs 冷/热负荷散点对比。

## 结论的正确表述

本实验不应表述为"预测未来能耗"，而应表述为：**在已知同一小时部分变量的情况下，
恢复缺失变量，并保持 IES 多变量联合分布。** MIRI 不一定在所有机制下取得最低
RMSE，但预期在 `masked_mmd`、`joint_mmd`、`corr_error` 等分布/联合关系指标上更强，
这与其关注条件分布填补与分布保真度的设计目标一致。

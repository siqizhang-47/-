# 完整复现命令序列（从 xlsx 到全部结果表）

所有命令在仓库根目录 `lsa-ccdm/` 下执行。默认设备 `cuda:2`（`configs/base.yaml`），
全流程只用一个随机种子 2026。

## 0. 环境与数据

```bash
pip install -r requirements.txt
cp aligned_energy_weather_summary_with_GHI_and_correlation.xlsx data/raw/
python scripts/prepare_data.py          # -> data/processed/energy.csv + 清洗报告
pytest tests/ -x -q                     # 验收测试（此时 data/backbone 相关自动跳过→通过后全绿）
```

## 1. 烟测（强烈建议先跑）

```bash
bash scripts/smoke_test.sh              # 2020年1月 + M=20 迷你全流程
```

## 2. 主干训练（Phase 1，GPU）

```bash
python experiments/run_backbone_train.py                                   # wnorm_on 主干（带早停）
python experiments/run_backbone_train.py --override use_window_norm=false  # wnorm_off 主干（实验5）
pytest tests/ -q -m slow                # GHI 条件方向性烟测（需已训练 checkpoint）
```

## 3. 冻结部署 + 场景缓存（Phase 2，GPU，最贵一步，支持断点续跑）

```bash
python experiments/run_deploy.py                                   # -> data/scenarios/wnorm_on
python experiments/run_deploy.py --override use_window_norm=false  # -> data/scenarios/wnorm_off
pytest tests/test_scenarios.py -q       # 缓存抽查验收
```

## 4. 漂移诊断（Exp 1，CPU）

```bash
python experiments/run_drift_diagnosis.py           # -> results/drift/*.csv
```

## 5. 超参粗扫（可选，只允许用 2020 上半年）

```bash
python experiments/run_adapter.py --method proposed --tune --override lam_s=0.01  --name tune_lams_x0.1
python experiments/run_adapter.py --method proposed --tune --override lam_s=1.0   --name tune_lams_x10
python experiments/run_adapter.py --method proposed --tune --override lam_delta=1e-4 --name tune_lamd_x0.1
python experiments/run_adapter.py --method proposed --tune --override lam_delta=1e-2 --name tune_lamd_x10
```

## 6. 主实验（Exp 2/3，CPU 分钟级/方法）

```bash
python experiments/run_retrain_baselines.py --mode retrain    # 基线2（GPU，3 次重训+采样）
python experiments/run_retrain_baselines.py --mode finetune   # 基线3（GPU）
bash scripts/exp3_all_methods.sh        # 8 方法全对比 + summary_table.csv（含 DM 检验）
```

## 7. 依赖保持（Exp 4）

```bash
python experiments/run_adapter.py --method frozen      --eval-dependence --name exp4_frozen
python experiments/run_adapter.py --method independent --eval-dependence --name exp4_independent
python experiments/run_adapter.py --method proposed    --eval-dependence --name exp4_proposed
```

## 8. 窗口归一化 2×2（Exp 5）与消融

```bash
bash scripts/exp5_wnorm_2x2.sh          # 实验5
bash scripts/abl_E_K_sweep.sh           # 消融 E（K）+ F（适配频率）
bash scripts/abl_HDI.sh                 # 消融 D（尺度共享）+ H（Δ结构）+ I（目标函数）
```

## 9. 汇总

```bash
python experiments/make_summary.py      # -> results/summary_table.csv
```

## 10. 模型图（真实 vs 生成：逐日曲线 / 相关矩阵热力图 / PDF）

先带 `--save-adapted` 重放需要画图的方法（保存适配后场景，frozen 不需要）：

```bash
python experiments/run_adapter.py --method proposed --save-adapted
python experiments/run_adapter.py --method cosa     --save-adapted
```

再画图（每个模型 3 张 + 一张多模型对比，默认输出 `results/figures/`）：

```bash
python scripts/plot_model_figures.py --models frozen proposed cosa \
    --day 2021-06-15 --start 2021-06-01 --end 2021-06-30
```

`--day` 决定逐日曲线图的日期，`--start/--end` 决定相关矩阵与 PDF 的统计时段；
`--no-band` 可隐藏 90% 区间带；`--models` 也可以直接给 results 目录路径。

各方法结果：`results/<name>/daily_metrics.parquet`（逐日全指标 + s_c/Δ_c 轨迹）、
`yearly_metrics.csv`、`monthly_metrics.csv`；成本报告：`results/{retrain,finetune}_cost.json`
与各缓存目录 `manifest.json`（逐日采样耗时）。

# HEEW 概率负荷预测实验代码库

基于 NsDiff（ICML 2025）的多能负荷概率预测实验，HEEW 数据集（2014–2022，小时级，78,888 行）。

- 3 个预测目标：Electricity / Cooling / Heat
- 外生条件：Temperature、Dew Point、Humidity + 7 个日历特征（**预测区间内天气使用真实观测，oracle 设定，所有模型一致**）
- 模型：**NsDiff**（天气条件版）、**D3U**、**WaveStitch**、**DeepVAR**
- 输出：Table I（各变量 MAPE / AW / PICP，最优加粗、次优下划线）+ figure1 负荷预测曲线 + figure2 真实/生成相关矩阵 + figure3 真实/预测分布
- 默认 GPU 编号 **0**（`GPU=1 bash scripts/...` 可覆盖），全流程 tqdm 进度条

## 1. 环境配置

```bash
cd zg-nsdiff-lowcarbon
conda create -n heew python=3.10 -y && conda activate heew
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 按你的 CUDA 版本选
pip install -r requirements.txt
```

把数据放到：

```bash
cp /path/to/HEEW数据.xlsx data/HEEW.xlsx
```

## 2. 一键运行

```bash
bash scripts/run_all_models.sh   # 数据准备 + DeepVAR/D3U/WaveStitch/NsDiff 训练与测试导出
bash scripts/build_results.sh    # Table I + figure1/2/3 + experiment_results.md
```

## 3. 分步运行

```bash
bash scripts/prepare_heew.sh     # Excel -> npy + 70/10/20 划分 + 训练集统计
bash scripts/run_deepvar.sh
bash scripts/run_d3u.sh
bash scripts/run_wavestitch.sh
bash scripts/run_nsdiff.sh       # F/G 预训练 -> 联合训练 -> 测试导出
bash scripts/build_results.sh
```

导出中断补跑（不重训）：所有入口都支持 `--skip_train`，例如：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m src.baselines.deepvar_adapter \
    --config configs/deepvar_heew.yaml --skip_train --seeds 1
```

## 4. 测试

```bash
PYTHONPATH=. pytest tests/ -q
```

## 5. 输出

```
results/table1.md                Table I（best 加粗，second-best 下划线）
results/experiment_results.md    Table I + 三张图的汇总文档
figures/figure1_load_prediction_curve.png
figures/figure2_correlation_matrices.png
figures/figure3_load_distribution.png
results/per_seed_metrics.csv / summary_metrics.csv
artifacts/predictions/<model>/seed_1/   统一 NPZ shard（samples [N,24,3,1000] 物理单位）
```

## 6. 关键设定

- 时间顺序划分 70/10/20；scaler 只用训练集拟合；目标与天气均 z-score
- 所有模型使用完全相同的窗口、真值与 oracle 未来天气；评估前做逐元素对齐断言
- PICP 为 95% 中心区间（0.025–0.975 分位）计数式累计；AW 为区间平均宽度（物理单位）
- 训练期间只 train+val（val NCRPS / NLL 选最佳 checkpoint），测试只在训练结束后跑一次（1000 samples，分 shard 落盘）
- `configs/heew_common.yaml` 的 `test_stride: 24`（每天一个不重叠 24h 预测起点）与 `val_max_batches: 100` 控制运行时长；改为 `1`/`null` 即为全量

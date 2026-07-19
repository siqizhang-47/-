# ZG-NsDiff 低碳社区实验代码库

基于 NsDiff（ICML 2025）的零膨胀门控扩散概率预测实验，按《ZG-NsDiff 实验代码实现与结果输出方案 V2》实现。

- 4 个预测目标：electricity / cooling / heating / pv（后三者零膨胀）
- 主实验设定：**预测区间内的天气为真实观测（oracle future weather）**，所有模型使用完全相同的信息集
- 模型：NsDiff（公平天气条件版）、ZG-NsDiff、D3U、WaveStitch
- 默认 GPU 编号 **6**（`GPU=0 bash scripts/...` 可覆盖），全流程 tqdm 进度条

## 0. 环境

```bash
cd zg-nsdiff-lowcarbon
conda create -n zg-nsdiff python=3.10 -y && conda activate zg-nsdiff
pip install -r requirements.txt
pip freeze > artifacts_pip_freeze.txt   # 训练前记录环境
```

把数据放到 `data/processed_data.xlsx`（Merged 工作表）。

## 1. 一键运行（完整实验）

```bash
bash scripts/run_all_models.sh    # 数据准备 + 4 个模型 × 3 seeds 训练与测试导出
bash scripts/build_results.sh     # 统一评估 + 表格 + 图片 + experiment_results.md
```

## 2. 分步运行

```bash
# 数据准备（Excel -> npy/npz + 划分 + 统计 + 审计）
bash scripts/prepare_low_carbon.sh

# NsDiff：F/G 预训练 -> 联合训练 -> 测试导出
bash scripts/pretrain_nsdiff_low_carbon.sh
bash scripts/run_nsdiff_low_carbon.sh

# ZG-NsDiff：F+Gate/G 预训练 -> 联合训练 -> Bernoulli gate 采样测试导出
bash scripts/run_zg_nsdiff_low_carbon.sh

# 基线
bash scripts/run_d3u_low_carbon.sh
bash scripts/run_wavestitch_low_carbon.sh

# 消融（gate-only / mask-only / deterministic gate）
bash scripts/run_ablations.sh

# 结果
bash scripts/build_results.sh
```

单卡指定与种子控制：

```bash
GPU=6 SEEDS="1 2 3" bash scripts/run_zg_nsdiff_low_carbon.sh
# 或直接：
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m src.experiments.ZGNsDiffLowCarbon \
    --config configs/zg_nsdiff_low_carbon.yaml --pretrain --seeds 1 2 3
```

## 3. 测试

```bash
PYTHONPATH=. pytest tests/ -q
```

## 4. 输出

```
artifacts/data/low_carbon/        预处理产物 + split + 统计
artifacts/runs/<model>/seed_k/    原子 checkpoint（单文件含全部模块/优化状态/RNG）
artifacts/predictions/<model>/seed_k/  统一 NPZ shard（samples [N,H,4,S] 物理单位）
results/per_seed_metrics.csv      每 seed 全指标
results/summary_metrics.csv       mean ± std
results/tables/*.md               论文主表 / 联合指标表 / 零状态表
results/efficiency.csv            参数量 / 训练时间 / 显存 / 采样速度
results/experiment_results.md     论文实验结果文档（数值全部自动写入）
figures/*.png                     相关性 / 分布 / 区间 / 校准 / gate reliability
```

## 5. 关键实现说明

- 原 NsDiff 扩散公式完整保留（`src/models/diffusion_utils.py`），只增加数值稳定 clamp（对所有模型同时启用）；
- 原仓库问题已修复：PICP 计数式累计 + 0.025/0.975 分位、CRPS 排序向量化、去掉每 epoch 测试、去掉多进程 metric 副本；
- 未来真实天气进入均值网络（decoder condition）、方差网络（condition MLP 增量）、去噪网络（history+future context），并有单元测试保证三条路径真实生效（`tests/test_exogenous_conditioning.py`）；
- NsDiff 与 ZG-NsDiff 唯一差异 = occurrence head + active mask + Bernoulli gate；
- scaler / 正值尺度 / 类别权重只用训练集拟合；缺失窗口整体删除，缺失不当零值；
- 训练期间只 train+val，最佳 validation NCRPS checkpoint 训练结束后测试一次（1000 samples，分 shard 落盘）。

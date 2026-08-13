# LSA-CCDM

长期分布漂移下综合能源联合概率预测 + 位置–尺度仿射测试时适配（Location–Scale Adaptive CCDM）。

基于 **CCDM**（扩散主干，改造为"4 目标通道 + 未来天气条件"）与 **COSA**（TTA 工程模式参考）实现，
完整方案见研究方案 v4 与实现方案文档。

## 关键工程特性（按需求实现）

1. **进度条**：训练（epoch/batch/验证）、部署采样（逐日）、adapter 重放（逐日）全部使用 tqdm；
2. **早停机制**：主干训练每 `eval_every` 个 epoch 在验证集上评估采样 CRPS，
   `patience` 次评估无改进即停止（`min_epochs` 保底），并只保留最优 checkpoint；
3. **单一随机种子**：全仓库只有一个种子 `seed: 2026`（`configs/base.yaml`），
   由 `utils/seed.py::set_seed` 在每个入口脚本调用一次，除此之外无任何其他随机种子；
4. **GPU 编号 2**：`configs/base.yaml` 中 `device: cuda:2`（无 GPU 自动回退 CPU，
   可用 `--override device=cuda:0` 修改）。

## 目录结构

```
configs/          # base / backbone / deploy / exp 配置
data/raw          # 放入 aligned_energy_weather_summary_with_GHI_and_correlation.xlsx
backbone/         # 改造自 CCDM：data_loader / network / diffusion / model
adapters/         # context 特征、LSA（核心新代码）、COSA 原版移植、TAFAS 移植
deployment/       # rolling（冻结部署+场景缓存）、replay（adapter 实验重放器）
evaluation/       # 全部指标、DM 检验、漂移诊断
experiments/      # 各阶段入口脚本
scripts/          # 数据准备 + 一键实验 shell 脚本
tests/            # pytest 验收测试（Phase 0–5）
```

## 两条核心工程原则

- **场景缓存与重放**：冻结主干采样只跑一次（`run_deploy.py`），所有 adapter/基线/消融
  实验共享同一份 `data/scenarios/*/*.npz` 缓存，单个 adapter 实验分钟级完成；
- **严格无前视泄漏**：部署日 t 的适配参数只由 ≤ t−1 日真值训练，当日指标记录先于
  当日一切更新（`tests/test_adapter.py::test_no_leakage` 用调用顺序断言验证）。

## 快速开始

完整命令序列见 [REPRODUCE.md](REPRODUCE.md)。

```bash
pip install -r requirements.txt
cp <你的数据>.xlsx data/raw/aligned_energy_weather_summary_with_GHI_and_correlation.xlsx
pytest tests/ -x -q                    # 不依赖数据的单元测试先行
python scripts/prepare_data.py         # Phase 0
bash scripts/smoke_test.sh             # 迷你端到端烟测（2020年1月 + M=20）
```

## 实现说明（移植决策）

- **COSA 原版基线**：严格按源码配置（缓冲目标均值上下文、`tanh(g)` 门控线性层、MSE 目标、
  VAR_WISE_GATING），修正量作用于场景均值后对所有场景共享平移——COSA 是点预测修正器，
  共享平移是对生成模型的忠实提升方式；
- **TAFAS 基线**：门控校准模块的逐日粒度移植，同样以共享平移作用于场景集合；
- **消融 J（逐场景独立适配）**：以逐场景 MSE 目标实现"每条场景独立拟合真值"的对照组
  （方案中 μ^(m)=自身 的字面形式会使尺度项失效，代码实现取其本意），预期离散度收缩、
  校准崩坏，作为 sanity check；
- **adapter 优化空间**：重放器在训练期 StandardScaler 归一化空间内做在线优化（数值良态），
  指标全部在原始量纲计算；正仿射与逐载体线性缩放可交换，两者等价。

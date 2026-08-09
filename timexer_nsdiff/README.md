# TimeXer 条件非平稳扩散模型（TimeXer-conditioned NsDiff）

按《TimeXer 外生条件编码器接入 NsDiff 的四能源变量概率预测方案 v3》实现的完整实验代码库。
四个能源变量 `Electricity / PV / Cooling / Heat` 联合概率预测，未来日历与天气作为外生条件。

- 进度条：每个训练阶段、验证、扩散采样都有 tqdm 进度条
- 早停：每个阶段独立早停器 + 最优权重回滚
- 随机种子：固定 3 个（`seeds: [1, 2, 3]`）
- GPU：默认 `--gpu 2`
- 模型代码：TimeXer 与 NsDiff 的核心模块直接取自你提供的两个压缩包（见下方"代码来源"）

---

## 1. 快速开始

```bash
# 0) 环境
pip install -r requirements.txt

# 1) 放数据（Excel 里必须有 Aligned_Data 表）
cp /path/to/HEEW数据集.xlsx  data/HEEW.xlsx

# 2) 冒烟测试（每阶段 1 epoch，约 2 分钟，验证全链路）
bash scripts/smoke_test.sh

# 3) 主模型 A7，3 个种子，GPU 2
python run.py --ablation A7 --gpu 2

# 4) 全部 6 组消融 + 汇总表
bash scripts/run_all.sh
```

结果目录：

```
results/<消融编号>/seed<k>/
    config.json          本次运行的完整配置
    train.log            全部日志
    history.json         各阶段 train/val 曲线
    stage{1,2,3,4}_*.pth 各阶段最优检查点
    final_model.pth      最终权重
    metrics_test.json    测试集全部指标
    attention.json/.png  [4, 16] 交叉注意力热力图
results/<消融编号>/summary.json     3 个种子的 mean ± std
results/ablation_table.md / .csv    全部消融对比表
```

---

## 2. 终端运行指令

### 2.1 常用命令

```bash
# 主模型（A7），3 种子，GPU 2 —— 论文主结果
python run.py --ablation A7 --gpu 2

# 单个种子
python run.py --ablation A7 --gpu 2 --seeds 1

# 指定 3 个种子（默认就是 1 2 3）
python run.py --ablation A7 --gpu 2 --seeds 1 2 3

# 基线：原始 NsDiff（无外生变量）
python run.py --ablation A0 --gpu 2

# 共享 global token vs 四目标专属 token（方案里最关键的对比）
python run.py --ablation A3 --gpu 2
python run.py --ablation A4 --gpu 2

# 条件注入位置消融
python run.py --ablation A5 --gpu 2      # 只接 f_phi
python run.py --ablation A6 --gpu 2      # f_phi + g_psi
python run.py --ablation A7 --gpu 2      # f_phi + g_psi + denoiser

# 六组消融一次跑完并生成对比表
bash scripts/run_all.sh
# 或指定 GPU：GPU=2 bash scripts/run_all.sh

# 汇总已有结果为 markdown / csv 表格
python aggregate_ablations.py --results_dir results

# 只评估（复用已训练好的 final_model.pth）
python run.py --ablation A7 --gpu 2 --seeds 1 --eval_only

# 只跑部分阶段（阶段名：mean,scale,joint,diffusion,finetune）
python run.py --ablation A7 --gpu 2 --stages mean,scale
```

### 2.2 常用调参开关

```bash
# 显存不足：d_model 128 -> 64，采样分块调小
python run.py --ablation A7 --gpu 2 --d_model 64 --sample_chunk 10 --eval_batch_size 8

# 用验证集 CRPS 做扩散阶段早停（更贴近报告指标，但慢很多）
python run.py --ablation A7 --gpu 2 --diff_val_metric crps --val_subsample 512

# lambda_sigma 网格（方案 15.3 给的三档）
for L in 0.01 0.1 1.0; do
  python run.py --ablation A7 --gpu 2 --lambda_sigma $L --output_dir results_lam$L
done

# 打开第五阶段端到端微调（默认关闭）
python run.py --ablation A7 --gpu 2 --epochs_finetune 10 --lr_finetune 2e-5

# 全维 Variogram Score（96 维，较慢；默认是逐小时 4 变量的 cross_var）
python run.py --ablation A7 --gpu 2 --variogram_mode full

# 快速评估：测试集每 4 个窗口取 1 个
python run.py --ablation A7 --gpu 2 --test_stride 4

# CPU 调试
python run.py --ablation A7 --gpu -1 --num_workers 0
```

> `CUDA_VISIBLE_DEVICES` 与 `--gpu` 不要同时用。若已 `export CUDA_VISIBLE_DEVICES=2`，则改传 `--gpu 0`。

`configs/energy_timexer_nsdiff.yaml` 里的任何字段都能用同名 `--flag` 覆盖。

---

## 3. 目录结构

```
timexer_nsdiff/
├── run.py                        单次实验入口（一个消融 × 若干种子，四阶段 + 评估）
├── aggregate_ablations.py        汇总消融对比表
├── configs/energy_timexer_nsdiff.yaml
├── scripts/{run_all,run_main,smoke_test}.sh
└── src/
    ├── config.py                 ExpConfig + 消融表 A0/A3/A4/A5/A6/A7
    ├── data_provider/
    │   ├── data_quality_fixes.py GHI 时区平移、闰日缺口窗口过滤、0 值插值
    │   ├── holiday_features.py   holidays.US(subdiv='AZ')，预留 UA 校历 IsBreak
    │   ├── solar_clearsky.py     pvlib 图森 ClearskyGHI（含无 pvlib 的 Haurwitz 兜底）
    │   ├── scaler.py             逐列标准化，目标/天气分开
    │   └── energy_weather_dataset.py
    ├── models/
    │   ├── timexer_condition_encoder.py  CyclicCalendarEncoder / 外生 variate token /
    │   │                                 TimeXer 交叉注意力 / HorizonConditionAdapter
    │   ├── location_scale.py     逐时刻 f_phi 与 g_psi（独立参数）+ 高斯 NLL
    │   └── nsdiff_conditioned.py NsDiff 调度 + 扩散损失 + 联合场景采样
    ├── third_party/              ← 直接来自你给的两个压缩包
    │   ├── timexer/              SelfAttention_Family / Embed / EnEmbedding+Encoder
    │   └── nsdiff/               nsdiff_utils / denoise / sigma
    ├── metrics/prob_metrics.py   MAE RMSE sMAPE CRPS QICE PICP 区间宽度 ES VS + 分组指标
    └── engine/
        ├── trainer.py            四（五）阶段训练 + 进度条 + 早停
        └── evaluate.py           场景生成、指标、注意力导出
```

---

## 4. 代码来源（尽量复用压缩包原码）

| 本仓库文件 | 来源 | 改动 |
|---|---|---|
| `src/third_party/timexer/SelfAttention_Family.py` | `TimeXer-main/layers/SelfAttention_Family.py` | 逐字复制 `FullAttention`、`AttentionLayer`；原文件另有 `reformer_pytorch`/`einops` 依赖的变体，未使用故未引入 |
| `src/third_party/timexer/Embed.py` | `TimeXer-main/layers/Embed.py` | 逐字复制 `PositionalEmbedding` |
| `src/third_party/timexer/masking.py` | `TimeXer-main/utils/masking.py` | 逐字复制 `TriangularCausalMask` |
| `src/third_party/timexer/timexer_blocks.py` | `TimeXer-main/models/TimeXer.py` | `EnEmbedding` / `Encoder` / `EncoderLayer` 原样搬运，三处 `# [MOD]`：①可共享 global token（A3）②返回交叉注意力权重 ③`cross=None` 时退化为纯自注意力（A0）。原 `FlattenHead` 预测头按方案第 6 节要求不使用 |
| `src/third_party/nsdiff/nsdiff_utils.py` | `NsDiff-main/src/layer/nsdiff_utils.py` | 扩散推导完全保留；`# [MOD]`：①去噪器调用签名改为 `model(y_t, y_0_hat, gx, t, cond)` ②两处 `clamp` 数值保护（原码在 float32 下会对极小负数开方产生 NaN） |
| `src/third_party/nsdiff/denoise.py` | `NsDiff-main/src/layer/denoise.py` | `ConditionalLinear` 逐字复制；`ConditionalGuidedModel` 仅增加 `cond_dim` 输入通道（方案 14 节的"第一版直接拼接"）。`cond_dim=0` 时与原实现完全一致 |
| `src/third_party/nsdiff/sigma.py` | `NsDiff-main/src/utils/sigma.py` | 逐字复制 `wv_sigma` / `wv_sigma_trailing` |
| 扩散调度、训练损失 | `NsDiff-main/src/models/NsDiff.py`、`src/experiments/NsDiff.py::_process_train_batch` | `DiffusionSchedule` 与 `diffusion_loss` 按原式重写为不依赖 `torch_timeseries` 的形式，`betas_tilde / alphas_hat / gx_term / kl_loss` 公式一致 |

> 两个原仓库都依赖 `torch_timeseries`（NsDiff）与整套 `data_provider/exp` 框架（TimeXer）。本仓库把用到的模块单独 vendored 进来，因此 `pip install -r requirements.txt` 之后即可直接运行，不需要安装这两个包。

---

## 5. 与方案文档的对应关系

### 5.1 数据修正（第 2.1 节，全部已实现且已在本数据上验证）

| 问题 | 处理 | 实测验证 |
|---|---|---|
| GHI 存在 +7 小时时区错位 | `df["GHI"].shift(-7)`，丢弃尾部 7 行 | corr(GHI, PV) 由 **−0.205 → +0.937**，峰值小时由 19 → 12 |
| 闰日删除造成 2 处 24 小时缺口 | 滑窗掩码显式过滤跨缺口窗口 | 2016-02-28 23:00 与 2020-02-28 23:00 各检出 25h 跳变，**共丢弃 382 个窗口**（78642 → 78260） |
| 天气 0 值为缺失填充 | Temperature/Humidity 的精确 0 值、跳变异常的 Dew Point 0 值置 NaN 后按时间线性插值 | Temperature 15、Humidity 17、Dew Point 85 个，占天气单元格 **0.037%** |

### 5.2 张量维度（第 17 节，代码实测输出一致）

| 张量 | 形状 |
|---|---|
| `history_energy` | `[B, 168, 4]` |
| `future_calendar` | `[B, 24, 7]` |
| `future_weather` | `[B, 24, 5]` |
| 日历编码 | `[B, 24, 11]` |
| 全部外生变量 | `[B, 24, 16]` |
| 外生 variate tokens | `[B, 16, 128]` |
| `target_tokens` | `[B, 4, 128]` |
| `cond_horizon` | `[B, 24, 4, 128]` |
| `cond_denoiser` | `[B, 24, 512]` |
| `attention` | `[B, 8, 4, 16]`（A3 为 `[B, 8, 1, 16]`） |
| μ / σ | `[B, 24, 4]` |
| 场景 | `[B, 100, 24, 4]` |

### 5.3 外生 token 固定顺序（第 22 节）

```
0  YearTrend    4  DoYCos     8  WeekdayCos   12  DewPoint
1  MonthSin     5  HourSin    9  IsWeekend    13  Humidity
2  MonthCos     6  HourCos   10  IsHoliday    14  GHI
3  DoYSin       7  WeekdaySin 11 Temperature  15  ClearskyGHI
```

### 5.4 五个训练阶段（第 15 节）

| 阶段 | 训练对象 | 损失 | 学习率 | 早停 |
|---|---|---|---|---|
| 1 | condition encoder + horizon adapter + f_phi | MSE / Huber | 1e-3, wd 1e-4 | patience 8，监控 val MSE |
| 2 | g_psi（均值 detach，前 5 epoch 冻结 encoder） | 高斯 NLL | head 5e-4 / encoder 1e-4 | patience 8，监控 val NLL |
| 3 | 位置—尺度联合微调 | `L_mu + 0.1 · L_sigma` | 3e-4 | patience 8 |
| 4 | NsDiff 去噪器（条件模块冻结） | NsDiff 原始扩散损失 | 2e-4 | patience 8，监控 val loss（可切 CRPS） |
| 5 | 可选端到端微调（默认关闭，历史 Transformer 保持冻结） | 扩散损失 | 2e-5 | patience 5 |

### 5.5 消融表（第 20 节）

| 编号 | 配置 | `use_exog` | `shared_global_token` | cond → f_phi / g_psi / denoiser |
|---|---|---|---|---|
| A0 | 原始 NsDiff，仅历史 | ✗ | – | ✗ / ✗ / ✗ |
| A3 | TimeXer，单个共享 global token | ✓ | ✓ | ✓ / ✓ / ✓ |
| A4 | TimeXer，四目标专属 global token | ✓ | ✗ | ✓ / ✓ / ✓ |
| A5 | A4，只接 f_phi | ✓ | ✗ | ✓ / ✗ / ✗ |
| A6 | A4，接 f_phi + g_psi | ✓ | ✗ | ✓ / ✓ / ✗ |
| A7 | A4，接 f_phi + g_psi + denoiser | ✓ | ✗ | ✓ / ✓ / ✓ |

两点实现说明：

1. **A4 与 A7 的开关组合相同**（方案表格里 A4 行未指定注入位置，A5–A7 才逐级指定）。两者仍作为独立条目分别记录，方便直接引用；若只想省一次训练，跑 A7 即可。
2. **A5 / A6 中未接受条件的模块不会"少一路输入"**：它们改吃一条同形状的"仅历史"条件（由一个纯自注意力的 endogenous encoder 副本 + 无外生的 horizon adapter 生成）。否则消融会同时变成容量消融，A5/A6 的参数量因此略高于 A4/A7，这是有意为之。

### 5.6 常见实现错误自查（第 24 节）

| 条目 | 落实位置 |
|---|---|
| 24.1 未来能源不进 condition encoder | `TimeXerExogenousConditionEncoder.forward` 只接收 history / calendar / weather |
| 24.2 目标与天气使用不同 scaler，且逐列 | `ColumnStandardScaler`，`target_scaler` 与 `weather_scaler` 分离 |
| 24.3 Year 不做类别 embedding | `YearNorm = (Year-2014)/8` |
| 24.4 天气保留完整 24 小时轨迹 | variate token = `Linear(24 → 128)`，无时间池化 |
| 24.5 四目标联合生成 | `sample()` 单条反向轨迹同时产出 4 个变量 |
| 24.6 Oracle Weather 标注 | 每个 `metrics_*.json` 都写入 `"weather_setting": "OracleWeather"` |
| 24.7 不提前平均四个 token | 仅 A3 显式共享，A4–A7 保持四个独立 query |
| 24.8 尺度必须为正 | `Softplus(·) + 1e-4` |
| 24.9 f_phi/g_psi 使用逐时刻条件 | 输入 `u_{h,k} = [g_k'; C_{h,k}]`，`C_{h,k}` 含 `E_h` 与 `R_h` |
| 24.10 二值标志不标准化 | `IsWeekend`/`IsHoliday` 由 `CyclicCalendarEncoder` 直通 |
| 24.11 GHI 时区修正 | `fix_ghi_timezone`，在标准化统计量之前执行 |
| 24.12 滑窗不跨闰日缺口 | `contiguous_window_mask` |

---

## 6. 需要留意的几点

1. **外生变量数按 16 实现。** 方案 5.3 节算式写 `M_exo = 11 + 4 = 15`，但第 8、17 节的张量表、第 9 节的注意力形状 `[B,8,4,16]` 都按 16 给出；16 = 11 日历 + 4 天气 + ClearskyGHI 才自洽（第 22 节的 15 项清单漏了 ClearskyGHI）。这里取 16，`--n_exo` 可改。
2. **`future_calendar` 的第 3 个通道是 day-of-year，不是 day-of-month。** 方案 5.3 明确要求用 DoY 编码，日期号本身不进模型，所以数据集直接输出闰日校正后的 DoY，7 个通道仍为"5 原始日历 + IsWeekend + IsHoliday"。
3. **所有结果都是 Oracle Weather。** 模型读取真实未来天气，是性能上界，不能表述为真实运行条件下的预测性能。
4. **Variogram Score 默认 `cross_var`。** 即在每个预测小时上对 4 个目标算 4×4 变差，再沿 24 小时平均，用来衡量能源变量之间的联合结构。`--variogram_mode full` 会改用完整 96 维（约 9216 对/窗口，慢很多）。
5. **`IsBreak`（UA 校历）按方案要求首版未启用**，接口已留在 `HolidayFeatureBuilder.is_break`，传入 `use_break=True` 与 `break_ranges` 即可开启。
6. **`holidays` / `pvlib` 缺失时有兜底实现**（内置美国联邦节假日表、Haurwitz 晴空模型），但会打印警告；论文实验请安装这两个包，以保证与方案口径一致。

---

## 7. 参考文献

1. Wang et al. *TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables.* NeurIPS 2024. https://arxiv.org/abs/2402.19072
2. Ye, Xu & Gui. *Non-stationary Diffusion for Probabilistic Time Series Forecasting.* ICML 2025. https://arxiv.org/abs/2505.04278
3. Nie et al. *A Time Series Is Worth 64 Words.* ICLR 2023. https://arxiv.org/abs/2211.14730

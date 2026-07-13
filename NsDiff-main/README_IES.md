# NsDiff 四变量联合概率预测(Oracle 天气版)

在原版 `NsDiff-main` 上做 **Electricity / Cooling / Heating / PV** 四变量联合概率预测。
`f` 保留 NS-Transformer,未来天气用实测值(Oracle),**扩散核心数学一行不改**(全部复用 `src/layer/nsdiff_utils.py`)。

本目录 `ies/` 是完整实现;`src/`、`configs/nsdiff.yml` 为原版保留不动。

---

## 目录结构

```
NsDiff-main/
├─ ies/                         # ← 本方案全部代码
│  ├─ data.py                   # 数据集 + dataloader + scaler(仅训练段 fit)
│  ├─ weather_embedding.py      # WeatherDataEmbedding(天气经 mark 注入)
│  ├─ mu_weather.py             # f:封装 NS-Transformer + 换天气版 embedding
│  ├─ g_weather.py              # g:天气感知方差估计器 + 滑窗方差监督目标
│  ├─ hurdle.py                 # 零膨胀 ZeroHead + 非负/夜间 PV=0 约束
│  ├─ build_model.py            # 组装 NsDiff + f + g + hurdle
│  ├─ metrics.py                # CRPS/QICE/EnergyScore/Variogram/PSD/ACF/APD/AvgStd/DTW/MAE/RMSE/CorrErr
│  ├─ plots.py                  # 图1~4
│  └─ run.py                    # 训练/验证/测试/采样/保存(独立主程序)
├─ configs/
│  ├─ ies_oracle.yaml           # 主配置
│  ├─ ies_oracle_smoke.yaml     # 快速 sanity(1 epoch、少量 batch、num_samples=5)
│  ├─ ies_wo_weather.yaml       # 消融:w/o 天气
│  ├─ ies_wo_lsnm.yaml          # 消融:w/o LSNM(端点 N(f, I))
│  ├─ ies_wo_uans.yaml          # 消融:w/o UANS(完美估计器 p_sample_loop_pe)
│  └─ ies_wo_hurdle.yaml        # 消融:w/o Hurdle
├─ data/processed_data.xlsx     # 数据集(已放置)
└─ scripts/
   ├─ run_ies.sh                # 主实验(GPU 4)
   ├─ run_ies_smoke.sh          # 快速 sanity(GPU 4)
   └─ run_ies_ablation.sh       # 全套消融依次跑(GPU 4)
```

---

## 环境安装

原仓库依赖之外,本方案需要:

```bash
pip install -r requirements_ies.txt
# 或最小集:
# pip install torch torchvision torch_timeseries==0.1.10 pandas openpyxl scikit-learn matplotlib pyyaml
```

> `torch_timeseries` 提供 NS-Transformer 骨干(`DataEmbedding` / `Transformer_EncDec` / `SelfAttention_Family`),它间接依赖 `torchvision`。

---

## 运行(GPU 4)

所有脚本已固定 `CUDA_VISIBLE_DEVICES=4`,即把物理 4 号卡映射为进程内 `cuda:0`,再传 `--gpu 0`。

**第 0 步 —— 先跑快速 sanity(强烈建议):**

```bash
bash scripts/run_ies_smoke.sh
```

确认不报形状错(dec_mark 维 = label_len+O、mark 维 = 8、样本 (B,S,O,4)),几分钟即出一版指标/图。

**第 1 步 —— 主实验:**

```bash
bash scripts/run_ies.sh
```

**第 2 步 —— 全套消融(§13):**

```bash
bash scripts/run_ies_ablation.sh
```

也可手动单独跑某个配置:

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID
python3 -m ies.run --config configs/ies_oracle.yaml --gpu 0
```

> 若想直接用物理索引而不做屏蔽,可改为:不设 `CUDA_VISIBLE_DEVICES`,并传 `--gpu 4`(要求机器可见 ≥5 张卡)。

---

## 输出

每个 `output_dir`(默认 `outputs_ies_oracle/`)下:

- `tables/metrics.json` —— 全套指标(kW 分变量 + 标准化空间双口径)。
- `figures/fig1_real_vs_gen.png` —— 真值 vs 生成均值。
- `figures/fig2_corr.png` —— 变量相关矩阵(真实 vs 生成)。
- `figures/fig3_pdf.png` —— 各变量边缘分布直方图。
- `figures/fig4_scenarios.png` —— 源荷场景(多条采样 + 均值 ±5/95 分位)。
- `ckpt/best.pt` —— 按标准化 CRPS early-stop 存的最优权重。
- `test_samples.npz` —— `y_true (N,O,4)` 与 `samples (N,S,O,4)`(kW),便于二次分析。

---

## 消融一览(§13)

| 配置 | 改动 | 检验点 |
|---|---|---|
| `ies_oracle`     | Oracle天气 + LSNM + UANS + Hurdle | 主结果/上界 |
| `ies_wo_weather` | `ablation.no_weather=true`,天气全抹零、`weather_mlp=false` | 天气增益 |
| `ies_wo_lsnm`    | `ablation.no_lsnm=true`,端点 `N(f, I)`(g≡1、y_sigma≡1) | 复现论文变体 |
| `ies_wo_uans`    | `ablation.no_uans=true`,完美估计器,用 `p_sample_loop_pe` | 复现论文变体 |
| `ies_wo_hurdle`  | `hurdle.enable=false` | 零膨胀贡献 |

消融开关由 `run.py` 顶层 `ablation:` 配置块驱动(默认无该块 = 完整模型)。

---

## 关键实现要点(与原码核对)

1. **`enc_in = dec_in = c_out = 4`**:NS-Transformer 用 `x_enc` 自身 mean/std 反归一化,天气绝不进值通道。
2. **Oracle 未来天气经 `y_mark` 注入解码器**(值通道未来段被内部清零),由 `WeatherMeanF` 拼成 `dec_mark`。
3. **去噪器 `ConditionalGuidedModel` 只拼 `(y_t, f, g)`,不用 `enc_out`**;为统一 8 维 mark,`NsDiff.enc_embedding` 一并换成天气版。
4. **g 的方差监督用滑窗方差** `wv_sigma_trailing`(与原版 `_process_train_batch` 完全一致)。
5. **约束(非负 / 夜间 PV=0 / hurdle)在反标准化回 kW 之后施加**。
6. **打分双口径**:kW 分变量(工程 MAE/RMSE/CRPS)+ 标准化空间(QICE/联合指标)。
7. **采样分块** `sample_chunk` 避免 B×S×L 爆显存;early-stop 监控标准化 CRPS,测试前 load best。

---

## Oracle 定位(论文写法,二选一)

- **条件场景生成(推荐)**:给定一条天气场景生成对应源荷场景,Oracle 天气是任务设定的一部分,非泄漏(对应图4)。
- **运行时预测**:把 Oracle 明确标注为「完美天气信息上界」,并用 `ies_wo_weather` 一列作对照,展示天气增益空间。

---

## 已验证

已在 CPU 上用 `ies_oracle_smoke.yaml` 跑通完整链路(数据→建模→训练→扩散采样→全指标→出图),
并逐一验证 `w/o weather`、`w/o LSNM`、`w/o UANS`、`w/o hurdle` 四条消融代码路径均正常运行。
正式实验请按上面「运行(GPU 4)」在 GPU 机器上执行。

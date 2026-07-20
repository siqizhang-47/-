"""Assemble results/experiment_results.md following the user's result template:

  评价指标 (MAPE / PICP / PINAW formulas)
  图：变量相关性热力图
  图：真实分布与预测分布对比图
  图：真实值、预测值及预测区间对比图
  模型对比结果表

All numbers are read from results/summary_metrics.csv — nothing hand-written.

python -m src.evaluation.build_result_markdown \
    --metrics results/summary_metrics.csv --figures figures --output results/experiment_results.md
"""
import argparse
import os

import pandas as pd

from src.data.low_carbon_schema import TARGET_NAMES

MODEL_ORDER = ["nsdiff", "d3u", "wavestitch", "zg_nsdiff"]
MODEL_LABEL = {"nsdiff": "NsDiff", "d3u": "D3U", "wavestitch": "WaveStitch",
               "zg_nsdiff": "本文模型（ZG-NsDiff）"}
VAR_LABEL = {"electricity": "电负荷", "cooling": "冷负荷", "heating": "热负荷", "pv": "光伏出力"}

METRIC_SECTION = r"""### 评价指标：

MAPE用于衡量预测值与真实值之间的平均相对误差，计算公式为：

$$
\mathrm{MAPE}
=
\frac{1}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
\times 100\%
$$

其中，$n$ 为样本数量，$y_i$ 为第 $i$ 个样本的真实值，$\hat{y}_i$ 为对应的预测值。MAPE越小，说明模型的点预测精度越高。

PICP用于衡量真实观测值落入预测区间的比例，计算公式为：

$$
\mathrm{PICP}
=
\frac{1}{n}
\sum_{i=1}^{n} c_i
$$

其中：

$$
c_i=
\begin{cases}
1, & y_i\in[L_i,U_i] \\
0, & \text{其他}
\end{cases}
$$

$L_i$ 和 $U_i$ 分别为第 $i$ 个样本预测区间的下界和上界。PICP越接近设定的置信水平，说明预测区间的可靠性越高。

PINAW用于衡量预测区间的平均宽度，计算公式为：

$$
\mathrm{PINAW}
=
\frac{1}{n}
\sum_{i=1}^{n}
\frac{U_i-L_i}{y_{\max}-y_{\min}}
$$

其中，$y_{\max}$ 和 $y_{\min}$ 分别为样本中的最大值和最小值。PINAW越小，说明预测区间越紧凑。
"""


def fmt(mean, std, digits=4):
    if pd.notna(std) and std > 0:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"{mean:.{digits}f}"


def build_table(df: pd.DataFrame) -> str:
    models = [m for m in MODEL_ORDER if m in df["model"].values]
    models += [m for m in df["model"].values if m not in models]
    headers = []
    for name in TARGET_NAMES:
        v = VAR_LABEL[name]
        headers += [f"{v} MAPE", f"{v} PINAW", f"{v} PICP"]
    lines = ["### 模型对比结果表", "",
             "| 模型 | " + " | ".join(headers) + " |",
             "|---|" + "---:|" * len(headers)]
    for m in models:
        row = df.loc[df.model == m].iloc[0]
        cells = []
        for name in TARGET_NAMES:
            mape_key = "electricity_mape" if name == "electricity" else f"{name}_mape_pos"
            cells.append(fmt(row[f"{mape_key}_mean"], row.get(f"{mape_key}_std")))
            cells.append(fmt(row[f"{name}_pinaw_mean"], row.get(f"{name}_pinaw_std")))
            cells.append(fmt(row[f"{name}_picp_mean"], row.get(f"{name}_picp_std")))
        lines.append(f"| {MODEL_LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    lines += ["",
              "注：冷负荷、热负荷、光伏出力的 MAPE 在正值样本（$y_i>0$）上计算；"
              "PINAW 的 $y_{\\max}-y_{\\min}$ 取各变量测试集真值范围，所有模型共用；"
              "PICP 对应 95% 预测区间（0.025–0.975 分位）。"]
    return "\n".join(lines)


def fig_section(title, bullets, fig_path):
    lines = [f"#### 图：{title}", ""]
    lines += [f"- {b}" for b in bullets]
    lines += ["", f"![{title}]({fig_path})" if os.path.exists(fig_path)
              else f"_（figures 尚未生成：{fig_path}，运行 bash scripts/build_results.sh）_", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/summary_metrics.csv")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--output", default="results/experiment_results.md")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    var_list = "、".join(VAR_LABEL[n] for n in TARGET_NAMES)

    parts = ["# 实验结果", "", METRIC_SECTION, ""]

    parts.append(fig_section(
        "变量相关性热力图",
        [f"**横坐标**：参与分析的目标变量（{var_list}）。",
         "**纵坐标**：与横坐标相同的变量集合。",
         "**颜色条**：相关系数，取值范围为 $[-1,1]$。",
         "**图中数值**：任意两个变量之间的相关系数。",
         "**颜色含义**：正值表示正相关，负值表示负相关，颜色深浅表示相关程度的强弱。"],
        os.path.join(args.figures, "target_correlation_heatmap.png"),
    ))

    parts.append(fig_section(
        "真实分布与预测分布对比图",
        ["**横坐标**：目标变量取值，单位为 kW。",
         "**纵坐标**：概率密度。",
         "**平滑曲线**：真实样本和预测样本对应的KDE曲线。",
         "**子图设置**：子图(a)：电负荷；子图(b)：冷负荷；子图(c)：热负荷；子图(d)：光伏出力。"],
        os.path.join(args.figures, "distribution_comparison.png"),
    ))

    parts.append(fig_section(
        "真实值、预测值及预测区间对比图",
        ["**横坐标**：测试样本时间。",
         "**纵坐标**：对应目标变量的取值，单位为 kW。",
         "**曲线1**：真实值。",
         "**曲线2**：预测均值。",
         "**阴影区域**：95% 预测区间。",
         "**子图设置**：子图(a)：电负荷；子图(b)：冷负荷；子图(c)：热负荷；子图(d)：光伏出力。"],
        os.path.join(args.figures, "interval_comparison.png"),
    ))

    parts.append(build_table(df))
    parts.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    text = "\n".join(parts)
    assert "[AUTO]" not in text
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

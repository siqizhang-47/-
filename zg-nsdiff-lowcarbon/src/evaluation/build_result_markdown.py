"""Assemble results/experiment_results.md from computed CSVs and figures
(spec section 20).  All numbers come from the CSVs — nothing is hand-written.

python -m src.evaluation.build_result_markdown \
    --metrics results/summary_metrics.csv --figures figures --output results/experiment_results.md
"""
import argparse
import glob
import json
import os

import pandas as pd

LIMITATION_TEXT = (
    "未来天气使用真实观测，因此结果代表 oracle exogenous condition 下的性能；"
    "实际部署性能还会受到天气预报误差影响。"
)


def rel_change(a, b):
    return 100.0 * (a - b) / a if a not in (0, None) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/summary_metrics.csv")
    ap.add_argument("--per_seed", default="results/per_seed_metrics.csv")
    ap.add_argument("--tables_dir", default="results/tables")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--efficiency", default="results/efficiency.csv")
    ap.add_argument("--data_audit", default="artifacts/data/low_carbon/data_audit.json")
    ap.add_argument("--output", default="results/experiment_results.md")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics).set_index("model")
    audit = {}
    if os.path.exists(args.data_audit):
        with open(args.data_audit, encoding="utf-8") as f:
            audit = json.load(f)

    def tbl(name):
        p = os.path.join(args.tables_dir, name)
        return open(p, encoding="utf-8").read() if os.path.exists(p) else f"_missing: {name}_"

    parts = ["# ZG-NsDiff 实验结果", ""]

    parts += ["## 1. 数据集", ""]
    if audit:
        parts += [
            f"- 时间范围: {audit.get('time_start')} 至 {audit.get('time_end')}，共 {audit.get('n_rows')} 行（1 小时分辨率）。",
            f"- 缺失: {audit.get('missing_per_target')}",
            f"- 零值计数: {audit.get('zero_counts')}",
            f"- 窗口统计: {audit.get('windows')}",
            "",
        ]

    parts += [
        "## 2. 真实未来天气条件设置", "",
        "The observed future weather variables are used as oracle exogenous "
        "conditions for all compared models.", "",
        "## 3. 数据划分与预处理", "",
        "- 时间顺序划分 70/10/20；scaler、正值尺度与类别权重仅用训练集拟合。",
        "- 缺失窗口整体删除；缺失值不作为物理零值。", "",
        "## 4. 模型与训练配置", "",
        "见 configs/*.yaml 与 artifacts/runs/*/seed_*/best_checkpoint.pt 内嵌配置。", "",
        "## 5. 评价指标", "",
        "MAPE / MAPE+ / PICP(95%, 0.025–0.975) / AW / PINAW / CRPS / NCRPS / "
        "Energy Score / Variogram Score(0.5) / Corr. Error / Brier / ECE / Active F1。", "",
    ]

    parts += ["## 6. 模型对比结果", "", tbl("main_table.md"), ""]
    parts += ["## 7. 联合概率预测结果", "", tbl("joint_table.md"), ""]
    parts += ["## 8. 零状态预测结果", ""]
    for f in sorted(glob.glob(os.path.join(args.tables_dir, "zero_table_*.md"))):
        parts += [open(f, encoding="utf-8").read(), ""]

    parts += ["## 9. 消融实验", ""]
    ab = tbl("ablation_table.md")
    parts += [ab if "missing" not in ab else "_消融运行后自动生成（scripts/run_ablations.sh）_", ""]

    parts += ["## 10. 可视化", ""]
    figs = sorted(glob.glob(os.path.join(args.figures, "*.png")))
    parts += [f"![{os.path.basename(f)}]({f})" for f in figs] or ["_figures 目录为空_"]
    parts += [""]

    parts += ["## 11. 效率", ""]
    if os.path.exists(args.efficiency):
        eff = pd.read_csv(args.efficiency)
        try:
            parts += [eff.to_markdown(index=False), ""]
        except ImportError:  # tabulate not installed
            parts += ["```", eff.to_string(index=False), "```", ""]
    else:
        parts += ["_efficiency.csv 尚未生成_", ""]

    # 12. verifiable auto statements
    parts += ["## 12. 结果讨论", ""]
    if "nsdiff" in df.index and "zg_nsdiff" in df.index:
        for var in ["cooling", "heating", "pv"]:
            a = df.loc["nsdiff", f"{var}_mape_pos_mean"]
            b = df.loc["zg_nsdiff", f"{var}_mape_pos_mean"]
            ce_a = df.loc["nsdiff", f"{var}_coverage_error_mean"]
            ce_b = df.loc["zg_nsdiff", f"{var}_coverage_error_mean"]
            parts.append(
                f"- {var}: ZG-NsDiff 将 positive-only MAPE 由 {a:.4f} 变为 {b:.4f}"
                f"（相对变化 {rel_change(a, b):.2f}%）；95% PICP coverage error 由 "
                f"{ce_a:.4f} 变为 {ce_b:.4f}。"
            )
        a = df.loc["nsdiff", "ncrps_mean_mean"]
        b = df.loc["zg_nsdiff", "ncrps_mean_mean"]
        parts.append(f"- NCRPS(mean): NsDiff {a:.4f} → ZG-NsDiff {b:.4f}（相对变化 {rel_change(a, b):.2f}%）。")
    parts += [""]

    parts += ["## 13. 局限性", "", LIMITATION_TEXT, ""]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    text = "\n".join(parts)
    assert "[AUTO]" not in text
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

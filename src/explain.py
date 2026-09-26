"""
第 7 步（上）：用全部有标签的数据训练 LightGBM → 给 FY2025 公司打分 → SHAP 找出每家公司的前 3 个危险信号

用法：
    python src/explain.py                        # 默认 raw+industry 特征
输出：
    data/processed/scored_fy2025.parquet          # Streamlit 页面读取这个
    data/processed/peer_medians.parquet           # 行业中位数（同行对比表）
    results/shap_summary.png                      # README 用的图
"""
import argparse, pathlib, sys

import matplotlib
matplotlib.use("Agg")                               # 不弹窗，直接存图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_sklearn import FEATURE_SETS, RAW, make_lgbm

LIVE_YEAR = 2025


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/model_table.parquet")
    ap.add_argument("--features", default="raw+industry", choices=list(FEATURE_SETS))
    args = ap.parse_args()
    cols = FEATURE_SETS[args.features]

    df = pd.read_parquet(args.data)
    labelled = df[df["label"].notna()]                  # FY2014–2024：用全部历史数据训练
    live = df[df["fiscal_year"] == LIVE_YEAR].copy()    # FY2025：还不知道结果，拿来"实时"打分
    print(f"训练 {len(labelled):,} 行，打分 {len(live):,} 家 FY{LIVE_YEAR} 公司")

    model = make_lgbm().fit(labelled[cols], labelled["label"])
    live["risk_score"] = model.predict_proba(live[cols])[:, 1]

    # 百分位：比百分之多少的公司风险高（全部 A 股 / 仅科创板）
    live["pct_all"] = live["risk_score"].rank(pct=True)
    live["pct_star"] = np.nan
    star = live["is_star"]
    live.loc[star, "pct_star"] = live.loc[star, "risk_score"].rank(pct=True)

    # ---- SHAP：每个特征把这家公司的风险推高/拉低了多少 ----
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(live[cols])
    if isinstance(sv, list):                            # 老版本 shap 返回 [类别0, 类别1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                                    # 有的版本返回 (样本, 特征, 类别)
        sv = sv[:, :, 1]

    top3 = np.argsort(-sv, axis=1)[:, :3]               # 每行把风险推高最多的 3 个特征
    for j in range(3):
        live[f"flag_{j+1}"] = [cols[i] for i in top3[:, j]]
        live[f"flag_{j+1}_shap"] = sv[np.arange(len(sv)), top3[:, j]]

    out = pathlib.Path("data/processed")
    live.to_parquet(out / "scored_fy2025.parquet", index=False)

    # 同行业中位数（给页面的对比表用）
    peers = live.groupby("industry")[RAW].median().reset_index()
    peers.to_parquet(out / "peer_medians.parquet", index=False)

    # README 用的 SHAP 汇总图
    pathlib.Path("results").mkdir(exist_ok=True)
    shap.summary_plot(sv, live[cols], show=False, max_display=15)
    plt.tight_layout()
    plt.savefig("results/shap_summary.png", dpi=150)
    print("已保存 scored_fy2025.parquet、peer_medians.parquet、results/shap_summary.png")

    print("\n风险最高的 10 家：")
    print(live.nlargest(10, "risk_score")[["code", "name", "industry", "risk_score",
                                          "flag_1", "flag_2", "flag_3"]].to_string(index=False))


if __name__ == "__main__":
    main()

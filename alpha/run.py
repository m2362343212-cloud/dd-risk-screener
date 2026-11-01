"""
Step A4: scores -> forward returns -> tests -> alpha/output/report.md + summary.png

    python alpha/run.py                                  # real: walk-forward scores + data/prices
    python alpha/run.py --scores alpha/output/scores_walkforward_default.csv --out alpha/output_default
    python alpha/run.py --synthetic --effect 0.004        # pipeline check, planted effect
    python alpha/run.py --synthetic --effect 0            # pipeline check, size effect only
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evaluate_cn as E  # noqa: E402


def fmt(df, d=4):
    return df.to_markdown(floatfmt=f".{d}f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="alpha/output/scores_walkforward_tuned.csv")
    ap.add_argument("--prices", default="data/prices")
    ap.add_argument("--out", default="alpha/output")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--effect", type=float, default=0.004)
    ap.add_argument("--placebo", type=int, default=200)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores, dtype={"code": str})
    if args.synthetic:
        from synthetic_cn import make_prices
        px, be = make_prices(scores, effect=args.effect)
    else:
        px = E.load_prices(args.prices)
        be = E.load_book_equity()
    panel = E.build_panel(scores, px, be)
    panel = E.add_score_rank(E.add_groups(panel))
    n_m = panel["month"].nunique()
    print(f"panel: {len(panel):,} stock-months, {n_m} months, {panel['code'].nunique():,} stocks")

    ic = E.ic_table(panel)
    ew = E.group_returns(panel)
    vw = E.group_returns(panel, weight="float_mcap")
    cols = ["Q1", "Q2", "Q3", "Q4", "Q5", "TOP10", "ALL", "EX_TOP10"]
    q_tab = pd.DataFrame({c: {**dict(zip(["mean_monthly", "t_NW"], E.nw_mean_t(ew[c])[:2])),
                              **E.perf(ew[c])} for c in cols}).T
    screen = []
    for name, df in [("equal-weighted", ew), ("float-cap-weighted", vw)]:
        for c in ["SCREEN_GAIN", "LS_Q1_Q5"]:
            m, t, n = E.nw_mean_t(df[c])
            screen.append({"weighting": name, "series": c, "mean_monthly": m, "t_NW": t,
                           "ann": m * 12, "hit_rate": (df[c] > 0).mean(), "months": n})
    screen = pd.DataFrame(screen).set_index(["weighting", "series"])
    fm0 = E.fama_macbeth(panel, [])
    fm1 = E.fama_macbeth(panel, E.CONTROLS if panel["log_bm"].notna().any()
                         else [c for c in E.CONTROLS if c != "log_bm"])
    size_tab, big70 = E.size_split(panel)
    fm_big = E.fama_macbeth(big70, [c for c in E.CONTROLS if c in fm1.index])
    yr = E.by_year(ew)
    plc = E.placebo(panel, n_iter=args.placebo)

    for name, df in [("ic_by_horizon", ic), ("portfolio_returns_ew", ew),
                     ("portfolio_returns_vw", vw), ("fama_macbeth", fm1)]:
        df.to_csv(out / f"{name}.csv")

    # ---- plots
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    (1 + ew[["Q1", "Q2", "Q3", "Q4", "Q5"]].fillna(0)).cumprod().plot(
        ax=ax[0, 0], logy=True, title="Growth of 1 by risk quintile (Q1 safest, EW)")
    ew[cols].mean().mul(100).plot.bar(ax=ax[0, 1], color="#4C72B0",
                                      title="Mean next-month return (%), EW")
    E.ic_series(panel).rolling(12).mean().plot(ax=ax[1, 0], title="12-month rolling IC (h=1)")
    ax[1, 0].axhline(0, color="k", lw=0.8)
    pd.DataFrame({"EW: drop riskiest 10% vs all": (1 + ew["SCREEN_GAIN"].fillna(0)).cumprod(),
                  "VW: drop riskiest 10% vs all": (1 + vw["SCREEN_GAIN"].fillna(0)).cumprod()}
                 ).plot(ax=ax[1, 1], title="Long-only screen: cumulative gain")
    ax[1, 1].axhline(1, color="k", lw=0.8)
    for a in ax.flat:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "summary.png", dpi=130)

    src = (f"SYNTHETIC (planted effect {args.effect}/month)" if args.synthetic
           else "REAL (walk-forward screener scores + AKShare/Eastmoney monthly prices)")
    lines = [
        "# Screener risk score -> A-share returns", "",
        f"- Data: {src}",
        f"- Scores: `{args.scores}` (FY Y score usable from 1 May of Y+1)",
        f"- Sample: {panel['month'].min():%Y-%m} to {panel['month'].max():%Y-%m} formation months "
        f"({n_m}), {panel['code'].nunique():,} stocks, avg {len(panel) / n_m:,.0f} stocks/month",
        f"- Filters: >= {E.MIN_AGE_MONTHS} months since listing; traded in the formation month; "
        "next month's price available", "",
        "**Hypothesis:** higher risk score -> lower future return "
        "(mean IC < 0, Q1 - Q5 > 0, dropping the riskiest 10% helps).", "",
        "## 1. Information coefficient (Spearman)", fmt(ic), "",
        "## 2. Portfolios, next-month return (equal-weighted)", fmt(q_tab), "",
        "## 3. Long-only screen and long-short",
        "SCREEN_GAIN = (all stocks minus the riskiest 10%) - (all stocks). "
        "LS_Q1_Q5 is theoretical (shorting A-shares is very limited).", fmt(screen), "",
        "## 4. Fama-MacBeth (score_rank: -0.5 safest .. +0.5 riskiest, "
        "so its coefficient = monthly return gap riskiest vs safest)",
        "### Score only", fmt(fm0), "",
        "### Score + size, book-to-market, momentum, 1-month reversal (controls ranked -0.5..0.5 within month)", fmt(fm1), "",
        "## 5. Size", fmt(size_tab), "",
        "### Fama-MacBeth without the smallest 30% of stocks", fmt(fm_big), "",
        "## 6. Year by year (compounded, EW; years run June-May)", fmt(yr), "",
        "## 7. Placebo (score shuffled across stocks within each month)",
        f"Real IC t-stat {plc['real_IC_t']:.2f} (mean IC {plc['real_mean_IC']:.4f}) vs shuffled t-stats "
        f"{plc['placebo_mean']:.2f} +/- {plc['placebo_std']:.2f}; share at least as extreme: {plc['two_sided_p']:.3f} "
        f"({plc['n_iter']} shuffles).", "",
        "![summary](summary.png)", "",
        "## Caveats",
        "- Survivorship: the financial data contain almost only firms still listed today, so "
        "firms that were delisted (the most distressed) are missing.",
        "- Financial statement values come from Eastmoney's current database; their "
        "announcement dates are mostly later than the original annual report, so some "
        "values may be restated versions, not exactly what investors saw in April.",
        "- Float market cap is estimated as monthly amount / turnover; book-to-market uses "
        "total equity / float cap.",
        "- No ST flags or price-limit handling; LightGBM hyper-parameters were tuned on "
        "FY2021-22 (re-run with default params to check).",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"mean IC(1m) {ic.loc[1, 'mean_IC']:.4f} t {ic.loc[1, 'IC_t_NW']:.2f}")
    print(screen.round(4).to_string())
    print(fm1.round(4).to_string())
    print(f"placebo p {plc['two_sided_p']:.3f}\nreport -> {out / 'report.md'}")


if __name__ == "__main__":
    main()

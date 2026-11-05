"""
Robustness checks for the A-share test: does the conclusion survive a few
reasonable changes in how the test is run?

    python alpha/robustness.py                      # real: walk-forward scores + data/prices
    python alpha/robustness.py --synthetic --effect 0.004   # pipeline check, planted effect
    python alpha/robustness.py --synthetic --effect 0       # pipeline check, size effect only

Variants (each one re-runs IC, Fama-MacBeth and the long-only screen):
  baseline        the test in run.py, unchanged
  winsorized      next-month returns clipped at the 1st/99th percentile each month
                  (A-share returns have extreme outliers)
  liquid 80%      drop the 20% least-traded stocks each month (trading value `amount`);
                  a rough stand-in for "could you actually have traded it"
  signal +1m      use the score one extra month late (return from t+1 to t+2)
  first/second half   the sample split in the middle, to see if one period drives it

Then transaction costs: the long-only screen (hold everything except the riskiest 10%)
changes its holdings when a stock enters or leaves the flagged set. Net gain is
computed for several ROUND-TRIP cost assumptions. The costs are assumptions to
show sensitivity, not measured trading costs.

NOT done, and why: ST flags and price-limit (limit-up / limit-down) handling. The
price data are month-end bars, so a limit hit on the formation day cannot be seen, and
historical ST status is not in the data (only today's list is available, which would
leak the future). These stay listed as limitations.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evaluate_cn as E  # noqa: E402

ROUND_TRIP_BPS = [0, 20, 40, 80]          # assumptions, in basis points of traded value


# ---------------------------------------------------------------- variants
def winsorize_fwd(panel: pd.DataFrame, lo=0.01, hi=0.99) -> pd.DataFrame:
    p = panel.copy()
    q = p.groupby("month")["fwd_1m"]
    p["fwd_1m"] = p["fwd_1m"].clip(q.transform(lambda s: s.quantile(lo)),
                                   q.transform(lambda s: s.quantile(hi)))
    return p


def drop_illiquid(panel: pd.DataFrame, frac=0.20) -> pd.DataFrame:
    """Keep stocks above the bottom `frac` of monthly trading value, ranked within each month."""
    rk = panel.groupby("month")["amount"].rank(pct=True)
    return panel[rk > frac].copy()


def delay_signal(panel: pd.DataFrame, months=1) -> pd.DataFrame:
    """Score known at t, but the position only starts `months` later: use the return
    that stock earns from t+months to t+months+1."""
    lk = panel[["code", "month", "fwd_1m"]].copy()
    lk["month"] = lk["month"] - pd.offsets.MonthEnd(months)        # key = formation month
    lk = lk.rename(columns={"fwd_1m": "fwd_1m_delayed"})
    p = panel.merge(lk, on=["code", "month"], how="left")
    p["fwd_1m"] = p["fwd_1m_delayed"]
    return p.dropna(subset=["fwd_1m"]).drop(columns="fwd_1m_delayed").reset_index(drop=True)


def halves(panel: pd.DataFrame):
    months = np.sort(panel["month"].unique())
    cut = months[len(months) // 2]
    return panel[panel["month"] < cut].copy(), panel[panel["month"] >= cut].copy()


# ---------------------------------------------------------------- one row of results
def summarize(panel: pd.DataFrame, label: str) -> dict:
    p = E.add_score_rank(E.add_groups(panel))
    controls = E.CONTROLS if p["log_bm"].notna().any() else [c for c in E.CONTROLS if c != "log_bm"]
    ic = E.ic_series(p, 1)
    ic_m, ic_t, _ = E.nw_mean_t(ic)
    fm = E.fama_macbeth(p, controls).loc["score_rank"]
    ew = E.group_returns(p)
    sg_m, sg_t, _ = E.nw_mean_t(ew["SCREEN_GAIN"])
    return {"variant": label, "months": p["month"].nunique(),
            "stocks_per_month": round(len(p) / p["month"].nunique()),
            "mean_IC": ic_m, "IC_t": ic_t,
            "FM_score_coef_%_per_month": fm["avg_coef_monthly"] * 100, "FM_t": fm["t_NW"],
            "screen_gain_%_per_month": sg_m * 100, "screen_t": sg_t}


# ---------------------------------------------------------------- transaction costs
def one_way_turnover(prev: set, cur: set) -> float:
    """One-way turnover of an equal-weight portfolio that moves from `prev` to `cur`
    (ignores price drift between months): half the sum of absolute weight changes."""
    if not prev:
        return np.nan                                  # first month: building the portfolio, not rebalancing
    w0 = 1 / len(prev)
    w1 = 1 / len(cur)
    names = prev | cur
    return 0.5 * sum(abs((w1 if n in cur else 0.0) - (w0 if n in prev else 0.0)) for n in names)


def screen_turnover(panel: pd.DataFrame) -> pd.DataFrame:
    """Per month: one-way turnover of 'all stocks' and of 'all except the riskiest 10%'.
    Indexed by the month the return is earned (same labelling as E.group_returns)."""
    p = E.add_groups(panel)
    prev_all, prev_ex = set(), set()
    rows = {}
    for m, g in p.groupby("month"):
        all_now = set(g["code"])
        ex_now = set(g.loc[~g["flagged"], "code"])
        rows[m] = {"TO_ALL": one_way_turnover(prev_all, all_now),
                   "TO_EX": one_way_turnover(prev_ex, ex_now)}
        prev_all, prev_ex = all_now, ex_now
    t = pd.DataFrame(rows).T
    t.index = t.index + pd.offsets.MonthEnd(1)
    return t


def cost_table(panel: pd.DataFrame, round_trip_bps=ROUND_TRIP_BPS) -> pd.DataFrame:
    p = E.add_score_rank(E.add_groups(panel))
    ew = E.group_returns(p)
    to = screen_turnover(panel)
    d = ew[["ALL", "EX_TOP10"]].join(to, how="inner").dropna()
    rows = []
    for bps in round_trip_bps:
        c = bps / 2 / 1e4                                           # one way
        net = (d["EX_TOP10"] - c * d["TO_EX"]) - (d["ALL"] - c * d["TO_ALL"])
        m, t, n = E.nw_mean_t(net)
        rows.append({"round_trip_cost_bps": bps, "net_screen_gain_%_per_month": m * 100, "t_NW": t,
                     "avg_one_way_turnover_screened": d["TO_EX"].mean(),
                     "avg_one_way_turnover_all": d["TO_ALL"].mean(), "months": n})
    return pd.DataFrame(rows).set_index("round_trip_cost_bps")


# ---------------------------------------------------------------- driver
def run_all(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first, second = halves(panel)
    rows = [summarize(panel, "baseline"),
            summarize(winsorize_fwd(panel), "winsorized returns"),
            summarize(drop_illiquid(panel), "liquid 80%"),
            summarize(delay_signal(panel), "signal +1 month"),
            summarize(first, "first half"),
            summarize(second, "second half")]
    return pd.DataFrame(rows).set_index("variant"), cost_table(panel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="alpha/output/scores_walkforward_tuned.csv")
    ap.add_argument("--prices", default="data/prices")
    ap.add_argument("--out", default="alpha/output/robustness")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--effect", type=float, default=0.004)
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

    variants, costs = run_all(panel)
    variants.to_csv(out / "variants.csv")
    costs.to_csv(out / "costs.csv")

    src = (f"SYNTHETIC (planted effect {args.effect}/month)" if args.synthetic
           else "REAL (walk-forward scores + Sina monthly prices)")
    lines = [
        "# Robustness of the screener-score -> A-share return test", "",
        f"- Data: {src}",
        "- Each row repeats the main test with one change. FM = Fama-MacBeth coefficient of the score "
        "(riskiest minus safest, % per month, controls: size, book-to-market, momentum, reversal). "
        "Screen = hold all stocks except the riskiest 10%, minus holding all stocks.",
        "- If the conclusion is 'no effect', the thing to look for is whether any row changes the sign "
        "or pushes |t| above 2.", "",
        "## 1. Variants", variants.to_markdown(floatfmt=".3f"), "",
        "## 2. Transaction costs on the long-only screen",
        "Round-trip costs are assumptions for sensitivity, not measured. Turnover counts only stocks "
        "entering or leaving a portfolio and ignores price drift between months; both legs rebalance "
        "to equal weights, so most of the drift cost cancels in the difference, but it would not for a "
        "stand-alone portfolio.", "", costs.to_markdown(floatfmt=".3f"), "",
        "## Still not covered",
        "- ST flags and price limits (month-end bars; no historical ST status in the data).",
        "- Survivorship bias (delisted firms missing from the financial data).",
        "- One market, one sample period.",
    ]
    (out / "robustness.md").write_text("\n".join(lines), encoding="utf-8")
    print(variants.round(3).to_string())
    print()
    print(costs.round(3).to_string())
    print(f"\nreport -> {out / 'robustness.md'}")


if __name__ == "__main__":
    main()

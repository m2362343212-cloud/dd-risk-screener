"""
Step A3: do the screener's risk scores predict A-share returns?

Hypothesis: higher risk score today -> lower future return.

Tests
  1. IC: monthly Spearman(score, next-h-month return), h = 1/3/6/12, Newey-West t
  2. Quintiles (equal-weighted) + the top-10% "flagged" group
  3. LONG-ONLY SCREEN (the practical test, A-shares can't really be shorted):
       all stocks equal-weighted  vs  same portfolio minus the 10% riskiest
  4. Long-short Q1 (safest) - Q5 (riskiest): theoretical only
  5. Fama-MacBeth: each month regress next-month return on the score rank AND
     size, book-to-market, momentum, 1-month reversal. Is the score still
     significant once we hold those fixed?  (No Ken French factors for China.)
  6. Size check: IC inside small / mid / large terciles, and without the
     smallest 30% (shell-value stocks; Liu-Stambaugh-Yuan 2019 drop them too)
  7. Year by year + placebo (score shuffled within each month)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HORIZONS = [1, 3, 6, 12]
NW_LAGS = 6
MIN_AGE_MONTHS = 6          # drop the first 6 months after IPO (A-share IPO pops)
MAX_STALE_DAYS = 400        # one annual score is used for at most ~13 months
TOP_FRAC = 0.10


# ---------------------------------------------------------------- data
def load_prices(price_dir: str) -> pd.DataFrame:
    import pathlib
    files = sorted(pathlib.Path(price_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no price files in {price_dir}; run alpha/fetch_prices.py first")
    px = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["month"] = pd.to_datetime(px["month"]) + pd.offsets.MonthEnd(0)
    return px.drop_duplicates(["code", "month"]).sort_values(["code", "month"])


def load_book_equity(raw_dir: str = "data/raw") -> pd.DataFrame:
    import pathlib
    out = []
    for f in sorted(pathlib.Path(raw_dir).glob("zcfz_*.parquet")):
        z = pd.read_parquet(f, columns=["股票代码", "股东权益合计", "fiscal_year"])
        out.append(z)
    be = pd.concat(out).rename(columns={"股票代码": "code", "股东权益合计": "book_equity"})
    be["code"] = be["code"].astype(str).str.zfill(6)
    be["book_equity"] = pd.to_numeric(be["book_equity"], errors="coerce")
    return be.drop_duplicates(["code", "fiscal_year"])


def build_panel(scores: pd.DataFrame, px: pd.DataFrame, be: pd.DataFrame | None) -> pd.DataFrame:
    """One row = (month-end t, stock): score known at t, characteristics at t,
    forward returns t -> t+h."""
    px = px.copy()
    g = px.groupby("code")
    px["age"] = g.cumcount()                                      # months since first bar
    mi = (px["month"].dt.year * 12 + px["month"].dt.month)
    px["_mi"] = mi
    wide = px.pivot(index="month", columns="code", values="close_hfq").asfreq("ME")
    for h in HORIZONS:
        f = (wide.shift(-h) / wide - 1).stack().rename(f"fwd_{h}m")
        px = px.merge(f.reset_index().rename(columns={"level_1": "code"}), on=["month", "code"], how="left")
    r1 = (wide / wide.shift(1) - 1)
    mom = (wide.shift(1) / wide.shift(12) - 1)                    # t-12 -> t-1, skips last month
    for name, w in [("rev_1m", r1), ("mom_12_2", mom)]:
        px = px.merge(w.stack().rename(name).reset_index().rename(columns={"level_1": "code"}),
                      on=["month", "code"], how="left")

    s = scores.copy()
    s["code"] = s["code"].astype(str).str.zfill(6)
    s["available"] = pd.to_datetime(s["date"]) + pd.Timedelta(days=1)  # usable the day after
    s = s.sort_values("available")
    px = px.sort_values("month")
    panel = pd.merge_asof(px, s[["code", "available", "distress", "fiscal_year"]],
                          left_on="month", right_on="available", by="code", direction="backward")
    stale = (panel["month"] - panel["available"]).dt.days > MAX_STALE_DAYS
    panel.loc[stale, "distress"] = np.nan
    assert (panel.dropna(subset=["distress"])["available"] <= panel.dropna(subset=["distress"])["month"]).all()

    if be is not None:
        panel = panel.merge(be, on=["code", "fiscal_year"], how="left")
        panel["log_bm"] = np.log(panel["book_equity"] / panel["float_mcap"]).where(panel["book_equity"] > 0)
    else:
        panel["log_bm"] = np.nan
    panel["log_mcap"] = np.log(panel["float_mcap"].astype(float))
    panel = panel[(panel["age"] >= MIN_AGE_MONTHS) & panel["distress"].notna()
                  & panel["fwd_1m"].notna() & panel["log_mcap"].notna()]
    return panel.reset_index(drop=True)


# ---------------------------------------------------------------- helpers
def nw_mean_t(x: pd.Series, lags: int = NW_LAGS):
    x = pd.Series(x).dropna()
    if len(x) < 12:
        return np.nan, np.nan, len(x)
    r = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[0]), float(r.tvalues[0]), len(x)


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    vol = r.std() * np.sqrt(12)
    return {"ann_return": r.mean() * 12, "ann_vol": vol,
            "sharpe": r.mean() * 12 / vol if vol > 0 else np.nan,
            "max_drawdown": (cum / cum.cummax() - 1).min(), "months": len(r)}


def _next_month(idx):
    return idx + pd.offsets.MonthEnd(1)


# ---------------------------------------------------------------- 1. IC
def ic_series(p: pd.DataFrame, h=1, col="distress") -> pd.Series:
    def f(g):
        g = g[[col, f"fwd_{h}m"]].dropna()
        return stats.spearmanr(g[col], g[f"fwd_{h}m"]).statistic if len(g) >= 50 else np.nan
    return p.groupby("month")[[col, f"fwd_{h}m"]].apply(f).dropna()


def ic_table(p):
    rows = []
    for h in HORIZONS:
        ic = ic_series(p, h)
        m, t, n = nw_mean_t(ic, max(NW_LAGS, h))
        rows.append({"horizon_m": h, "mean_IC": m, "IC_t_NW": t, "IC_std": ic.std(),
                     "pct_months_IC<0": (ic < 0).mean(), "months": n})
    return pd.DataFrame(rows).set_index("horizon_m")


# ---------------------------------------------------------------- 2-4. portfolios
def add_groups(p):
    p = p.copy()
    rk = p.groupby("month")["distress"].rank(pct=True, method="first")
    p["q"] = np.ceil(rk * 5).clip(1, 5).astype(int)              # 1 safest ... 5 riskiest
    p["flagged"] = rk > 1 - TOP_FRAC
    return p


def group_returns(p, weight=None):
    def wmean(g):
        if weight is None:
            return g["fwd_1m"].mean()
        w = g[weight]
        return np.average(g["fwd_1m"], weights=w)
    q = p.groupby(["month", "q"])[["fwd_1m"] + ([weight] if weight else [])].apply(wmean).unstack()
    q.columns = [f"Q{c}" for c in q.columns]
    allr = p.groupby("month")[["fwd_1m"] + ([weight] if weight else [])].apply(wmean).rename("ALL")
    ex = p[~p["flagged"]].groupby("month")[["fwd_1m"] + ([weight] if weight else [])].apply(wmean).rename("EX_TOP10")
    top = p[p["flagged"]].groupby("month")[["fwd_1m"] + ([weight] if weight else [])].apply(wmean).rename("TOP10")
    out = pd.concat([q, allr, ex, top], axis=1)
    out.index = _next_month(out.index)                            # label by the month the return is earned
    out["SCREEN_GAIN"] = out["EX_TOP10"] - out["ALL"]
    out["LS_Q1_Q5"] = out["Q1"] - out["Q5"]
    return out


# ---------------------------------------------------------------- 5. Fama-MacBeth
CONTROLS = ["log_mcap", "log_bm", "mom_12_2", "rev_1m"]


def fama_macbeth(p, controls):
    rows = {}
    for m, g in p.groupby("month"):
        cols = ["score_rank"] + controls
        d = g[["fwd_1m"] + cols].dropna()
        if len(d) < 100:
            continue
        X = d[cols].copy()
        for c in controls:                                      # rank controls within month (-0.5..0.5):
            X[c] = X[c].rank(pct=True) - 0.5                    # robust to outliers and non-linear size effects
        res = sm.OLS(d["fwd_1m"], sm.add_constant(X)).fit()
        rows[m] = res.params
    coefs = pd.DataFrame(rows).T
    out = []
    for c in coefs.columns:
        mean, t, n = nw_mean_t(coefs[c])
        out.append({"variable": c, "avg_coef_monthly": mean, "t_NW": t, "months": n})
    return pd.DataFrame(out).set_index("variable")


def add_score_rank(p):
    p = p.copy()
    p["score_rank"] = p.groupby("month")["distress"].rank(pct=True) - 0.5   # -0.5 safest .. +0.5 riskiest
    return p


# ---------------------------------------------------------------- 6-7. robustness
def size_split(p):
    p = p.copy()
    p["size_t"] = p.groupby("month")["log_mcap"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["small", "mid", "large"]))
    rows = []
    for name, g in p.groupby("size_t", observed=True):
        m, t, n = nw_mean_t(ic_series(g, 1))
        rows.append({"group": f"{name} third", "mean_IC": m, "IC_t": t, "months": n})
    big70 = p[p.groupby("month")["log_mcap"].rank(pct=True) > 0.30]
    m, t, n = nw_mean_t(ic_series(big70, 1))
    rows.append({"group": "drop smallest 30%", "mean_IC": m, "IC_t": t, "months": n})
    return pd.DataFrame(rows).set_index("group"), big70


def by_year(gr):
    """Formation year = June of Y+1 .. May of Y+2 holds FY Y scores; label by return month."""
    y = gr.index.to_series().apply(lambda d: d.year if d.month >= 6 else d.year - 1)
    tab = gr.groupby(y.values)[["ALL", "TOP10", "SCREEN_GAIN", "LS_Q1_Q5"]].agg(
        lambda r: (1 + r).prod() - 1)
    tab.index.name = "June-May starting"
    return tab


def placebo(p, n_iter=200, seed=0):
    """Permutation test (studentized, yearly shuffle)."""
    rng = np.random.default_rng(seed)
    q = p[["month", "code", "fiscal_year", "distress", "fwd_1m"]].copy()
    real_t = nw_mean_t(ic_series(q))[1]
    base = q[["fiscal_year", "code", "distress"]].drop_duplicates(["fiscal_year", "code"])
    sims = []
    for _ in range(n_iter):
        b = base.copy()
        b["shuf"] = b.groupby("fiscal_year")["distress"].transform(lambda s: rng.permutation(s.values))
        qq = q.merge(b[["fiscal_year", "code", "shuf"]], on=["fiscal_year", "code"], how="left")
        sims.append(nw_mean_t(ic_series(qq, col="shuf"))[1])
    sims = np.array(sims)
    return {"real_mean_IC": ic_series(q).mean(), "real_IC_t": real_t,
            "placebo_mean": sims.mean(), "placebo_std": sims.std(),
            "two_sided_p": (np.abs(sims) >= abs(real_t)).mean(), "n_iter": n_iter}

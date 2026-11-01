"""
Fake A-share prices with a known answer, built around the REAL score file.

Built-in trap: small stocks earn a premium (like A-share shell value), and
risky companies tend to be small. So a raw sort can show "risky stocks do
better" even with effect = 0. Fama-MacBeth with a size control should remove it.

  effect > 0 : riskier score -> lower return by `effect` per month per unit of
               rank gap (riskiest vs safest ~ effect)
  effect = 0 : score has no direct effect at all
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_prices(scores: pd.DataFrame, effect=0.004, size_premium=0.006, seed=3):
    rng = np.random.default_rng(seed)
    s = scores.copy()
    s["code"] = s["code"].astype(str).str.zfill(6)
    codes = sorted(s["code"].unique())
    months = pd.date_range("2016-01-31", "2026-09-30", freq="ME")
    T, N = len(months), len(codes)
    col = {c: i for i, c in enumerate(codes)}

    # score known in each month (same timing rule as the real pipeline)
    S = np.full((T, N), np.nan)
    s["avail"] = pd.to_datetime(s["date"]) + pd.Timedelta(days=1)
    for _, r in s.sort_values("avail").iterrows():
        rows = (months >= r["avail"]) & (months <= r["avail"] + pd.Timedelta(days=400))
        S[rows, col[r["code"]]] = r["distress"]
    rank = pd.DataFrame(S).rank(axis=1, pct=True).values - 0.5          # -0.5 safe .. +0.5 risky

    # log size: risky firms are smaller
    base = rng.normal(22.5, 1.0, N)
    log_mcap = base[None, :] - 0.8 * np.nan_to_num(rank) + rng.normal(0, 0.3, (T, N))
    size_rank = pd.DataFrame(log_mcap).rank(axis=1, pct=True).values - 0.5

    mkt = rng.normal(0.006, 0.07, T)
    R = np.zeros((T, N))
    for t in range(1, T):
        R[t] = (mkt[t] + rng.normal(1, 0.25, N) * mkt[t]
                - size_premium * size_rank[t - 1] * 2           # small -> higher return
                - effect * np.nan_to_num(rank[t - 1])               # risky -> lower return
                + rng.normal(0, 0.10, N))
    R = np.clip(R, -0.6, 1.5)
    close = 10 * np.cumprod(1 + R, axis=0)
    mcap = np.exp(log_mcap)
    turnover = rng.uniform(5, 60, (T, N))
    px = pd.DataFrame({
        "code": np.repeat([codes], T, axis=0).ravel(),
        "month": np.repeat(months, N),
        "close_hfq": close.ravel(),
        "turnover": turnover.ravel(),
        "amount": (mcap * turnover / 100).ravel(),
    })
    px["float_mcap"] = px["amount"] / (px["turnover"] / 100)
    be = s[["code", "fiscal_year"]].drop_duplicates().copy()
    be["book_equity"] = np.exp(rng.normal(21.5, 1.0, len(be)))
    return px, be

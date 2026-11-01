"""Step A2: monthly A-share prices + float market cap via AKShare (Eastmoney).
One file per stock in data/prices/<code>.parquet; re-running skips finished stocks.
float_mcap = amount / (turnover/100)  ~= float shares * average price."""
from __future__ import annotations

import argparse
import pathlib
import random
import time

import pandas as pd

OUT = pathlib.Path("data/prices")
START, END = "20170101", "20261031"
RENAME = {"日期": "date", "收盘": "close", "成交额": "amount", "换手率": "turnover"}


def bars(code: str, adjust: str) -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=code, period="monthly", start_date=START,
                            end_date=END, adjust=adjust)
    missing = [c for c in RENAME if c not in df.columns]
    if missing:
        raise SystemExit(f"AKShare columns changed, missing {missing}. Got: {list(df.columns)}")
    df = df[list(RENAME)].rename(columns=RENAME)
    df["date"] = pd.to_datetime(df["date"])
    return df


def amount_is_unadjusted(code="000001") -> bool:
    a = bars(code, "hfq").set_index("date")["amount"]
    b = bars(code, "").set_index("date")["amount"]
    diff = (a / b - 1).abs().max()
    print(f"check {code}: max relative difference of amount hfq vs raw = {diff:.2e}")
    return diff < 0.01


def fetch_one(code: str, raw_amount: bool) -> pd.DataFrame:
    h = bars(code, "hfq")
    out = pd.DataFrame({"month": h["date"] + pd.offsets.MonthEnd(0),
                        "close_hfq": h["close"], "amount": h["amount"],
                        "turnover": h["turnover"]})
    if raw_amount:
        r = bars(code, "")
        r["month"] = r["date"] + pd.offsets.MonthEnd(0)
        out = out.drop(columns="amount").merge(r[["month", "amount"]], on="month", how="left")
    t = pd.to_numeric(out["turnover"], errors="coerce")
    out["float_mcap"] = pd.to_numeric(out["amount"], errors="coerce") / (t / 100)
    out.loc[t <= 0, "float_mcap"] = pd.NA
    out.insert(0, "code", code)
    return out


def with_retry(fn, tries=4, waits=(5, 15, 45)):
    """Eastmoney sometimes drops connections when requests come too fast."""
    for k in range(tries):
        try:
            return fn()
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            w = waits[min(k, len(waits) - 1)]
            print(f"    connection problem ({type(e).__name__}), waiting {w}s and retrying...")
            time.sleep(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--codes", default="data/processed/model_table.parquet")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    if args.check:
        print(with_retry(lambda: bars("000001", "hfq")).tail(3))
        print("amount unaffected by adjustment:", with_retry(amount_is_unadjusted))
        return

    codes = sorted(pd.read_parquet(args.codes, columns=["code"])["code"]
                   .astype(str).str.zfill(6).unique())
    OUT.mkdir(parents=True, exist_ok=True)
    flag = OUT / "_amount_check.txt"
    if flag.exists():
        raw_amount = flag.read_text().strip() == "raw"
    else:
        raw_amount = not with_retry(amount_is_unadjusted)
        flag.write_text("raw" if raw_amount else "hfq_ok")
    if raw_amount:
        print("amount differs under hfq -> also downloading unadjusted bars (2x slower)")

    todo = [c for c in codes if not (OUT / f"{c}.parquet").exists()]
    print(f"{len(codes):,} stocks, {len(codes) - len(todo):,} already downloaded, {len(todo):,} to go")
    failed, streak = [], 0
    for i, c in enumerate(todo, 1):
        try:
            df = with_retry(lambda: fetch_one(c, raw_amount))
            df.to_parquet(OUT / f"{c}.parquet", index=False)
            streak = 0
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            failed.append(c)
            streak += 1
            print(f"  [fail] {c}: {type(e).__name__}")
            if streak >= 15:
                print("\n15 stocks in a row failed: Eastmoney is probably blocking us for now.\n"
                      "Wait ~30 minutes (or switch Wi-Fi / hotspot), then run the same command;\n"
                      "finished stocks are kept and skipped.")
                return
            if streak % 5 == 0:
                print("  5 failures in a row -> pausing 5 minutes")
                time.sleep(300)
        time.sleep(random.uniform(0.6, 1.4) * args.sleep)
        if i % 100 == 0:
            done = len(codes) - len(todo) + i - len(failed)
            print(f"  {i:,}/{len(todo):,}  (saved so far: {done:,})")
    print(f"done. failed: {len(failed)} {failed[:10]}")
    if failed:
        print("re-run the same command to retry only the failed ones")


if __name__ == "__main__":
    main()

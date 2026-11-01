"""Backup price source: SINA via AKShare. Same output as fetch_prices.py."""
from __future__ import annotations
import argparse, pathlib, random, signal, time
import pandas as pd

OUT = pathlib.Path("data/prices")
START, END = "20170101", "20261031"


def sina_symbol(code):
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def daily(code, adjust):
    import akshare as ak
    df = ak.stock_zh_a_daily(symbol=sina_symbol(code), start_date=START, end_date=END, adjust=adjust)
    if df is None or len(df) == 0:
        raise ValueError("empty")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_one(code):
    raw = daily(code, "")
    hfq = daily(code, "hfq")
    need = {"close", "amount", "outstanding_share"}
    if not need <= set(raw.columns):
        raise SystemExit(f"Sina columns changed. Need {need}, got {list(raw.columns)}")
    d = raw[["date", "close", "amount", "outstanding_share"]].merge(
        hfq[["date", "close"]].rename(columns={"close": "close_hfq"}), on="date", how="inner")
    for c in ["close", "amount", "outstanding_share", "close_hfq"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    m = d.set_index("date").resample("ME").agg(
        {"close": "last", "close_hfq": "last", "amount": "sum", "outstanding_share": "last"})
    m = m.dropna(subset=["close_hfq"])
    m["float_mcap"] = m["close"] * m["outstanding_share"]
    m["turnover"] = m["amount"] / m["float_mcap"] * 100
    m = m.reset_index().rename(columns={"date": "month"})
    m.insert(0, "code", code)
    return m[["code", "month", "close_hfq", "amount", "turnover", "float_mcap"]]


def _alarm(signum, frame):
    raise TimeoutError("no answer in 90s")


def with_timeout(fn, sec=90):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(sec)
    try:
        return fn()
    finally:
        signal.alarm(0)


def with_retry(fn, tries=3, waits=(10, 60)):
    for k in range(tries):
        try:
            return with_timeout(fn)
        except SystemExit:
            raise
        except Exception as e:
            if k == tries - 1:
                raise
            w = waits[min(k, len(waits) - 1)]
            print(f"    connection problem ({type(e).__name__}), waiting {w}s...")
            time.sleep(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--codes", default="data/processed/model_table.parquet")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--part", default="1/1", help="e.g. 1/2 and 2/2 in two terminals")
    args = ap.parse_args()
    if args.check:
        raw = with_retry(lambda: daily("000001", ""))
        print("Sina columns:", list(raw.columns))
        print(with_retry(lambda: fetch_one("000001")).tail(3).to_string())
        print("\nOK if float_mcap above is about 2e11.")
        return
    codes = sorted(pd.read_parquet(args.codes, columns=["code"])["code"].astype(str).str.zfill(6).unique())
    k, n = (int(x) for x in args.part.split("/"))
    codes = [c for j, c in enumerate(codes) if j % n == k - 1]
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [c for c in codes if not (OUT / f"{c}.parquet").exists()]
    print(f"{len(codes):,} stocks, {len(codes) - len(todo):,} done, {len(todo):,} to go")
    failed, streak = [], 0
    for i, c in enumerate(todo, 1):
        try:
            with_retry(lambda: fetch_one(c)).to_parquet(OUT / f"{c}.parquet", index=False)
            streak = 0
        except SystemExit:
            raise
        except Exception as e:
            failed.append(c)
            streak += 1
            print(f"  [fail] {c}: {type(e).__name__}")
            if streak >= 10:
                print("\n10 in a row failed: Sina is blocking for now. Wait a few hours, then re-run.")
                return
        time.sleep(random.uniform(0.6, 1.4) * args.sleep)
        if i % 50 == 0:
            print(f"  {i:,}/{len(todo):,}  (failed so far: {len(failed)})")
    print(f"done. failed: {len(failed)} {failed[:10]}")


if __name__ == "__main__":
    main()

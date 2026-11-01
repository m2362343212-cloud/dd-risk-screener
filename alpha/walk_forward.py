"""
Step A1: walk-forward (expanding-window) risk scores -> every score is out-of-sample.

Why: the original split (train FY2014-20 / val 21-22 / test 23-24) gives only 2
out-of-sample years. To test scores against stock returns we need a score for
every year that the model could have produced AT THAT TIME.

Timing rule (A-share annual reports must be out by 30 April):
    score fiscal year Y  -> portfolio formed after 30 April of Y+1
    train on rows FY <= Y-1, whose label = "did year FY+1 deteriorate?"
    the last of those labels (outcome in year Y) is revealed by the FY Y
    annual report, i.e. by 30 April of Y+1  -> known at formation time. OK.

Usage (from the repo root):
    python alpha/walk_forward.py                   # tuned LightGBM params (same as the screener)
    python alpha/walk_forward.py --params default  # stricter: LightGBM defaults, no tuning info
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from train_sklearn import FEATURE_SETS, evaluate, make_lgbm  # noqa: E402  (the screener's own model)

DATA = "data/processed/model_table.parquet"
FIRST_SCORE_YEAR = 2017      # needs >= 3 years of training data (FY2014-2016)


def make_default():
    return lgb.LGBMClassifier(is_unbalance=True, random_state=0, verbose=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--params", default="tuned", choices=["tuned", "default"])
    ap.add_argument("--features", default="raw+industry", choices=list(FEATURE_SETS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cols = FEATURE_SETS[args.features]
    make = make_lgbm if args.params == "tuned" else make_default
    df = pd.read_parquet(args.data)
    df["code"] = df["code"].astype(str).str.zfill(6)
    last_year = int(df["fiscal_year"].max())

    scores, rows = [], []
    for y in range(FIRST_SCORE_YEAR, last_year + 1):
        train = df[(df["fiscal_year"] <= y - 1) & df["label"].notna()]
        test = df[df["fiscal_year"] == y].copy()
        model = make().fit(train[cols], train["label"])
        test["distress"] = model.predict_proba(test[cols])[:, 1]
        test["date"] = pd.Timestamp(f"{y + 1}-04-30")
        scores.append(test[["code", "date", "distress", "fiscal_year"]])

        info = {"score_FY": y, "formed_after": f"{y + 1}-04-30",
                "train_FY": f"{int(train.fiscal_year.min())}-{y - 1}",
                "train_rows": len(train), "scored": len(test)}
        lab = test["label"].notna()
        if lab.sum() > 0 and test.loc[lab, "label"].nunique() == 2:
            m = evaluate(test.loc[lab, "label"], test.loc[lab, "distress"])
            info.update({"base_rate": test.loc[lab, "label"].mean(),
                         "pr_auc": m["pr_auc"], "prec_top10": m["prec_top10"]})
        rows.append(info)
        print(f"FY{y}: trained on {info['train_FY']} ({len(train):,} rows) -> scored {len(test):,}"
              + (f" | PR-AUC {info['pr_auc']:.3f} vs base {info['base_rate']:.3f}"
                 if "pr_auc" in info else " | (labels not known yet)"))

    out = pathlib.Path(args.out or f"alpha/output/scores_walkforward_{args.params}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(scores).to_csv(out, index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(out.with_name(f"walkforward_classification_{args.params}.csv"), index=False)
    print(f"\nsaved {out}  ({sum(len(s) for s in scores):,} company-years)")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

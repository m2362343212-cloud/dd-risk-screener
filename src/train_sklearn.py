"""
第 4–5 步：按时间切分 → 两个基准模型 → 逻辑回归 & LightGBM（两套特征）→ MLflow 记录

用法：
    python src/train_sklearn.py                                    # 在验证集上比较所有模型
    python src/train_sklearn.py --final --model lgbm --features raw+industry   # 选好后，测试集只跑一次
"""
import argparse, json, pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import mlflow
except ImportError:
    mlflow = None

DATA = "data/processed/model_table.parquet"
TRAIN_YEARS, VAL_YEARS, TEST_YEARS = range(2014, 2021), range(2021, 2023), range(2023, 2025)

RAW = ["gross_margin", "roe", "op_margin", "rev_growth", "profit_growth", "debt_ratio",
       "cash_ratio", "cfo_assets", "accruals", "recv_vs_rev", "inv_vs_rev",
       "exp_ratio_chg", "log_assets"]
IND = [f + "_ind" for f in RAW if f != "log_assets"]
FEATURE_SETS = {"raw": RAW, "raw+industry": RAW + IND}
RATIO5 = ["debt_ratio", "op_margin", "cash_ratio", "cfo_assets", "log_assets"]   # 类 Altman 的 5 个比率


# ---------------- 评价指标 ----------------
def precision_at_top(y_true, score, frac=0.10):
    """模型认为风险最高的前 10% 公司里，真正恶化的比例。"""
    k = max(1, int(len(score) * frac))
    top = np.argsort(-score)[:k]
    return float(np.asarray(y_true)[top].mean())


def evaluate(y, p):
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "prec_top10": precision_at_top(y, p)}


# ---------------- 模型 ----------------
def make_logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000))


def make_lgbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                              min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, is_unbalance=True,   # 自动处理类别不平衡
                              random_state=0, verbose=-1)               # 缺失值 LightGBM 自己处理


MODELS = {"logreg": make_logreg, "lgbm": make_lgbm}


def log_run(run_name, params, metrics):
    if mlflow is None:
        return
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--model", default="lgbm", choices=list(MODELS))
    ap.add_argument("--features", default="raw+industry", choices=list(FEATURE_SETS))
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = df[df["label"].notna()]                               # 没有标签的行（FY2025）不参与训练
    train = df[df.fiscal_year.isin(TRAIN_YEARS)]
    val   = df[df.fiscal_year.isin(VAL_YEARS)]
    test  = df[df.fiscal_year.isin(TEST_YEARS)]
    print(f"训练 {len(train):,} 行（恶化 {train.label.mean():.1%}） | "
          f"验证 {len(val):,} 行（{val.label.mean():.1%}） | 测试 {len(test):,} 行（{test.label.mean():.1%}）")

    out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    if mlflow:
        mlflow.set_experiment("dd-risk-screener")

    # ================= 最终测试：只跑一次 =================
    if args.final:
        cols = FEATURE_SETS[args.features]
        trval = pd.concat([train, val])                         # 训练 + 验证 合并重训
        model = MODELS[args.model]().fit(trval[cols], trval["label"])
        p = model.predict_proba(test[cols])[:, 1]
        m = evaluate(test["label"], p)
        rule = evaluate(test["label"], -test["profit_growth"].fillna(0))
        ratio = make_logreg().fit(trval[RATIO5], trval["label"]).predict_proba(test[RATIO5])[:, 1]
        m_ratio = evaluate(test["label"], ratio)
        star = test["is_star"].to_numpy()
        m_star = evaluate(test["label"][star], p[star]) if test["label"][star].nunique() == 2 else {}

        print(f"\n===== 测试集 FY{TEST_YEARS[0]}–{TEST_YEARS[-1]}（基准比例 {test.label.mean():.3f}）=====")
        for name, r in [("经验规则", rule), ("5 比率逻辑回归", m_ratio), (f"{args.model}|{args.features}", m)]:
            print(f"{name:28s} PR-AUC {r['pr_auc']:.3f}  ROC-AUC {r['roc_auc']:.3f}  前10%精确率 {r['prec_top10']:.3f}")
        if m_star:
            print(f"{'  其中科创板':28s} PR-AUC {m_star['pr_auc']:.3f}  前10%精确率 {m_star['prec_top10']:.3f}")
        json.dump({"model": m, "rule": rule, "ratio_logreg": m_ratio, "star": m_star},
                  open(out / f"test_{args.model}_{args.features}.json", "w"), indent=2)
        log_run(f"{args.model}|{args.features}|TEST",
                {"model": args.model, "features": args.features, "stage": "test"},
                {f"test_{k}": v for k, v in m.items()})
        return

    # ================= 在验证集上比较 =================
    rows = []
    def record(model_name, feat_name, n_feat, m):
        rows.append({"model": model_name, "features": feat_name, **{f"val_{k}": v for k, v in m.items()}})
        log_run(f"{model_name}|{feat_name}", {"model": model_name, "features": feat_name,
                                              "n_features": n_feat},
                {f"val_{k}": v for k, v in m.items()})

    # 基准 1：经验规则 —— 利润下降越多越危险，分数 = −利润增速
    record("rule_of_thumb", "profit_growth", 1, evaluate(val["label"], -val["profit_growth"].fillna(0)))

    # 基准 2：5 个比率的逻辑回归
    p = make_logreg().fit(train[RATIO5], train["label"]).predict_proba(val[RATIO5])[:, 1]
    record("ratio_logreg", "5 ratios", 5, evaluate(val["label"], p))

    # 主模型：2 个模型 × 2 套特征 = 4 次
    for m_name, make in MODELS.items():
        for f_name, cols in FEATURE_SETS.items():
            model = make().fit(train[cols], train["label"])
            p = model.predict_proba(val[cols])[:, 1]
            record(m_name, f_name, len(cols), evaluate(val["label"], p))

    res = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
    print(f"\n验证集基准比例 {val.label.mean():.3f}（随机排序的 PR-AUC 约等于它）\n")
    print(res.round(3).to_string(index=False))
    res.to_csv(out / "sklearn_val.csv", index=False)
    print("\n已保存 results/sklearn_val.csv。选验证集 PR-AUC 最高的，然后加 --final 跑测试集。")


if __name__ == "__main__":
    main()

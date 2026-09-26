"""
Step 6 — PyTorch MLP with three loss designs (plain BCE, weighted BCE, focal loss).

Usage
-----
    python src/train_torch.py --demo                       # synthetic data, checks the pipeline runs
    python src/train_torch.py                              # real table from Steps 2–3
    python src/train_torch.py --features raw+industry      # choose feature set
    python src/train_torch.py --final --loss focal         # retrain on train+val, score test ONCE

Expects data/processed/model_table.parquet with one row per company-year and columns:
    code, fiscal_year, industry, label, <feature columns>, <feature>_ind columns
"""
import argparse, json, pathlib, random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

try:                                   # MLflow is optional so the script still runs without it
    import mlflow
except ImportError:
    mlflow = None

# ----------------------------------------------------------------------------------------------
# Config — the time split from Step 4 (years are FEATURE years; label is year t+1)
# ----------------------------------------------------------------------------------------------
TRAIN_YEARS = range(2014, 2021)        # FY2014–2020
VAL_YEARS   = range(2021, 2023)        # FY2021–2022
TEST_YEARS  = range(2023, 2025)        # FY2023–2024 — touch once, with --final

NON_FEATURES = {"code", "name", "fiscal_year", "industry", "label", "is_star",
                "np_next", "rev_next", "yr_next"}          # never features (leakage / IDs)
MISSING_FLAG_THRESHOLD = 0.05          # add a "was missing" column if >5% missing in train


# ----------------------------------------------------------------------------------------------
# Metrics (same as Step 4, so numbers are comparable with LightGBM)
# ----------------------------------------------------------------------------------------------
def precision_at_top(y_true, score, frac=0.10):
    k = max(1, int(len(score) * frac))
    top = np.argsort(-score)[:k]
    return float(y_true[top].mean())


def evaluate(y, p):
    return {"pr_auc": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p),
            "prec_top10": precision_at_top(y, p)}


# ----------------------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).  gamma=0 -> alpha-weighted BCE."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # = -log(p_t)
        p_t = torch.exp(-bce)                                   # prob. assigned to the true class
        a_t = self.alpha * y + (1 - self.alpha) * (1 - y)       # alpha for positives, 1-alpha for negatives
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()     # easy examples (p_t≈1) get ~0 weight


def make_loss(name, n_pos, n_neg, gamma, alpha):
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]))
    if name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    raise ValueError(name)


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def make_mlp(n_features, hidden=(64, 32), dropout=0.2):
    layers, d = [], n_features
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, 1))     # one logit; sigmoid is inside the loss
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------------------------
# Preprocessing — everything is FITTED ON TRAIN ONLY, then applied to val/test
# ----------------------------------------------------------------------------------------------
class Preprocessor:
    def fit(self, df, cols):
        self.cols = cols
        miss = df[cols].isna().mean()
        self.flag_cols = [c for c in cols if miss[c] > MISSING_FLAG_THRESHOLD]
        self.median = df[cols].median()
        filled = df[cols].fillna(self.median)
        self.mean, self.std = filled.mean(), filled.std().replace(0, 1.0)
        return self

    def transform(self, df):
        X = df[self.cols].replace([np.inf, -np.inf], np.nan)
        flags = X[self.flag_cols].isna().astype(np.float32).add_suffix("_missing")
        X = (X.fillna(self.median) - self.mean) / self.std
        X = X.clip(-5, 5)                                  # guard against leftover extreme values
        out = pd.concat([X, flags], axis=1)
        return out.to_numpy(np.float32), list(out.columns)


# ----------------------------------------------------------------------------------------------
# Training loop with early stopping on validation PR-AUC
# ----------------------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def predict(model, X):
    model.eval()                                           # turns dropout OFF
    return torch.sigmoid(model(torch.from_numpy(X)).squeeze(1)).numpy()


def train_one(X_tr, y_tr, X_val, y_val, loss_fn, args, seed, fixed_epochs=None):
    """Train an MLP. If fixed_epochs is given (final retrain), no validation / early stopping."""
    set_seed(seed)
    model = make_mlp(X_tr.shape[1], dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=g)

    best = {"pr_auc": -1, "epoch": 0, "state": None}
    bad_epochs, history = 0, []
    n_epochs = fixed_epochs or args.epochs

    for epoch in range(1, n_epochs + 1):
        model.train()                                      # dropout ON
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(ds)

        if fixed_epochs:                                   # final retrain: just run N epochs
            continue

        val_pr = average_precision_score(y_val, predict(model, X_val))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pr_auc": val_pr})
        if val_pr > best["pr_auc"]:                        # keep the best checkpoint
            best = {"pr_auc": val_pr, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:                # early stopping
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best["epoch"], history


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def make_demo_data(seed=0):
    """Fake company-years with ~15% positives, only for checking the script runs end to end."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2014, 2026):
        n = 1500
        d = pd.DataFrame({
            "code": [f"{i:06d}" for i in range(n)], "fiscal_year": year,
            "industry": rng.choice(["软件", "电子", "机械", "医药", "化工"], n),
            "gross_margin": rng.normal(.3, .15, n), "roe": rng.normal(.08, .1, n),
            "op_margin": rng.normal(.08, .12, n), "rev_growth": rng.normal(.1, .3, n),
            "profit_growth": rng.normal(.05, .6, n), "debt_ratio": rng.uniform(.1, .9, n),
            "cash_ratio": rng.uniform(0, .4, n), "cfo_assets": rng.normal(.05, .08, n),
            "accruals": rng.normal(0, .08, n), "recv_vs_rev": rng.normal(0, .3, n),
            "inv_vs_rev": rng.normal(0, .3, n), "exp_ratio_chg": rng.normal(0, .03, n),
            "log_assets": rng.normal(22, 1.2, n)})
        z = (-2.2 - 4 * d.op_margin + 2 * d.debt_ratio + 5 * d.accruals - 3 * d.cfo_assets
             - .5 * d.rev_growth + .8 * d.recv_vs_rev - .3 * (d.log_assets - 22)
             + rng.normal(0, 1, n))
        d["label"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
        mask = rng.uniform(size=n) < .1                    # some missing values, like real data
        d.loc[mask, "cfo_assets"] = np.nan
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    raw = [c for c in df.columns if c not in NON_FEATURES]
    for f in raw:                                          # industry-relative versions (Step 3)
        grp = df.groupby(["fiscal_year", "industry"])[f]
        df[f + "_ind"] = (df[f] - grp.transform("median")) / (
            grp.transform(lambda s: s.quantile(.75) - s.quantile(.25)) + 1e-9)
    return df


def feature_sets(df):
    feats = [c for c in df.columns
             if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
    ind = [c for c in feats if c.endswith("_ind")]
    raw = [c for c in feats if c not in ind]
    return {"raw": raw, "raw+industry": raw + ind}


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/model_table.parquet")
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    ap.add_argument("--features", default="raw+industry", choices=["raw", "raw+industry"])
    ap.add_argument("--losses", default="bce,weighted,focal")
    ap.add_argument("--seeds", type=int, default=3, help="repeat each run to see the noise")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--final", action="store_true", help="retrain on train+val, score test once")
    ap.add_argument("--loss", default="focal", help="loss used with --final")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.set_num_threads(4)
    df = make_demo_data() if args.demo else pd.read_parquet(args.data)
    df = df[df["label"].notna()]                           # FY2025 rows have no label yet
    cols = feature_sets(df)[args.features]

    train = df[df.fiscal_year.isin(TRAIN_YEARS)]
    val   = df[df.fiscal_year.isin(VAL_YEARS)]
    test  = df[df.fiscal_year.isin(TEST_YEARS)]
    print(f"train {len(train):,}  val {len(val):,}  test {len(test):,}  "
          f"| positives: train {train.label.mean():.1%}  val {val.label.mean():.1%}  "
          f"| {len(cols)} features ({args.features})")

    out = pathlib.Path(args.out); out.mkdir(exist_ok=True)
    if mlflow:
        mlflow.set_experiment("dd-risk-screener")

    # ---------------- Model selection on validation years ----------------
    if not args.final:
        pre = Preprocessor().fit(train, cols)
        X_tr, names = pre.transform(train)
        X_val, _ = pre.transform(val)
        y_tr = train.label.to_numpy(np.float32)
        y_val = val.label.to_numpy()
        n_pos, n_neg = y_tr.sum(), len(y_tr) - y_tr.sum()

        rows = []
        for loss_name in args.losses.split(","):
            for seed in range(args.seeds):
                loss_fn = make_loss(loss_name, n_pos, n_neg, args.gamma, args.alpha)
                model, best_epoch, hist = train_one(X_tr, y_tr, X_val, y_val, loss_fn, args, seed)
                m = evaluate(y_val, predict(model, X_val))
                rows.append({"loss": loss_name, "seed": seed, "best_epoch": best_epoch,
                             **{f"val_{k}": v for k, v in m.items()}})
                print(f"{loss_name:9s} seed {seed}  epoch {best_epoch:3d}  "
                      f"PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  "
                      f"top10% prec {m['prec_top10']:.3f}")
                if mlflow:
                    with mlflow.start_run(run_name=f"mlp|{loss_name}|{args.features}|s{seed}"):
                        mlflow.log_params({"model": "mlp", "loss": loss_name,
                                           "features": args.features, "n_features": len(names),
                                           "gamma": args.gamma if loss_name == "focal" else None,
                                           "alpha": args.alpha if loss_name == "focal" else None,
                                           "seed": seed, "best_epoch": best_epoch,
                                           "lr": args.lr, "weight_decay": args.weight_decay})
                        mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
                        for h in hist:                     # learning curve in the MLflow UI
                            mlflow.log_metrics({"train_loss": h["train_loss"],
                                                "val_pr_auc_curve": h["val_pr_auc"]},
                                               step=h["epoch"])

        res = pd.DataFrame(rows)
        summary = (res.groupby("loss")
                      .agg(val_pr_auc=("val_pr_auc", "mean"), pr_auc_sd=("val_pr_auc", "std"),
                           val_prec_top10=("val_prec_top10", "mean"),
                           val_roc_auc=("val_roc_auc", "mean"),
                           median_epoch=("best_epoch", "median"))
                      .sort_values("val_pr_auc", ascending=False))
        print(f"\nBase rate (val): {y_val.mean():.3f}  ← PR-AUC of a random ranking\n")
        print(summary.round(3).to_string())
        res.to_csv(out / f"torch_val_{args.features}.csv", index=False)
        summary.to_csv(out / f"torch_val_summary_{args.features}.csv")
        print(f"\nSaved to {out}/. Pick the loss on VALIDATION, then run with --final --loss <name>.")
        return

    # ---------------- Final: retrain on train+val, score test years ONCE ----------------
    summ_path = out / f"torch_val_summary_{args.features}.csv"
    if not summ_path.exists():
        raise SystemExit(f"Run model selection first (no {summ_path}).")
    n_epochs = int(pd.read_csv(summ_path, index_col=0).loc[args.loss, "median_epoch"])

    trval = pd.concat([train, val])
    pre = Preprocessor().fit(trval, cols)
    X_trval, _ = pre.transform(trval)
    X_te, _ = pre.transform(test)
    y_trval = trval.label.to_numpy(np.float32)
    y_te = test.label.to_numpy()
    n_pos, n_neg = y_trval.sum(), len(y_trval) - y_trval.sum()

    preds = []
    for seed in range(args.seeds):                         # average seeds = small ensemble
        loss_fn = make_loss(args.loss, n_pos, n_neg, args.gamma, args.alpha)
        model, _, _ = train_one(X_trval, y_trval, None, None, loss_fn, args, seed,
                                fixed_epochs=n_epochs)
        preds.append(predict(model, X_te))
    p = np.mean(preds, axis=0)
    m = evaluate(y_te, p)
    star = test["code"].astype(str).str.startswith("688").to_numpy()
    m_star = evaluate(y_te[star], p[star]) if star.sum() > 50 and y_te[star].sum() > 0 else {}

    print(f"TEST (FY{TEST_YEARS[0]}–{TEST_YEARS[-1]}) — MLP, {args.loss} loss, {n_epochs} epochs, "
          f"{args.seeds}-seed average")
    print(f"  base rate {y_te.mean():.3f}  PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  "
          f"top10% prec {m['prec_top10']:.3f}")
    if m_star:
        print(f"  STAR only ({star.sum()} rows): PR-AUC {m_star['pr_auc']:.3f}  "
              f"top10% prec {m_star['prec_top10']:.3f}")
    json.dump({"all": m, "star": m_star, "loss": args.loss, "epochs": n_epochs},
              open(out / f"torch_test_{args.loss}_{args.features}.json", "w"), indent=2)
    if mlflow:
        with mlflow.start_run(run_name=f"mlp|{args.loss}|{args.features}|TEST"):
            mlflow.log_params({"model": "mlp", "loss": args.loss, "features": args.features,
                               "stage": "test", "epochs": n_epochs})
            mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})


if __name__ == "__main__":
    main()

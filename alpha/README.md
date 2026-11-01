# Alpha test: do the screener's risk scores predict A-share returns?

The screener predicts **fundamental** deterioration (48% hit rate in the top 10%).
This folder asks the next question: **is that risk already in the stock price,
or do high-risk stocks earn lower returns afterwards?**

```
walk-forward scores  →  monthly prices  →  forward returns  →  IC / portfolios / Fama-MacBeth  →  report
```

## Run (from the repo root)

```bash
pip install -r alpha/requirements.txt
python alpha/walk_forward.py                       # 1. out-of-sample score for every year (~1 min)
python alpha/walk_forward.py --params default      #    stricter version, no tuned hyper-parameters
python -m pytest -q alpha/test_alpha.py            # 2. pipeline checks on fake prices
python alpha/fetch_prices.py --check               # 3. test the AKShare connection on one stock
python alpha/fetch_prices.py                       #    all ~5,100 stocks (30-60 min, resumable)
python alpha/run.py                                # 4. tests → alpha/output/report.md + summary.png
python alpha/run.py --scores alpha/output/scores_walkforward_default.csv --out alpha/output_default
```

## Design choices (fixed before looking at returns)

| Choice | Rule | Why |
|---|---|---|
| Scores | Expanding-window retrain every year: score FY *Y* with a model trained on FY2014…*Y*−1 | The original split only has 2 out-of-sample years; scoring earlier years with the final model would leak future outcomes into the score |
| Timing | FY *Y* score usable from 1 May of *Y*+1, kept ≤ 400 days | A-share annual reports are due by 30 April |
| Universe | All scored stocks, ≥ 6 months after listing, traded in the formation month | Avoid IPO pops and suspended stocks |
| Main test | All stocks (EW) vs all stocks minus the riskiest 10% | Shorting A-shares is very limited, so the realistic use is avoiding names |
| Controls | Fama-MacBeth with size, book-to-market, momentum, 1-month reversal (ranked within month) | Risky firms are small, and small A-shares carry a premium (shell value) |

## Walk-forward classification check (out of sample every year)

| Scored FY | Trained on | Base rate | PR-AUC | Precision top 10% |
|---|---|---|---|---|
| 2017 | 2014–16 | 10.3% | 0.280 | 30.2% |
| 2018 | 2014–17 | 10.7% | 0.308 | 34.4% |
| 2019 | 2014–18 | 15.3% | 0.388 | 46.1% |
| 2020 | 2014–19 | 10.2% | 0.294 | 32.8% |
| 2021 | 2014–20 | 18.9% | 0.413 | 52.4% |
| 2022 | 2014–21 | 18.3% | 0.347 | 41.1% |
| 2023 | 2014–22 | 19.4% | 0.430 | 51.8% |
| 2024 | 2014–23 | 16.1% | 0.389 | 46.7% |

Every year the riskiest 10% deteriorate at roughly 3× the base rate.

## Pipeline checks (fake prices, not results)

- Planted effect is recovered (Fama-MacBeth t ≈ −7.9).
- **Size trap:** with no score effect but a small-cap premium, a raw Q1−Q5 sort says "risky stocks do *better*" (t ≈ −3.3); Fama-MacBeth with a size control correctly finds nothing (t ≈ −0.4).

## Limitations

- **Survivorship:** the financial data contain almost only firms listed today.
- **Possibly restated numbers:** Eastmoney's announcement dates for each fiscal year are mostly *later* than the original annual report (e.g. most FY2024 rows show dates from Oct 2025 onward), so some values may be restated, not exactly what investors saw in April.
- Float market cap is estimated as monthly amount ÷ turnover; no ST flags or price-limit handling.
- LightGBM hyper-parameters were tuned on FY2021–22; `--params default` removes that.

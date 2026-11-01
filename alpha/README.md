# Do the screener's risk scores predict A-share returns?

The screener predicts **fundamental** deterioration. This folder asks the next question:
is that risk already in the stock price, or do high-risk stocks earn lower returns afterwards?

```
walk-forward scores -> monthly prices -> forward returns -> IC / portfolios / Fama-MacBeth -> report
```

## Result (May 2018 - Sep 2026, ~4,200 stocks a month, 101 months)

**Overall, no.** The score does not predict next-month returns once size is accounted for.

| Test | Result | t (Newey-West) |
|---|---|---|
| Monthly rank IC | -0.005 | -0.52 |
| Long-only screen: drop the riskiest 10% vs hold all | -0.02% / month | -0.86 |
| Fama-MacBeth, controls for size, value, momentum, reversal | riskiest vs safest -0.25% / month | -0.83 |
| Permutation test (scores shuffled once per year, compares t-stats) | p = 0.64 | |

**The size trap is real.** Sorted raw, the safest quintile earned the least (0.69%/month) and the riskiest 10% the most (1.01%/month). Risky firms are small, and small A-shares earned a premium (size coefficient t = -2.6). Adding size to the regression flips the score's sign from +0.22% to -0.25% a month.

**The effect splits by size.** The score works in the direction expected among mid- and large-cap stocks and reverses among small caps, so the two cancel out:

| Size group (within month) | Mean IC | t |
|---|---|---|
| Smallest third | +0.019 | 1.58 |
| Middle third | -0.021 | -2.41 |
| Largest third | -0.030 | -2.94 |
| All but the smallest 30% | -0.021 | -1.99 |

Without the smallest 30%, the Fama-MacBeth score coefficient is -0.50%/month (t = -1.64): partly explained by value and momentum. The size split was written into the code before any returns were seen. Why small risky stocks do not lag is not tested here (one guess: A-share speculation on restructuring and shell value).

Untuned LightGBM defaults (`--params default`) give the same picture (IC t = -0.44; largest third t = -2.89).

## Scores: walk-forward, out of sample every year

Each fiscal year Y is scored by a model trained only on FY2014...Y-1, and the score is used from 1 May of Y+1 (A-share annual reports are due by 30 April).

| Scored FY | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|
| Riskiest 10% that deteriorated | 30% | 34% | 46% | 33% | 52% | 41% | 52% | 47% |
| Base rate | 10% | 11% | 15% | 10% | 19% | 18% | 19% | 16% |

So the score does find companies that run into trouble (2.2-3.2x the base rate every year); the market appears to price that in.

## Run (from the repo root)

```bash
pip install -r alpha/requirements.txt
python alpha/walk_forward.py                         # scores, tuned params
python alpha/walk_forward.py --params default        # scores, LightGBM defaults
python -m pytest -q alpha/test_alpha.py              # pipeline checks on simulated prices
python alpha/fetch_prices_sina.py --check            # one stock
python alpha/fetch_prices_sina.py --part 1/2         # all stocks, split over two terminals
python alpha/fetch_prices_sina.py --part 2/2         #   (resumable, ~10 h; Eastmoney version: fetch_prices.py)
python alpha/run.py                                  # -> alpha/output/report.md + summary.png
python alpha/run.py --scores alpha/output/scores_walkforward_default.csv --out alpha/output_default
```

## Pipeline checks (simulated prices, not results)

- A planted effect is recovered (Fama-MacBeth t = -7.9).
- Size trap: with no score effect but a small-cap premium, a raw sort says risky stocks win (t = -3.3); Fama-MacBeth with size finds nothing (t = -0.4).

## Data notes and limitations

- Prices: Sina via AKShare (Eastmoney blocked repeated requests). Float market cap = month-end close x outstanding shares. 14 of 5,112 stocks had no Sina data.
- Survivorship: the financial data contain almost only firms still listed today, so the most distressed (delisted) firms are missing.
- Possibly restated numbers: Eastmoney's announcement dates are mostly later than the original annual report, so some values may be restated versions, not what investors saw in April.
- No ST flags or price-limit handling; equal and float-cap weights only; one sample period.

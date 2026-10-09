# Due-Diligence Risk Screener

A first-pass screening tool that flags which listed Chinese companies are likely to **deteriorate financially next year**, and explains why in plain language.

It is built as an investment workflow rather than a single model score: screen an industry, compare a shortlist side by side, and read a short plain-English summary of each company.

**Live demo:** https://dd-risk-screener.streamlit.app

## Motivation

During an internship at Zhejiang Yongxi Investment Management, I saw specialized companies evaluated through site visits and manual document review, with no structured database behind the first pass. This project tests how much of that first pass machine learning can support: which companies deserve a closer look, and what to ask about when you get there.

## How it fits together

**Python** pulls and cleans the filings and engineers the features → **SQLite** stores the company-year table for querying → **LightGBM** scores each company and **SHAP** explains the score → a **Streamlit** app turns the output into a screening and comparison workflow.

| Step | File | What it does |
|---|---|---|
| Ingest | `src/fetch_data.py` | Downloads 48 bulk datasets (four statements × twelve years) with timeouts, retries and resumable downloads |
| Transform | `src/build_dataset.py` | Merges, cleans, labels and builds features into one company-year table |
| Store and query | `src/db.py` | Loads the table into SQLite; screening, ranking and history queries in SQL |
| Model | `src/train_sklearn.py`, `src/train_torch.py` | Baselines, LightGBM and a PyTorch MLP, tracked in MLflow |
| Explain | `src/explain.py` | Scores FY2025 companies and finds each one's top risk drivers with SHAP |
| Workflow | `src/screening.py`, `app.py` | Screen, compare, summarize and export |

## Results (held-out test years, FY2023–2024)

| Model | PR-AUC | Precision in top 10% |
|---|---|---|
| Random ranking (base rate) | 0.178 | 17.8% |
| Rule of thumb (profit fell this year) | 0.214 | 20.7% |
| Ratio logistic regression (5 Altman-style ratios) | 0.216 | 24.4% |
| PyTorch MLP (BCE loss) | 0.366 | 42.7% |
| **LightGBM, raw + industry-relative features** | **0.402** | **48.1%** |
| LightGBM, STAR Market companies only | 0.357 | 46.7% |

Of the 10% of companies LightGBM flags as riskiest, **48% actually deteriorated the following year**, twice the rate of the ratio-based baseline.

The model was chosen on validation years (FY2021–2022) and scored on the test years **once**. Test PR-AUC (0.402) is slightly higher than validation (0.378), which suggests the model is not overfitted.

## Data

- **Source:** annual reports of all A-share companies, pulled with [AKShare](https://github.com/akfamily/akshare) from Eastmoney's public filings data (earnings summary, balance sheet, income statement, cash-flow statement), FY2014–FY2025.
- **Survivorship bias:** the source returns almost only companies that are still listed. Of 5,112 companies in the database, 5,103 have an FY2025 report and only 9 stop earlier, far fewer than the number of A-share delistings over the period (see Limitations).
- Banks, brokers and insurers are excluded (their statements have a different structure). Beijing Stock Exchange companies are excluded.
- **Result:** 51,732 labelled company-years (about 4,500–5,100 companies per year), plus 5,103 FY2025 companies scored "live".

## Label

Each row is one company in fiscal year *t*. Features use only the year-*t* report (and year *t−1* for growth rates). The label is **1 ("deteriorated")** if, in year *t+1*, either:

- net profit turns from positive to zero or negative, or
- revenue falls more than 20%.

A year-*t* annual report is published by 30 April of *t+1*, so a prediction is made in May of *t+1* and covers the rest of that year. The share of deteriorating companies rose from about 10–15% (FY2014–2020) to about 16–19% (FY2021–2024).

## Features

13 accounting features, each also standardized **within its industry and year** (median and interquartile range), because a 60% debt ratio is normal for a manufacturer but alarming for a software firm.

| Group | Feature | Formula |
|---|---|---|
| Profitability | Gross margin, ROE | from report |
| Profitability | Operating margin | operating profit / revenue |
| Growth | Revenue growth, profit growth | year over year |
| Leverage & liquidity | Debt ratio | total liabilities / total assets |
| Leverage & liquidity | Cash ratio | cash / total assets |
| Leverage & liquidity | Operating cash flow / assets | CFO / total assets |
| Red flag | Accruals | (net profit − CFO) / total assets |
| Red flag | Receivables vs revenue | receivables growth − revenue growth |
| Red flag | Inventory vs revenue | inventory growth − revenue growth |
| Red flag | Expense ratio change | Δ (selling + admin) / revenue |
| Size | Log assets | log(total assets) |

Each feature is winsorized at the 1st and 99th percentiles within each year.

## Method

**Time-based split, never random.** A random split would let the model learn from 2023 and be tested on 2019.

| Set | Feature years | Label years | Used for |
|---|---|---|---|
| Train | FY2014–2020 | 2015–2021 | fitting |
| Validation | FY2021–2022 | 2022–2023 | choosing model, features, loss |
| Test | FY2023–2024 | 2024–2025 | final numbers, touched once |

**Metrics:** PR-AUC (main metric for rare positives), ROC-AUC, and precision in the top 10%. Accuracy is not used, because predicting "never deteriorates" already scores over 80%.

All runs are tracked in MLflow:

![Validation PR-AUC by model](docs/mlflow_val_pr_auc.png)

## Findings

**1. Industry-relative features help a little, after fixing a data bug.** LightGBM validation PR-AUC went from 0.374 (raw) to 0.378 (raw + industry). An early version showed no gain. The cause was industries where almost every company had the same gross margin: the interquartile range was near zero, producing scores of about 400 million. Scores are now left blank when the range is near zero and capped at ±10.

**2. Loss-function design did not change ranking quality.** I trained the same PyTorch MLP (64→32→1, dropout 0.2, Adam, early stopping on validation PR-AUC) with plain BCE, weighted BCE and focal loss (γ = 2, α = 0.75), 3 seeds each:

| Loss | Validation PR-AUC (mean ± sd) |
|---|---|
| BCE | 0.343 ± 0.003 |
| Focal | 0.342 ± 0.004 |
| Weighted BCE | 0.339 ± 0.003 |

The differences are within seed-to-seed noise. Re-weighting mostly shifts predicted probabilities up or down, not the order of companies, and PR-AUC depends only on that order.

**3. LightGBM beat the neural network** (0.402 vs 0.366 test PR-AUC), a common pattern on tabular data: trees handle threshold rules, missing values and features on very different scales natively, and 31,000 training rows is small for a neural network.

## Explanations

SHAP values from LightGBM show which features push each company's risk up. Low ROE and weak revenue growth raise risk the most, followed by high leverage, high accruals and weak operating cash flow.

![SHAP summary](results/shap_summary.png)

Sanity check: the model was never shown the exchange's ST (special treatment) flag, yet two of its ten riskiest FY2025 companies carry a *ST warning, and real-estate developers and their suppliers dominate the list.

## Database layer (SQL)

The cleaned company-year table is loaded into a relational database. The Streamlit app queries it directly (see Screening app), and the same queries can be run from the command line:

**Python (ingest, clean, engineer features) → SQLite (structured storage and querying) → model and analytics → Streamlit interface**

| Table | One row per | Key | Contents |
|---|---|---|---|
| `companies` | company | `code` | name, latest industry, STAR Market flag |
| `financials` | company-year | (`code`, `fiscal_year`) | 13 financial ratios and the next-year deterioration label |
| `scores` | company (FY2025) | `code` | model probability, risk percentile, top three risk drivers |

The composite primary key on `financials` rejects duplicate company-years, and `build` reports the share of missing values per ratio, duplicate keys and orphan rows every time the database is rebuilt.

```bash
python src/db.py build                                   # create data/screener.db
python src/db.py summary                                 # companies and deterioration rate by year
python src/db.py industries --year 2025                  # industry averages (GROUP BY ... HAVING)
python src/db.py screen --industry 半导体 --min-rev-growth 0.1 --max-debt 0.5
python src/db.py rank --industry 半导体 --top 10          # lowest-risk companies within an industry
python src/db.py history 688981                          # one company's financials over time
```

Two of the queries, as written in `src/db.py`:

```sql
-- Lowest-risk companies within each industry (window function)
WITH ranked AS (
    SELECT c.industry, c.code, c.name, s.risk_score,
           RANK()   OVER (PARTITION BY c.industry ORDER BY s.risk_score) AS rank_in_industry,
           COUNT(*) OVER (PARTITION BY c.industry)                       AS industry_size
    FROM scores s
    JOIN companies c ON c.code = s.code
)
SELECT * FROM ranked WHERE rank_in_industry <= ?;

-- One company's history with year-over-year change (LAG, skipping gaps in the filing record)
SELECT f.fiscal_year, f.op_margin,
       CASE WHEN LAG(f.fiscal_year) OVER w = f.fiscal_year - 1
            THEN f.op_margin - LAG(f.op_margin) OVER w END AS op_margin_change
FROM financials f
WHERE f.code = ?
WINDOW w AS (PARTITION BY f.code ORDER BY f.fiscal_year);
```

All queries pass values as parameters rather than building SQL strings, and the app opens the database read-only (mode=ro), so nothing typed into the page can change it. tests/test_db.py checks every query against a four-company in-memory database whose answers can be worked out by hand, and checks that the read-only connection rejects writes (11 tests).

## Screening app

A predictive score alone is not something an investor can act on, so the app is built around a first-pass screening workflow rather than a single number.

**1. Screen.** Choose one or more industries and optional filters (maximum risk percentile, minimum revenue growth, minimum ROE, maximum debt ratio, positive operating cash flow). The app returns a ranked shortlist: the lowest-risk candidates, or the highest-risk watch list.

**2. Compare.** Pick two to four companies from the shortlist to see twelve ratios side by side, the industry median when they share an industry, and which company is best on each ratio (direction-aware: a lower debt ratio is better).

**3. Summarize.** Each selected company gets a three-to-five sentence plain-English summary: where it ranks, what stands out against peers, what the model is concerned about, what is missing from the report, and a first-pass conclusion. Summaries come from fixed templates, not a language model, so every sentence traces back to a number in the table. A concern is only stated with figures when the figure really is worse than the industry median.

**4. Export.** The shortlist and summaries download as a CSV.

A second tab shows the full profile of any single company: risk percentile, rank within its industry, the top three red flags, a peer table, and **questions for the site visit** generated from the flags (for example, receivables flag → "Ask about customer payment terms and overdue receivables").

Two views are served straight from SQLite. The company profile includes a financial history table and chart, built with a LAG() window function that shows each year’s change in operating margin and debt ratio (left blank when the previous year is missing). A third tab, Industry ranking, lists the lowest-risk companies within any industry using RANK() OVER (PARTITION BY industry ORDER BY risk).

The screening, comparison and summary logic lives in `src/screening.py`, separate from the interface, and is covered by `tests/test_screening.py` (12 tests on a hand-checkable toy dataset).

![Screener](docs/app_screenshot.png)

## Limitations

- Listed companies stand in for the private targets a fund would actually evaluate.
- Annual data only. Quarterly reports could give earlier warnings.
- Eastmoney may serve restated figures, which can be cleaner than what investors saw at the time.
- **Survivorship bias.** I first assumed that pulling by report date would keep delisted companies. A query on the database showed it does not: only 9 of 5,112 companies have a last report before FY2025. The worst outcomes (companies that failed and were delisted) are therefore missing, so the deterioration rate is under-counted and the model has not been tested on them. The reported precision applies to companies that survived to 2025.
- A company with no report in year *t+1* has no label and is dropped (111 company-years).
- Much of the signal is persistence: weak companies tend to stay weak. The subtler red flags (accruals, receivables, inventory) add to this but do not dominate it.
- AKShare column names can change between versions.
- The deployed database (app_data/screener.db) is a snapshot. After re-running the pipeline it has to be rebuilt with python src/db.py build and copied into app_data/, or it will drift from the scored files.
- The label thresholds (profit turning to loss, revenue −20%) are my choice. They were fixed before looking at the results.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-full.txt   # full pipeline; requirements.txt holds only what the deployed app needs

python src/fetch_data.py              # Step 1: download FY2014–2025 (about 1 hour)
python src/build_dataset.py           # Steps 2–3: label + features
python src/train_sklearn.py           # Steps 4–5: baselines, logistic regression, LightGBM (validation)
python src/train_sklearn.py --final --model lgbm --features raw+industry   # test set, once
python src/train_torch.py             # Step 6: PyTorch, three losses (validation)
python src/train_torch.py --final --loss bce                               # test set, once
python src/explain.py                 # Step 7: SHAP + FY2025 scores
python src/db.py build                # Step 8: load everything into SQLite (data/screener.db)
python src/db.py rank --top 5         #         example query: lowest-risk companies per industry
streamlit run app.py                  # Step 9: screening app
mlflow ui                             # experiment tracking, http://127.0.0.1:5000
```

Tests (no data needed; both use small hand-checkable datasets):

```bash
python tests/test_screening.py        # screening, comparison and summary logic (12 tests)
python tests/test_db.py               # database schema, SQL queries and read-only access (11 tests)
```

*Personal project. Not investment advice.*

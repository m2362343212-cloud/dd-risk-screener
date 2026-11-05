# Robustness of the screener-score -> A-share return test

- Data: REAL (walk-forward scores + Sina monthly prices)
- Each row repeats the main test with one change. FM = Fama-MacBeth coefficient of the score (riskiest minus safest, % per month, controls: size, book-to-market, momentum, reversal). Screen = hold all stocks except the riskiest 10%, minus holding all stocks.
- If the conclusion is 'no effect', the thing to look for is whether any row changes the sign or pushes |t| above 2.

## 1. Variants
| variant            |   months |   stocks_per_month |   mean_IC |   IC_t |   FM_score_coef_%_per_month |   FM_t |   screen_gain_%_per_month |   screen_t |
|:-------------------|---------:|-------------------:|----------:|-------:|----------------------------:|-------:|--------------------------:|-----------:|
| baseline           |  101.000 |           4168.000 |    -0.005 | -0.516 |                      -0.248 | -0.827 |                    -0.017 |     -0.859 |
| winsorized returns |  101.000 |           4168.000 |    -0.005 | -0.516 |                      -0.352 | -1.258 |                    -0.006 |     -0.315 |
| liquid 80%         |  101.000 |           3335.000 |    -0.014 | -1.403 |                      -0.370 | -1.210 |                    -0.003 |     -0.153 |
| signal +1 month    |  100.000 |           4158.000 |    -0.003 | -0.266 |                      -0.124 | -0.405 |                    -0.021 |     -1.072 |
| first half         |   50.000 |           3522.000 |    -0.005 | -0.310 |                      -0.141 | -0.317 |                    -0.030 |     -1.139 |
| second half        |   51.000 |           4801.000 |    -0.005 | -0.461 |                      -0.352 | -0.879 |                    -0.003 |     -0.124 |

## 2. Transaction costs on the long-only screen
Round-trip costs are assumptions for sensitivity, not measured. Turnover counts only stocks entering or leaving a portfolio and ignores price drift between months; both legs rebalance to equal weights, so most of the drift cost cancels in the difference, but it would not for a stand-alone portfolio.

|   round_trip_cost_bps |   net_screen_gain_%_per_month |   t_NW |   avg_one_way_turnover_screened |   avg_one_way_turnover_all |   months |
|----------------------:|------------------------------:|-------:|--------------------------------:|---------------------------:|---------:|
|                     0 |                        -0.020 | -0.981 |                           0.013 |                      0.006 |  100.000 |
|                    20 |                        -0.020 | -1.017 |                           0.013 |                      0.006 |  100.000 |
|                    40 |                        -0.021 | -1.053 |                           0.013 |                      0.006 |  100.000 |
|                    80 |                        -0.022 | -1.126 |                           0.013 |                      0.006 |  100.000 |

## Still not covered
- ST flags and price limits (month-end bars; no historical ST status in the data).
- Survivorship bias (delisted firms missing from the financial data).
- One market, one sample period.
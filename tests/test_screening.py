"""
筛选 / 对比 / 总结 逻辑的测试。用 5 家编出来的公司，结果可以手算核对。
运行：  python tests/test_screening.py      或      pytest tests/
"""
import pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from screening import METRICS, add_industry_rank, comparison_table, company_summary, results_table, screen


def toy():
    """5 家公司：软件 3 家（A 最安全，C 最危险），钢铁 2 家。"""
    df = pd.DataFrame({
        "code": ["000001", "000002", "688003", "000004", "000005"],
        "name": ["A", "B", "C", "D", "E"],
        "industry": ["Software", "Software", "Software", "Steel", "Steel"],
        "is_star": [False, False, True, False, False],
        "risk_score": [0.05, 0.20, 0.60, 0.10, 0.40],
        "rev_growth": [0.30, 0.05, -0.25, 0.10, np.nan],
        "roe": [0.18, 0.08, -0.10, 0.12, 0.02],
        "debt_ratio": [0.20, 0.45, 0.80, 0.55, 0.70],
        "cfo_assets": [0.10, 0.03, -0.05, 0.06, 0.01],
    })
    for f in METRICS:                                    # 其余指标填 0，行业版分数先全填 0
        if f not in df:
            df[f] = 0.0
        df[f + "_ind"] = 0.0
    df["rev_growth_ind"] = [1.5, 0.0, -2.0, 0.0, np.nan]  # A 的营收增速明显好于同行
    df["debt_ratio_ind"] = [-1.0, 0.0, 1.4, 0.0, 0.0]     # A 的负债率明显低于同行（越低越好）
    df["pct_all"] = df["risk_score"].rank(pct=True)
    df["flag_1"], df["flag_1_shap"] = ["roe", "roe", "rev_growth_ind", "debt_ratio", "debt_ratio"], [-0.1, 0.2, 0.9, 0.1, 0.5]
    df["flag_2"], df["flag_2_shap"] = ["debt_ratio"] * 5, [-0.2, 0.1, 0.5, 0.0, 0.2]
    df["flag_3"], df["flag_3_shap"] = ["rev_growth"] * 5, [-0.3, 0.0, 0.3, -0.1, 0.1]
    peers = df.groupby("industry")[list(METRICS)].median()
    return add_industry_rank(df), peers


def test_candidates_are_sorted_safest_first():
    df, _ = toy()
    assert screen(df)["name"].tolist() == ["A", "D", "B", "E", "C"]


def test_watchlist_is_sorted_riskiest_first():
    df, _ = toy()
    assert screen(df, objective="watchlist", top_n=2)["name"].tolist() == ["C", "E"]


def test_industry_and_star_filters():
    df, _ = toy()
    assert screen(df, industries=["Steel"])["name"].tolist() == ["D", "E"]
    assert screen(df, star_only=True)["name"].tolist() == ["C"]


def test_financial_filters_exclude_missing_values():
    df, _ = toy()
    # E 的营收增速缺失：一旦启用这个条件，E 就不能入选
    assert screen(df, min_rev_growth=0.0)["name"].tolist() == ["A", "D", "B"]
    assert screen(df, max_debt_ratio=0.5, positive_cfo=True)["name"].tolist() == ["A", "B"]
    assert screen(df, min_roe=0.5).empty


def test_industry_rank():
    df, _ = toy()
    a = df[df.name == "A"].iloc[0]
    c = df[df.name == "C"].iloc[0]
    assert (a["ind_rank"], a["ind_n"]) == (1, 3)
    assert (c["ind_rank"], c["ind_n"]) == (3, 3)


def test_missing_industry_does_not_crash():
    df, peers = toy()
    df.loc[df.name == "E", "industry"] = None              # 真实数据里有些公司没有行业
    df = add_industry_rank(df.drop(columns=["ind_rank", "ind_n"]))
    e = df[df.name == "E"].iloc[0]
    assert e["industry"] == "Unknown" and e["ind_rank"] == 1
    assert 3 <= len(company_summary(e, peers)) <= 5
    assert len(comparison_table(df[df.name.isin(["D", "E"])], peers)) == 13


def test_results_table_is_formatted():
    df, _ = toy()
    t = results_table(screen(df))
    assert t.loc[0, "Revenue growth"] == "30.0%"
    assert t.loc[0, "Top risk driver"] == "none material"          # A 没有任何特征在推高风险
    assert t.loc[3, "Revenue growth"] == "—"                   # E 缺失


def test_comparison_picks_best_with_direction():
    df, peers = toy()
    t = comparison_table(df[df.name.isin(["A", "B", "C"])], peers)
    assert "Industry median" in t.columns                      # 三家都是软件 → 显示行业中位数
    assert t.loc["Debt ratio", "Best"] == "A (000001)"         # 负债率越低越好
    assert t.loc["Revenue growth", "Best"] == "A (000001)"
    mixed = comparison_table(df[df.name.isin(["A", "D"])], peers)
    assert "Industry median" not in mixed.columns              # 不同行业 → 不显示


def test_summary_safe_company():
    df, peers = toy()
    s = company_summary(df[df.name == "A"].iloc[0], peers)
    assert 3 <= len(s) <= 5
    assert "ranks it 1 of 3" in s[0]
    assert "revenue growth (30.0% vs. 5.0% industry median)" in s[1]
    assert "low leverage (20.0% vs. 45.0% industry median)" in s[1]
    assert "No single feature" in s[2]
    assert "lower-risk" in s[-1]


def test_summary_risky_company():
    df, peers = toy()
    s = company_summary(df[df.name == "C"].iloc[0], peers)
    assert "riskiest" in s[0]
    assert "weak revenue momentum (-25.0% vs. 5.0% industry median)" in s[2]
    assert "high leverage (80.0% vs. 45.0% industry median)" in s[2]
    assert "high-risk" in s[-1]


def test_summary_does_not_contradict_itself():
    df, peers = toy()
    # D 的负债率 55% 低于钢铁行业中位数 62.5%：模型标了它，但总结不能说成 "high leverage (55% vs 62.5%)"
    s = company_summary(df[df.name == "D"].iloc[0], peers)
    assert "high leverage" not in " ".join(s)
    assert "not worse than the industry median" in s[2]


def test_summary_mentions_missing_data():
    df, peers = toy()
    s = company_summary(df[df.name == "E"].iloc[0], peers)
    assert any("Not reported" in x and "Revenue growth" in x for x in s)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\n{len(tests)} tests passed")

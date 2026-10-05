"""
SQL 数据库层的测试：用内存数据库 + 4 家编出来的公司，答案可以手算。
运行：  python tests/test_db.py      或      pytest tests/
"""
import pathlib, sqlite3, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import db


def toy_conn():
    """软件：A（3 年）、B（2 年，行业缺失后来补上）；钢铁：C、D。"""
    rows = [
        # code, name, industry, year, rev_growth, op_margin, debt_ratio, label
        ("000001", "A", "Software", 2023, 0.10, 0.20, 0.30, 0),
        ("000001", "A", "Software", 2024, 0.20, 0.25, 0.28, 0),
        ("000001", "A", "Software", 2025, 0.30, 0.22, 0.25, None),
        ("688002", "B", None,       2023, -0.10, 0.05, 0.60, 1),
        ("688002", "B", "Software", 2025, 0.05, 0.08, 0.55, None),      # 缺 2024 年
        ("000003", "C", "Steel",    2025, 0.15, 0.10, 0.70, None),
        ("000004", "D", "Steel",    2025, np.nan, 0.12, 0.40, None),
    ]
    t = pd.DataFrame(rows, columns=["code", "name", "industry", "fiscal_year",
                                    "rev_growth", "op_margin", "debt_ratio", "label"])
    t["is_star"] = t["code"].str.startswith("688")
    for f in db.FEATS:
        if f not in t:
            t[f] = 0.0
    scored = pd.DataFrame({"code": ["000001", "688002", "000003", "000004"], "fiscal_year": 2025,
                           "risk_score": [0.05, 0.40, 0.30, 0.10], "pct_all": [0.25, 1.0, 0.75, 0.5],
                           "flag_1": "roe", "flag_2": "debt_ratio", "flag_3": "rev_growth"})
    conn = sqlite3.connect(":memory:")
    counts = db.build(conn, t, scored)
    return conn, counts


def test_build_row_counts():
    _, counts = toy_conn()
    assert counts == {"companies": 4, "financials": 7, "scores": 4}


def test_company_gets_latest_industry():
    conn, _ = toy_conn()
    # B 在 2023 年没有行业，2025 年有：公司表应该用最新的
    assert conn.execute("SELECT industry, is_star FROM companies WHERE code='688002'").fetchone() == ("Software", 1)


def test_primary_key_blocks_duplicate_company_year():
    conn, _ = toy_conn()
    try:
        conn.execute("INSERT INTO financials (code, fiscal_year) VALUES ('000001', 2025)")
        assert False, "duplicate company-year should be rejected"
    except sqlite3.IntegrityError:
        pass


def test_yearly_summary():
    conn, _ = toy_conn()
    s = db.yearly_summary(conn).set_index("fiscal_year")
    assert s.loc[2023, "companies"] == 2 and s.loc[2023, "deterioration_rate"] == 0.5
    assert s.loc[2025, "companies"] == 4 and pd.isna(s.loc[2025, "deterioration_rate"])


def test_industry_summary():
    conn, _ = toy_conn()
    s = db.industry_summary(conn, 2025, min_companies=2).set_index("industry")
    assert s.loc["Software", "companies"] == 2
    assert s.loc["Steel", "avg_debt_ratio"] == 0.55             # (0.70 + 0.40) / 2
    assert s.loc["Steel", "avg_rev_growth"] == 0.15             # D 缺失，AVG 只算 C
    assert db.industry_summary(conn, 2025, min_companies=3).empty


def test_screen_filters_are_optional():
    conn, _ = toy_conn()
    assert db.screen(conn, 2025)["name"].tolist() == ["A", "C", "B", "D"]        # 按营收增速从高到低，缺失排最后
    assert db.screen(conn, 2025, industry="Steel")["name"].tolist() == ["C", "D"]
    assert db.screen(conn, 2025, min_rev_growth=0.10)["name"].tolist() == ["A", "C"]   # D 缺失 → 不入选
    assert db.screen(conn, 2025, max_debt_ratio=0.5)["name"].tolist() == ["A", "D"]
    assert db.screen(conn, 2025, industry="Software", min_op_margin=0.2, limit=5)["name"].tolist() == ["A"]
    assert db.screen(conn, 2024)["risk_score"].isna().all()     # 2024 年没有打分，LEFT JOIN 留空


def test_rank_in_industry_uses_window_function():
    conn, _ = toy_conn()
    r = db.rank_in_industry(conn, top=1)
    assert r[["industry", "name", "rank_in_industry", "industry_size"]].values.tolist() == \
        [["Software", "A", 1, 2], ["Steel", "D", 1, 2]]
    assert db.rank_in_industry(conn, industry="Steel", top=10)["name"].tolist() == ["D", "C"]


def test_company_history_and_year_over_year_change():
    conn, _ = toy_conn()
    h = db.company_history(conn, "000001")
    assert h["fiscal_year"].tolist() == [2023, 2024, 2025]
    assert pd.isna(h.loc[0, "op_margin_change"])                # 第一年没有上一年
    assert h.loc[1, "op_margin_change"] == 0.05                 # 0.25 − 0.20
    assert h.loc[2, "debt_ratio_change"] == -0.03               # 0.25 − 0.28
    gap = db.company_history(conn, "688002")
    assert pd.isna(gap.loc[1, "op_margin_change"])              # 2023 → 2025 中间缺一年，不算同比


def test_data_quality_report():
    conn, _ = toy_conn()
    q = db.data_quality(conn)
    assert q["duplicate_company_years"] == 0 and q["financials_without_company"] == 0
    assert q["missing_share"]["rev_growth"] == round(1 / 7, 3)


def test_queries_are_parameterized():
    conn, _ = toy_conn()
    # 把一段恶意 SQL 当行业名传进去：应该只是查不到结果，表还在
    assert db.screen(conn, 2025, industry="x'; DROP TABLE companies; --").empty
    assert conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 4


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\n{len(tests)} tests passed")

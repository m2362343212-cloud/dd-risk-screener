"""
第 9 步：SQL 数据库层（SQLite，Python 自带，不用装任何东西）

流程：  Python 抓数据 + 清洗  →  本文件把结果存进关系型数据库  →  用 SQL 查询 / 筛选 / 排名

用法：
    python src/db.py build                          # 建库：data/screener.db
    python src/db.py summary                        # 每年公司数和恶化比例
    python src/db.py industries --year 2025         # 各行业概况
    python src/db.py screen --year 2025 --industry 半导体 --min-rev-growth 0.1 --max-debt 0.5
    python src/db.py rank --industry 半导体 --top 10 # 行业内风险排名（窗口函数）
    python src/db.py history 688981                 # 单家公司历年财务 + 同比变化

三张表：
    companies   一家公司一行（代码、名称、行业、是否科创板）
    financials  一家公司一年一行（13 个财务指标 + 标签）
    scores      FY2025 模型打分（风险概率、百分位、三个危险信号）
所有查询都用参数（?）传值，不拼字符串，避免 SQL 注入。
"""
import argparse, pathlib, sqlite3

import pandas as pd

DB_PATH = "data/screener.db"
FEATS = ["gross_margin", "roe", "op_margin", "rev_growth", "profit_growth", "debt_ratio",
         "cash_ratio", "cfo_assets", "accruals", "recv_vs_rev", "inv_vs_rev",
         "exp_ratio_chg", "log_assets"]

SCHEMA = f"""
DROP TABLE IF EXISTS scores;
DROP TABLE IF EXISTS financials;
DROP TABLE IF EXISTS companies;

CREATE TABLE companies (
    code      TEXT PRIMARY KEY,          -- 股票代码
    name      TEXT,
    industry  TEXT NOT NULL,             -- 最新一年的行业；缺失记为 'Unknown'
    is_star   INTEGER NOT NULL           -- 1 = 科创板
);

CREATE TABLE financials (
    code         TEXT    NOT NULL REFERENCES companies(code),
    fiscal_year  INTEGER NOT NULL,
    {" REAL, ".join(FEATS)} REAL,
    label        INTEGER,                -- 1 = 下一年恶化；最新一年还不知道，为 NULL
    PRIMARY KEY (code, fiscal_year)      -- 一家公司一年只能有一行
);

CREATE TABLE scores (
    code         TEXT PRIMARY KEY REFERENCES companies(code),
    fiscal_year  INTEGER NOT NULL,
    risk_score   REAL NOT NULL,          -- 模型给出的恶化概率
    pct_all      REAL NOT NULL,          -- 在全部公司里的风险百分位（越高越危险）
    flag_1 TEXT, flag_2 TEXT, flag_3 TEXT
);

CREATE INDEX idx_companies_industry ON companies(industry);
CREATE INDEX idx_financials_year    ON financials(fiscal_year);
"""


# ------------------------------------------------------------------------------------------
# 1. 建库
# ------------------------------------------------------------------------------------------
def build(conn, table, scored=None):
    """
    把清洗好的公司-年份表（和 FY2025 打分表）写进数据库。返回每张表的行数。
    table  = model_table.parquet 的内容；scored = scored_fy2025.parquet 的内容（可以没有）
    """
    conn.executescript(SCHEMA)
    table = table.drop_duplicates(["code", "fiscal_year"], keep="last")     # 主键要求一家公司一年只有一行
    if scored is not None:
        scored = scored.drop_duplicates("code", keep="last")

    # 公司表：每家公司取最新一年的名称和行业
    latest = table.sort_values("fiscal_year").groupby("code", as_index=False).last()
    companies = latest[["code", "name", "industry", "is_star"]].copy()
    companies["industry"] = companies["industry"].fillna("Unknown")
    companies["is_star"] = companies["is_star"].astype(int)
    companies.to_sql("companies", conn, if_exists="append", index=False)

    table[["code", "fiscal_year"] + FEATS + ["label"]].to_sql("financials", conn, if_exists="append", index=False)

    if scored is not None:
        cols = ["code", "fiscal_year", "risk_score", "pct_all", "flag_1", "flag_2", "flag_3"]
        scored[cols].to_sql("scores", conn, if_exists="append", index=False)
    conn.commit()
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("companies", "financials", "scores")}


# ------------------------------------------------------------------------------------------
# 2. 查询（每个函数对应一个实际会问的问题）
# ------------------------------------------------------------------------------------------
def yearly_summary(conn):
    """每年有多少家公司？其中多少比例第二年恶化了？（模型的基准比例就是从这里来的）"""
    return pd.read_sql_query("""
        SELECT fiscal_year,
               COUNT(*)                       AS companies,
               ROUND(AVG(label), 3)           AS deterioration_rate,
               ROUND(AVG(rev_growth), 3)      AS avg_rev_growth
        FROM financials
        GROUP BY fiscal_year
        ORDER BY fiscal_year
    """, conn)


def industry_summary(conn, year, min_companies=10):
    """某一年各行业的概况：公司数、平均营收增速、平均负债率、恶化比例。公司太少的行业不看。"""
    return pd.read_sql_query("""
        SELECT c.industry,
               COUNT(*)                    AS companies,
               ROUND(AVG(f.rev_growth), 3) AS avg_rev_growth,
               ROUND(AVG(f.op_margin), 3)  AS avg_op_margin,
               ROUND(AVG(f.debt_ratio), 3) AS avg_debt_ratio,
               ROUND(AVG(f.label), 3)      AS deterioration_rate
        FROM financials f
        JOIN companies c ON c.code = f.code
        WHERE f.fiscal_year = ?
        GROUP BY c.industry
        HAVING COUNT(*) >= ?
        ORDER BY companies DESC
    """, conn, params=(year, min_companies))


def screen(conn, year, industry=None, min_rev_growth=None, min_op_margin=None,
           max_debt_ratio=None, limit=20):
    """
    按行业和财务条件筛选某一年的公司，营收增速高的排前面。
    条件是可选的：写成 (? IS NULL OR 列 >= ?)，参数是 NULL 时这个条件自动不生效。
    """
    return pd.read_sql_query("""
        SELECT c.code, c.name, c.industry,
               f.rev_growth, f.op_margin, f.roe, f.debt_ratio, f.cfo_assets,
               s.risk_score
        FROM financials f
        JOIN companies c      ON c.code = f.code
        LEFT JOIN scores s    ON s.code = f.code AND s.fiscal_year = f.fiscal_year
        WHERE f.fiscal_year = ?
          AND (? IS NULL OR c.industry   =  ?)
          AND (? IS NULL OR f.rev_growth >= ?)
          AND (? IS NULL OR f.op_margin  >= ?)
          AND (? IS NULL OR f.debt_ratio <= ?)
        ORDER BY f.rev_growth DESC
        LIMIT ?
    """, conn, params=(year, industry, industry, min_rev_growth, min_rev_growth,
                       min_op_margin, min_op_margin, max_debt_ratio, max_debt_ratio, limit))


def rank_in_industry(conn, industry=None, top=10):
    """
    每个行业里风险最低的前 top 家公司。
    用窗口函数 RANK() OVER (PARTITION BY 行业 ORDER BY 风险)：在每个行业内部单独排名。
    """
    return pd.read_sql_query("""
        WITH ranked AS (
            SELECT c.industry, c.code, c.name, s.risk_score,
                   RANK()   OVER (PARTITION BY c.industry ORDER BY s.risk_score) AS rank_in_industry,
                   COUNT(*) OVER (PARTITION BY c.industry)                       AS industry_size
            FROM scores s
            JOIN companies c ON c.code = s.code
        )
        SELECT * FROM ranked
        WHERE rank_in_industry <= ?
          AND (? IS NULL OR industry = ?)
        ORDER BY industry, rank_in_industry
    """, conn, params=(top, industry, industry))


def company_history(conn, code):
    """
    一家公司历年的财务数据，外加和上一年相比的变化。
    用窗口函数 LAG() 取上一年的值；年份不连续时（中间缺了一年）变化记为 NULL。
    """
    return pd.read_sql_query("""
        SELECT f.fiscal_year, c.name, c.industry,
               f.rev_growth, f.op_margin, f.roe, f.debt_ratio, f.cfo_assets, f.label,
               CASE WHEN LAG(f.fiscal_year) OVER w = f.fiscal_year - 1
                    THEN ROUND(f.op_margin  - LAG(f.op_margin)  OVER w, 4) END AS op_margin_change,
               CASE WHEN LAG(f.fiscal_year) OVER w = f.fiscal_year - 1
                    THEN ROUND(f.debt_ratio - LAG(f.debt_ratio) OVER w, 4) END AS debt_ratio_change
        FROM financials f
        JOIN companies c ON c.code = f.code
        WHERE f.code = ?
        WINDOW w AS (PARTITION BY f.code ORDER BY f.fiscal_year)
        ORDER BY f.fiscal_year
    """, conn, params=(code,))


def data_quality(conn):
    """数据质量检查：每个指标缺了多少、有没有重复的公司-年份、有没有找不到公司的财务记录。"""
    missing = pd.read_sql_query(
        "SELECT " + ", ".join(f"ROUND(AVG({f} IS NULL), 3) AS {f}" for f in FEATS) + " FROM financials", conn)
    duplicates = conn.execute("""
        SELECT COUNT(*) FROM (SELECT code, fiscal_year FROM financials
                              GROUP BY code, fiscal_year HAVING COUNT(*) > 1)""").fetchone()[0]
    orphans = conn.execute("""
        SELECT COUNT(*) FROM financials f
        LEFT JOIN companies c ON c.code = f.code
        WHERE c.code IS NULL""").fetchone()[0]
    return {"missing_share": missing.iloc[0].to_dict(), "duplicate_company_years": duplicates,
            "financials_without_company": orphans}


# ------------------------------------------------------------------------------------------
# 3. 命令行
# ------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SQL layer for the risk screener")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("summary")
    sub.add_parser("quality")
    p = sub.add_parser("industries"); p.add_argument("--year", type=int, default=2025)
    p = sub.add_parser("screen")
    p.add_argument("--year", type=int, default=2025); p.add_argument("--industry")
    p.add_argument("--min-rev-growth", type=float); p.add_argument("--min-op-margin", type=float)
    p.add_argument("--max-debt", type=float); p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("rank"); p.add_argument("--industry"); p.add_argument("--top", type=int, default=10)
    p = sub.add_parser("history"); p.add_argument("code")
    args = ap.parse_args()

    pd.set_option("display.width", 200, "display.max_rows", 200, "display.unicode.east_asian_width", True)
    conn = sqlite3.connect(DB_PATH)

    if args.cmd == "build":
        table = pd.read_parquet("data/processed/model_table.parquet")
        folder = "app_data" if pathlib.Path("app_data/scored_fy2025.parquet").exists() else "data/processed"
        scored_file = pathlib.Path(folder) / "scored_fy2025.parquet"
        scored = pd.read_parquet(scored_file) if scored_file.exists() else None
        counts = build(conn, table, scored)
        print(f"已建库 {DB_PATH}：" + "，".join(f"{t} {n:,} 行" for t, n in counts.items()))
        print("数据质量：", data_quality(conn))
    elif args.cmd == "summary":
        print(yearly_summary(conn).to_string(index=False))
    elif args.cmd == "quality":
        print(data_quality(conn))
    elif args.cmd == "industries":
        print(industry_summary(conn, args.year).to_string(index=False))
    elif args.cmd == "screen":
        print(screen(conn, args.year, args.industry, args.min_rev_growth, args.min_op_margin,
                     args.max_debt, args.limit).to_string(index=False))
    elif args.cmd == "rank":
        print(rank_in_industry(conn, args.industry, args.top).to_string(index=False))
    elif args.cmd == "history":
        print(company_history(conn, args.code).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()

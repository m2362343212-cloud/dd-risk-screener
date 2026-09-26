"""
第 2–3 步：合并四张表 → 清洗 → 打标签 → 构造特征 → 行业相对分数
输出 data/processed/model_table.parquet（每行 = 一家公司的一个财年）

用法：
    python src/build_dataset.py

⚠️ 下面 COLS 里的中文列名是按 AKShare 常见版本写的。
   先跑 `python src/fetch_data.py --check`，对照打印出来的列名，不一样就改这里右边的字符串。
"""
import pathlib

import numpy as np
import pandas as pd

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------------------
# 列名映射：{表名: {我们用的英文名: AKShare 里的中文列名}}
# ------------------------------------------------------------------------------------------
COLS = {
    "yjbb": {"code": "股票代码", "name": "股票简称", "industry": "所处行业",
             "roe": "净资产收益率", "gross_margin": "销售毛利率"},
    "zcfz": {"code": "股票代码", "cash": "资产-货币资金", "receivables": "资产-应收账款",
             "inventory": "资产-存货", "total_assets": "资产-总资产", "total_liab": "负债-总负债"},
    "lrb":  {"code": "股票代码", "net_profit": "净利润", "revenue": "营业总收入",
             "op_profit": "营业利润", "sell_exp": "营业总支出-销售费用",
             "admin_exp": "营业总支出-管理费用"},
    "xjll": {"code": "股票代码", "cfo": "经营性现金流-现金流量净额"},
}
FINANCIAL = ["银行", "证券", "保险", "多元金融", "券商", "信托"]   # 金融行业关键词，要剔除


# ------------------------------------------------------------------------------------------
# 1. 读取并合并四张表
# ------------------------------------------------------------------------------------------
def load_table(name):
    files = sorted(RAW.glob(f"{name}_*.parquet"))
    if not files:
        raise SystemExit(f"data/raw 里找不到 {name}_*.parquet，先跑第 1 步 fetch_data.py")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    mapping = COLS[name]
    missing = [v for v in mapping.values() if v not in df.columns]
    if missing:
        raise SystemExit(f"{name} 表里没有这些列：{missing}\n实际列名：{list(df.columns)}\n"
                         f"→ 修改 build_dataset.py 顶部的 COLS")
    df = df[list(mapping.values()) + ["fiscal_year"]].rename(columns={v: k for k, v in mapping.items()})
    df["code"] = df["code"].astype(str).str.zfill(6)          # 保留前导 0：1 → 000001
    return df.drop_duplicates(["code", "fiscal_year"])


def merge_tables():
    df = load_table("yjbb")
    for name in ["zcfz", "lrb", "xjll"]:
        df = df.merge(load_table(name), on=["code", "fiscal_year"], how="left")

    num = [c for c in df.columns if c not in ("code", "name", "industry", "fiscal_year")]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")          # "-"、空字符串等 → NaN

    # AKShare 的 ROE、毛利率是百分数（如 12.5 表示 12.5%），统一成小数
    df["roe"] /= 100
    df["gross_margin"] /= 100
    return df


# ------------------------------------------------------------------------------------------
# 2. 清洗
# ------------------------------------------------------------------------------------------
def clean(df):
    n0 = len(df)
    df = df[~df["industry"].fillna("").str.contains("|".join(FINANCIAL))]   # 剔除金融公司
    df = df[df["code"].str[0].isin(["0", "3", "6"])]           # 只留沪深 A 股（去掉北交所 8/4 开头）
    df = df[(df["total_assets"] > 0) & (df["revenue"] > 0)]    # 没有资产或营收的行没法算比率
    df["is_star"] = df["code"].str.startswith("688")            # 科创板标记
    print(f"清洗：{n0:,} → {len(df):,} 行")
    return df.sort_values(["code", "fiscal_year"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------
# 3. 标签：下一年是否"恶化"
# ------------------------------------------------------------------------------------------
REV_DROP = -0.20        # 营收下降超过 20% 算恶化 —— 写进 README，看过结果后不要改

def add_label(df):
    g = df.groupby("code")
    np_next  = g["net_profit"].shift(-1)                       # 下一年净利润
    rev_next = g["revenue"].shift(-1)                          # 下一年营收
    yr_next  = g["fiscal_year"].shift(-1)

    ok = yr_next == df["fiscal_year"] + 1                      # 下一年必须紧挨着（不能断档）
    turn_to_loss = (df["net_profit"] > 0) & (np_next <= 0)     # 由盈转亏
    rev_drop = (rev_next / df["revenue"] - 1) < REV_DROP       # 营收大跌
    df["label"] = np.where(ok, (turn_to_loss | rev_drop).astype(float), np.nan)
    # label 为 NaN 的行：FY2025（还没有下一年）或下一年缺失。训练时丢掉，FY2025 留着做"实时打分"
    return df


# ------------------------------------------------------------------------------------------
# 4. 特征（只用第 t 年和 t-1 年的年报，绝不用 t+1）
# ------------------------------------------------------------------------------------------
def safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)          # 除以 0 → NaN


def add_features(df):
    g = df.groupby("code")
    prev = lambda col: g[col].shift(1).where(g["fiscal_year"].shift(1) == df["fiscal_year"] - 1)

    rev_p, np_p = prev("revenue"), prev("net_profit")
    recv_p, inv_p = prev("receivables"), prev("inventory")
    exp_ratio = safe_div(df["sell_exp"].fillna(0) + df["admin_exp"].fillna(0), df["revenue"])
    exp_ratio_p = exp_ratio.groupby(df["code"]).shift(1).where(g["fiscal_year"].shift(1) == df["fiscal_year"] - 1)

    rev_growth = safe_div(df["revenue"], rev_p) - 1
    df["gross_margin"]  = df["gross_margin"]                                   # 盈利：毛利率
    df["roe"]           = df["roe"]                                            # 盈利：ROE
    df["op_margin"]     = safe_div(df["op_profit"], df["revenue"])             # 盈利：营业利润率
    df["rev_growth"]    = rev_growth                                           # 成长：营收增速
    df["profit_growth"] = safe_div(df["net_profit"] - np_p, np_p.abs())       # 成长：利润增速
    df["debt_ratio"]    = safe_div(df["total_liab"], df["total_assets"])       # 杠杆：资产负债率
    df["cash_ratio"]    = safe_div(df["cash"], df["total_assets"])             # 流动性：现金/总资产
    df["cfo_assets"]    = safe_div(df["cfo"], df["total_assets"])              # 经营现金流/总资产
    df["accruals"]      = safe_div(df["net_profit"] - df["cfo"], df["total_assets"])  # 危险：利润没有现金支撑
    df["recv_vs_rev"]   = (safe_div(df["receivables"], recv_p) - 1) - rev_growth      # 危险：应收比营收涨得快
    df["inv_vs_rev"]    = (safe_div(df["inventory"], inv_p) - 1) - rev_growth         # 危险：存货比营收涨得快
    df["exp_ratio_chg"] = exp_ratio - exp_ratio_p                              # 危险：费用率上升
    df["log_assets"]    = np.log(df["total_assets"])                           # 规模
    return df


FEATS = ["gross_margin", "roe", "op_margin", "rev_growth", "profit_growth", "debt_ratio",
         "cash_ratio", "cfo_assets", "accruals", "recv_vs_rev", "inv_vs_rev",
         "exp_ratio_chg", "log_assets"]


def winsorize_by_year(df):
    """每年分别在 1% 和 99% 分位数截尾。按年做，就不会用到未来年份的信息。"""
    for f in FEATS:
        lo = df.groupby("fiscal_year")[f].transform(lambda s: s.quantile(.01))
        hi = df.groupby("fiscal_year")[f].transform(lambda s: s.quantile(.99))
        df[f] = df[f].clip(lo, hi)
    return df


def ind_z(s):
    """行业内稳健标准化：(x − 行业中位数) / 行业四分位距。同行少于 10 家就不算。"""
    if s.notna().sum() < 10:
        return s * np.nan
    return (s - s.median()) / (s.quantile(.75) - s.quantile(.25) + 1e-9)


def add_industry_relative(df):
    for f in FEATS:
        if f == "log_assets":
            continue                                           # 规模本身不需要行业版本
        df[f + "_ind"] = df.groupby(["fiscal_year", "industry"])[f].transform(ind_z)
    return df


# ------------------------------------------------------------------------------------------
def main():
    df = merge_tables()
    df = clean(df)
    df = add_label(df)
    df = add_features(df)
    df = winsorize_by_year(df)
    df = add_industry_relative(df)

    keep = (["code", "name", "industry", "fiscal_year", "is_star", "label"]
            + FEATS + [c for c in df.columns if c.endswith("_ind")])
    df = df[keep]
    df.to_parquet(OUT / "model_table.parquet", index=False)

    # ---- 检查输出，这些数字要看一眼 ----
    print(f"\n保存 {OUT/'model_table.parquet'}：{len(df):,} 行，{df['code'].nunique():,} 家公司")
    print("\n每年公司数 和 正样本（恶化）比例：")
    print(df.groupby("fiscal_year").agg(公司数=("code", "size"), 恶化比例=("label", "mean")).round(3))
    print("\n每个特征的缺失比例：")
    print(df[FEATS].isna().mean().round(3).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()

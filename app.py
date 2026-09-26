"""
第 7 步（下）：Streamlit 尽调筛查页面
用法：  streamlit run app.py     （先跑 python src/explain.py 生成打分文件）
页面用英文，因为是给招聘方 / 面试官看的；注释用中文。
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DD Risk Screener", layout="wide")

# 特征 → 通俗英文说明（风险方向），以及实地调研要问的问题
PLAIN = {
    "gross_margin":  ("Gross margin is weak", "Is pricing power eroding? Ask about price cuts and input costs."),
    "roe":           ("Return on equity is low", "How is management planning to lift returns on capital?"),
    "op_margin":     ("Core business margin is thin", "Which product lines are losing money?"),
    "rev_growth":    ("Revenue momentum is weak", "What is the order backlog and pipeline for next year?"),
    "profit_growth": ("Profit is falling", "Is the profit decline one-off or structural?"),
    "debt_ratio":    ("Leverage is high", "What debt matures in the next 12 months, and how will it be refinanced?"),
    "cash_ratio":    ("Cash buffer is thin", "How many months of costs can current cash cover?"),
    "cfo_assets":    ("Operating cash flow is weak", "Why is cash from operations lagging reported profit?"),
    "accruals":      ("Profit is poorly backed by cash", "Walk through the gap between net profit and operating cash flow."),
    "recv_vs_rev":   ("Receivables are growing faster than sales", "Ask about customer payment terms and overdue receivables."),
    "inv_vs_rev":    ("Inventory is building up faster than sales", "Is inventory obsolete? Ask about stock ageing and write-downs."),
    "exp_ratio_chg": ("Costs are rising faster than sales", "What is driving the rise in selling and admin expenses?"),
    "log_assets":    ("Company is small", "How dependent is the business on a few customers or suppliers?"),
}
LABELS = {"gross_margin": "Gross margin", "roe": "ROE", "op_margin": "Operating margin",
          "rev_growth": "Revenue growth", "profit_growth": "Profit growth", "debt_ratio": "Debt ratio",
          "cash_ratio": "Cash / assets", "cfo_assets": "Op. cash flow / assets",
          "accruals": "Accruals / assets", "recv_vs_rev": "Receivables vs revenue growth",
          "inv_vs_rev": "Inventory vs revenue growth", "exp_ratio_chg": "Change in expense ratio"}


@st.cache_data
def load():
    live = pd.read_parquet("data/processed/scored_fy2025.parquet")
    peers = pd.read_parquet("data/processed/peer_medians.parquet").set_index("industry")
    return live, peers


live, peers = load()
base = lambda f: f.removesuffix("_ind")                         # accruals_ind → accruals

st.title("Due-Diligence Risk Screener")
st.caption("Probability that a listed Chinese company's financials deteriorate next year "
           "(profit turns to loss, or revenue falls >20%). Based on FY2025 annual reports.")

# ---- 侧边栏：科创板筛选 ----
star_only = st.sidebar.checkbox("STAR Market only (688xxx)")
pool = live[live["is_star"]] if star_only else live
pct_col = "pct_star" if star_only else "pct_all"
universe = "STAR Market companies" if star_only else "A-share companies"

st.sidebar.markdown("### Riskiest 20")
st.sidebar.dataframe(pool.nlargest(20, "risk_score")[["code", "name"]], hide_index=True)

# ---- 搜索 ----
q = st.text_input("Search by stock code or name", placeholder="e.g. 688981 or 中芯")
if not q:
    st.info("Type a stock code or company name to see its risk profile.")
    st.stop()
hits = pool[pool["code"].str.contains(q) | pool["name"].fillna("").str.contains(q)]
if hits.empty:
    st.warning("No match.")
    st.stop()
row = hits.iloc[0]
if len(hits) > 1:                                               # 多个匹配就让用户选
    code = st.selectbox("Several matches:", hits["code"].tolist(),
                        format_func=lambda c: f"{c}  {hits.loc[hits['code'] == c, 'name'].iloc[0]}")
    row = hits[hits["code"] == code].iloc[0]

# ---- 风险分数（百分位）----
st.header(f"{row['name']} ({row['code']}) — {row['industry']}")
pct = row[pct_col] * 100
c1, c2 = st.columns(2)
c1.metric("Risk percentile", f"{pct:.0f}th")
c1.write(f"Riskier than **{pct:.0f}%** of {universe}.")
c2.metric("Model probability", f"{row['risk_score']:.1%}")

# ---- 三个危险信号 ----
st.subheader("Top 3 red flags")
questions = []
for j in (1, 2, 3):
    f = base(row[f"flag_{j}"])
    text, question = PLAIN.get(f, (f, None))
    rel = " compared with industry peers" if row[f"flag_{j}"].endswith("_ind") else ""
    st.markdown(f"**{j}. {text}{rel}.**")
    if question and question not in questions:
        questions.append(question)

# ---- 同行对比表 ----
st.subheader("Peer comparison")
if row["industry"] in peers.index:
    tbl = pd.DataFrame({"This company": [row[f] for f in LABELS],
                        "Industry median": [peers.loc[row["industry"], f] for f in LABELS]},
                       index=list(LABELS.values()))
    st.dataframe(tbl.style.format("{:.2f}"))

# ---- 实地调研问题 ----
st.subheader("Questions for the site visit")
for qn in questions:
    st.markdown(f"- {qn}")

st.caption("First-pass screening tool for a personal project. Not investment advice.")

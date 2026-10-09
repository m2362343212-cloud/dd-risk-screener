"""
Streamlit 尽调筛查页面
用法：  streamlit run app.py     （先跑 python src/explain.py 生成打分文件）

两个标签页：
  1. Screen & compare —— 按行业和条件筛选 → 候选名单 → 选几家并排对比 → 每家 3–5 句总结 → 下载
  2. Company profile  —— 查单家公司：风险百分位、三个危险信号、同行对比、实地调研问题
页面用英文，因为是给招聘方 / 面试官看的；注释用中文。
"""
import pathlib, sqlite3, sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from screening import (METRICS, add_industry_rank, comparison_table, company_summary,
                       results_table, screen)
from db import company_history, rank_in_industry


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
LABELS = {f: label for f, (label, _) in METRICS.items()}


@st.cache_data
def load():
    folder = "app_data" if pathlib.Path("app_data/scored_fy2025.parquet").exists() else "data/processed"
    live = pd.read_parquet(f"{folder}/scored_fy2025.parquet")
    peers = pd.read_parquet(f"{folder}/peer_medians.parquet").set_index("industry")
    return add_industry_rank(live), peers

@st.cache_resource
def get_conn():
    """一个共用的只读连接。mode=ro：网页上的任何操作都不能改动数据库。"""
    folder = "app_data" if pathlib.Path("app_data/screener.db").exists() else "data"
    return sqlite3.connect(f"file:{folder}/screener.db?mode=ro", uri=True, check_same_thread=False)


live, peers = load()
base = lambda f: f.removesuffix("_ind")                         # accruals_ind → accruals

st.title("Due-Diligence Risk Screener")
st.caption("Probability that a listed Chinese company's financials deteriorate next year "
           "(profit turns to loss, or revenue falls >20%). Based on FY2025 annual reports.")

tab_screen, tab_profile, tab_rank = st.tabs(["Screen & compare", "Company profile", "Industry ranking"])

# ==========================================================================================
# 标签页 1：筛选 → 候选名单 → 对比 → 总结
# ==========================================================================================
with tab_screen:
    st.subheader("1. Set the screen")
    c1, c2, c3 = st.columns(3)
    industries = c1.multiselect("Industry (leave empty for all)", sorted(live["industry"].dropna().unique()))
    objective = c2.radio("Looking for", ["Candidates (lowest risk)", "Watch list (highest risk)"])
    top_n = c3.slider("Companies to show", 5, 50, 20, step=5)
    star_only = c3.checkbox("STAR Market only (688xxx)")

    with st.expander("Financial filters (optional)"):
        st.caption("A filter left at its lowest/highest setting is off. "
                   "Once a filter is on, companies missing that figure are excluded.")
        f1, f2, f3, f4 = st.columns(4)
        max_pct = f1.slider("Max risk percentile", 5, 100, 100, step=5)
        min_growth = f2.slider("Min revenue growth (%)", -50, 50, -50, step=5)
        min_roe = f3.slider("Min ROE (%)", -20, 30, -20, step=1)
        max_debt = f4.slider("Max debt ratio (%)", 10, 100, 100, step=5)
        pos_cfo = st.checkbox("Positive operating cash flow only")

    hits = screen(
        live, industries=industries, star_only=star_only,
        objective="candidates" if objective.startswith("Candidates") else "watchlist",
        max_risk_pct=None if max_pct == 100 else max_pct / 100,
        min_rev_growth=None if min_growth == -50 else min_growth / 100,
        min_roe=None if min_roe == -20 else min_roe / 100,
        max_debt_ratio=None if max_debt == 100 else max_debt / 100,
        positive_cfo=pos_cfo, top_n=top_n)

    st.subheader(f"2. Shortlist ({len(hits)} companies)")
    if hits.empty:
        st.warning("No company passes these filters. Loosen one of them.")
    else:
        st.dataframe(results_table(hits), hide_index=True, use_container_width=True)

        # ---- 选 2–4 家并排对比 ----
        st.subheader("3. Compare")
        label = lambda c: f"{c}  {hits.loc[hits['code'] == c, 'name'].iloc[0]}"
        picked = st.multiselect("Pick 2 to 4 companies from the shortlist", hits["code"].tolist(),
                                default=hits["code"].tolist()[:3], format_func=label, max_selections=4)
        rows = hits[hits["code"].isin(picked)]
        if len(rows) >= 2:
            st.table(comparison_table(rows, peers))
            st.caption("'Best' accounts for direction: lower is better for debt ratio, accruals, "
                       "receivables, inventory and expense-ratio change.")
        elif len(rows) == 1:
            st.info("Pick at least one more company to compare.")

        # ---- 每家公司的文字总结 ----
        if len(rows) >= 1:
            st.subheader("4. Plain-English summary")
            summaries = {}
            for _, r in rows.iterrows():
                sentences = company_summary(r, peers)
                summaries[r["code"]] = " ".join(sentences)
                st.markdown(f"**{r['name']} ({r['code']})**")
                st.write(" ".join(sentences))
            st.caption("Summaries are generated from fixed templates, so every sentence traces back "
                       "to a number in the tables above.")

            # ---- 下载候选名单（带总结），方便发给同事 ----
            export = results_table(hits)
            export["Summary"] = export["Code"].map(summaries).fillna("")
            st.download_button("Download shortlist (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                               file_name="shortlist.csv", mime="text/csv")

# ==========================================================================================
# 标签页 2：单家公司
# ==========================================================================================
with tab_profile:
    q = st.text_input("Search by stock code or name", placeholder="e.g. 688981 or 中芯")
    hits1 = live[live["code"].str.contains(q) | live["name"].fillna("").str.contains(q)] if q else live.iloc[0:0]
    if not q:
        st.info("Type a stock code or company name to see its risk profile.")
    elif hits1.empty:
        st.warning("No match.")
    else:
        row = hits1.iloc[0]
        if len(hits1) > 1:                                          # 多个匹配就让用户选
            code = st.selectbox("Several matches:", hits1["code"].tolist(),
                                format_func=lambda c: f"{c}  {hits1.loc[hits1['code'] == c, 'name'].iloc[0]}")
            row = hits1[hits1["code"] == code].iloc[0]

        # ---- 风险分数（百分位）----
        st.header(f"{row['name']} ({row['code']}) — {row['industry']}")
        pct = row["pct_all"] * 100
        top_share = max(1, int(np.ceil(100 - pct)))                 # 属于风险最高的前百分之几（至少 1%）
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk percentile", f"{min(pct, 99.9):.1f}")
        c1.write(f"Among the riskiest **{top_share}%** of A-share companies.")
        c2.metric("Model probability", f"{row['risk_score']:.1%}")
        c3.metric("Rank in industry (1 = lowest risk)", f"{int(row['ind_rank'])} of {int(row['ind_n'])}")

        st.subheader("Summary")
        st.write(" ".join(company_summary(row, peers)))

        # ---- 三个危险信号 ----
        st.subheader("Top 3 red flags")
        questions = []
        for j in (1, 2, 3):
            f = base(row[f"flag_{j}"])
            text, question = PLAIN.get(f, (f, None))
            if pd.isna(row.get(f)):                                 # 这个指标年报里缺失（ROE 缺失常见于净资产为负）
                text = ("Return on equity is unavailable, often a sign of negative equity" if f == "roe"
                        else f"{LABELS.get(f, f)} is missing from the report")
            rel = " compared with industry peers" if row[f"flag_{j}"].endswith("_ind") else ""
            st.markdown(f"**{j}. {text}{rel}.**")
            if question and question not in questions:
                questions.append(question)

        # ---- 同行对比表（百分比显示，缺失显示为 —）----
        st.subheader("Peer comparison")
        if row["industry"] in peers.index:
            fmt = lambda v: "—" if pd.isna(v) else f"{v:.1%}"
            tbl = pd.DataFrame({"This company": [fmt(row[f]) for f in LABELS],
                                "Industry median": [fmt(peers.loc[row["industry"], f]) for f in LABELS]},
                               index=list(LABELS.values()))
            st.table(tbl)

        st.subheader("Financial history")
        hist = company_history(get_conn(), row["code"])
        if hist.empty:
            st.info("No history in the database for this company.")
        else:
            cols = {"fiscal_year": "Year", "rev_growth": "Revenue growth", "op_margin": "Operating margin",
                    "op_margin_change": "Margin change vs prior year", "debt_ratio": "Debt ratio",
                    "debt_ratio_change": "Debt ratio change"}
            show = hist[list(cols)].rename(columns=cols)
            st.dataframe(show.style.format({c: "{:.1%}" for c in list(cols.values())[1:]}, na_rep="—"),
                         hide_index=True, use_container_width=True)
            st.line_chart(hist.set_index("fiscal_year")[["op_margin", "debt_ratio"]])
            st.caption("Queried from SQLite. A change is blank when the previous year is missing.")

        # ---- 实地调研问题 ----
        st.subheader("Questions for the site visit")
        for qn in questions:
            st.markdown(f"- {qn}")
# ==========================================================================================
# 标签页 3：行业内排名（SQL 窗口函数 RANK）
# ==========================================================================================
with tab_rank:
    c1, c2 = st.columns(2)
    industry = c1.selectbox("Industry", sorted(live["industry"].dropna().unique()))
    top = c2.slider("Lowest-risk companies to show", 5, 30, 10, step=5)
    ranked = rank_in_industry(get_conn(), industry, top)
    if ranked.empty:
        st.info("No scored companies in this industry.")
    else:
        st.dataframe(ranked.rename(columns={"code": "Code", "name": "Name", "risk_score": "Model probability",
                                            "rank_in_industry": "Rank", "industry_size": "Companies in industry"})
                           .drop(columns="industry")
                           .style.format({"Model probability": "{:.1%}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Ranked inside the database with RANK() OVER (PARTITION BY industry ORDER BY risk).")

st.caption("First-pass screening tool for a personal project. Not investment advice.")

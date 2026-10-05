"""
第 8 步：筛选 / 对比 / 文字总结 的纯逻辑（不依赖 Streamlit，所以可以单独测试）

app.py 只负责画界面；"哪些公司入选、表格里放什么、总结怎么写" 都在这里。
总结是模板生成的，不是大模型写的：每一句话都能追溯到表里的一个数字。
"""
import numpy as np
import pandas as pd

# 指标 → (英文名, 方向)。方向 +1 表示越高越好，-1 表示越低越好
METRICS = {
    "rev_growth":    ("Revenue growth", +1),
    "profit_growth": ("Profit growth", +1),
    "gross_margin":  ("Gross margin", +1),
    "op_margin":     ("Operating margin", +1),
    "roe":           ("ROE", +1),
    "debt_ratio":    ("Debt ratio", -1),
    "cash_ratio":    ("Cash / assets", +1),
    "cfo_assets":    ("Op. cash flow / assets", +1),
    "accruals":      ("Accruals / assets", -1),
    "recv_vs_rev":   ("Receivables vs revenue growth", -1),
    "inv_vs_rev":    ("Inventory vs revenue growth", -1),
    "exp_ratio_chg": ("Change in expense ratio", -1),
}
# 结果表里展示的几个关键指标（太多列反而看不清）
KEY = ["rev_growth", "roe", "op_margin", "debt_ratio", "cfo_assets"]

# 危险信号 → 通俗英文（和单家公司页面用同一套说法）
CONCERN = {
    "gross_margin": "weak gross margin", "roe": "low return on equity",
    "op_margin": "a thin operating margin", "rev_growth": "weak revenue momentum",
    "profit_growth": "falling profit", "debt_ratio": "high leverage",
    "cash_ratio": "a thin cash buffer", "cfo_assets": "weak operating cash flow",
    "accruals": "profit that is poorly backed by cash",
    "recv_vs_rev": "receivables growing faster than sales",
    "inv_vs_rev": "inventory building up faster than sales",
    "exp_ratio_chg": "costs rising faster than sales", "log_assets": "small size",
}
# 优点 → 通俗英文（负债率低要说成 "low leverage"，不能说 "debt ratio 很突出"）
STRENGTH = {
    "gross_margin": "gross margin", "roe": "return on equity", "op_margin": "operating margin",
    "rev_growth": "revenue growth", "profit_growth": "profit growth", "debt_ratio": "low leverage",
    "cash_ratio": "its cash buffer", "cfo_assets": "operating cash flow",
    "accruals": "cash-backed earnings", "recv_vs_rev": "receivables discipline",
    "inv_vs_rev": "inventory discipline", "exp_ratio_chg": "cost control",
}
# 中性说法：只是点名这个指标，不带好坏判断（用在"模型在意、但数字并不差"的那句话里）
NEUTRAL = {
    "gross_margin": "gross margin", "roe": "return on equity", "op_margin": "operating margin",
    "rev_growth": "revenue growth", "profit_growth": "profit growth", "debt_ratio": "leverage",
    "cash_ratio": "cash buffer", "cfo_assets": "operating cash flow", "accruals": "accruals",
    "recv_vs_rev": "receivables growth", "inv_vs_rev": "inventory growth",
    "exp_ratio_chg": "expense ratio", "log_assets": "company size",
}


def pct(v):
    """0.123 → '12.3%'；缺失 → '—'"""
    return "—" if pd.isna(v) else f"{v:.1%}"


def base(feature):
    """accruals_ind → accruals（行业相对特征还原成原始指标名）"""
    return feature.removesuffix("_ind")


def add_industry_rank(df):
    """每家公司在自己行业里的风险排名（1 = 行业内风险最低）和行业公司数。"""
    df = df.copy()
    df["industry"] = df["industry"].fillna("Unknown")      # 有些公司年报里没有行业：归到 "Unknown"，否则排名会是空值
    g = df.groupby("industry")["risk_score"]
    df["ind_rank"] = g.rank(method="min").fillna(0).astype(int)
    df["ind_n"] = g.transform("size").fillna(0).astype(int)
    return df


# ------------------------------------------------------------------------------------------
# 1. 筛选
# ------------------------------------------------------------------------------------------
def screen(df, industries=None, star_only=False, objective="candidates", max_risk_pct=None,
           min_rev_growth=None, min_roe=None, max_debt_ratio=None, positive_cfo=False,
           top_n=20):
    """
    按条件筛选，返回排好序的前 top_n 家公司。

    objective = "candidates"：找风险最低的（值得进一步看的候选）
              = "watchlist" ：找风险最高的（需要警惕的名单）
    某个条件一旦启用，这个指标缺失的公司会被排除：缺数据就无法确认它满足条件。
    """
    out = df
    if industries:
        out = out[out["industry"].isin(industries)]
    if star_only:
        out = out[out["is_star"]]
    if max_risk_pct is not None:
        out = out[out["pct_all"] <= max_risk_pct]
    if min_rev_growth is not None:
        out = out[out["rev_growth"] >= min_rev_growth]
    if min_roe is not None:
        out = out[out["roe"] >= min_roe]
    if max_debt_ratio is not None:
        out = out[out["debt_ratio"] <= max_debt_ratio]
    if positive_cfo:
        out = out[out["cfo_assets"] > 0]
    out = out.sort_values("risk_score", ascending=(objective == "candidates"))
    return out.head(top_n)


def top_concern(row):
    """模型认为最推高这家公司风险的那个特征；如果没有任何特征在推高风险，返回 None。"""
    if row.get("flag_1_shap", 1) <= 0:
        return None
    return CONCERN.get(base(row["flag_1"]), base(row["flag_1"]))


def results_table(hits):
    """筛选结果 → 给人看的表（数字已格式化）。"""
    t = pd.DataFrame({
        "Code": hits["code"], "Name": hits["name"], "Industry": hits["industry"],
        "Risk percentile": (hits["pct_all"] * 100).round(1),
        "Model probability": hits["risk_score"].map(pct),
    })
    for f in KEY:
        t[METRICS[f][0]] = hits[f].map(pct)
    t["Top risk driver"] = [top_concern(r) or "none material" for _, r in hits.iterrows()]
    return t.reset_index(drop=True)


# ------------------------------------------------------------------------------------------
# 2. 对比
# ------------------------------------------------------------------------------------------
def comparison_table(rows, peers):
    """
    几家公司并排对比。每行一个指标，最后一列指出哪家最好（考虑了方向：负债率是越低越好）。
    如果入选公司都在同一个行业，再加一列行业中位数。
    """
    names = [f"{r['name']} ({r['code']})" for _, r in rows.iterrows()]
    same_industry = rows["industry"].nunique() == 1 and rows["industry"].iloc[0] in peers.index
    data = {}
    risk = rows["pct_all"].to_numpy()
    line = [f"{v * 100:.1f}" for v in risk]
    if same_industry:
        line.append("—")
    data["Risk percentile (lower = safer)"] = line + [names[int(np.argmin(risk))]]

    for f, (label, direction) in METRICS.items():
        vals = rows[f].to_numpy(dtype=float)
        line = [pct(v) for v in vals]
        if same_industry:
            line.append(pct(peers.loc[rows["industry"].iloc[0], f]))
        if np.all(np.isnan(vals)):
            best = "—"
        else:
            best = names[int(np.nanargmax(vals * direction))]
        data[label] = line + [best]

    cols = names + (["Industry median"] if same_industry else []) + ["Best"]
    return pd.DataFrame.from_dict(data, orient="index", columns=cols)


# ------------------------------------------------------------------------------------------
# 3. 文字总结（3–5 句，模板生成）
# ------------------------------------------------------------------------------------------
def _strengths(row, peers, k=2, min_z=0.5):
    """相对同行最突出的 k 个优点：用行业内标准化分数 × 方向，至少要好出 0.5 个四分位距。"""
    scored = []
    for f, (label, direction) in METRICS.items():
        z = row.get(f + "_ind")
        if pd.isna(z) or pd.isna(row[f]) or row["industry"] not in peers.index:
            continue
        if z * direction >= min_z:
            scored.append((z * direction, STRENGTH[f], row[f], peers.loc[row["industry"], f]))
    scored.sort(reverse=True)
    return scored[:k]


def _concerns(row, peers, k=2):
    """
    模型给出的危险信号，分成两类返回：
      worse —— 模型在意、而且数字确实比行业中位数差的（最多 k 个），可以直接写进总结
      other —— 模型在意、但数字并不比同行差的（只报名字，避免写出"现金很薄（20% 对 10%）"这种自相矛盾的话）
    只看 SHAP > 0 的特征，也就是确实在推高风险的。
    """
    worse, other, seen = [], [], set()
    for j in (1, 2, 3):
        if row.get(f"flag_{j}_shap", 1) <= 0:
            continue
        f = base(row[f"flag_{j}"])
        if f in seen:                                     # 原始版和行业版指向同一个指标，只说一次
            continue
        seen.add(f)
        v = row.get(f)
        has_peer = row["industry"] in peers.index and f in METRICS
        med = peers.loc[row["industry"], f] if has_peer else np.nan
        if pd.isna(v) or pd.isna(med):
            other.append(f)
        elif (v - med) * METRICS[f][1] < 0:               # 乘上方向后小于 0 → 比同行差
            worse.append((f, v, med))
        else:
            other.append(f)
    return worse[:k], other


def company_summary(row, peers):
    """
    给一家公司写 3–5 句通俗英文总结。返回句子列表。
    结构：① 风险位置 ② 相对同行的优点 ③ 模型的主要顾虑 ④ 数据缺失（如果有）⑤ 初筛结论
    """
    s = []
    p = row["pct_all"]
    side = (f"the safest {max(1, round(p * 100))}%" if p < 0.5
            else f"the riskiest {max(1, round((1 - p) * 100))}%")
    prob = "under 1%" if row["risk_score"] < 0.01 else f"{row['risk_score']:.0%}"
    s.append(f"{row['name']} ({row['code']}, {row['industry']}) has a model-estimated "
             f"{prob} chance of financial deterioration next year, which puts it in "
             f"{side} of A-share companies and ranks it {int(row['ind_rank'])} of {int(row['ind_n'])} "
             f"in its industry (1 = lowest risk).")

    good = _strengths(row, peers)
    if good:
        parts = [f"{label} ({pct(v)} vs. {pct(m)} industry median)" for _, label, v, m in good]
        s.append("Relative to peers it stands out on " + " and ".join(parts) + ".")
    else:
        s.append("No metric stands out clearly against its industry peers.")

    worse, other = _concerns(row, peers)
    if worse:
        parts = [f"{CONCERN[f]} ({pct(v)} vs. {pct(m)} industry median)" for f, v, m in worse]
        verb = "concern is" if len(parts) == 1 else "concerns are"
        s.append(f"The model's main {verb} " + " and ".join(parts) + ".")
    elif other:
        names = " and ".join(NEUTRAL.get(f, f) for f in other[:2])
        one = len(other[:2]) == 1                         # 只有一个就用单数
        if p < 1 / 3:                                     # 低风险公司：不说"风险来自……"，直接说没有明显短板
            s.append(f"The model finds no material weakness: its largest risk {'driver' if one else 'drivers'} "
                     f"({names}) {'is' if one else 'are'} in line with or better than the industry median.")
        else:
            s.append(f"Its top risk {'driver' if one else 'drivers'} ({names}) {'is' if one else 'are'} not worse "
                     "than the industry median, so the risk comes from the overall combination rather than "
                     "one weak figure.")
    else:
        s.append("No single feature pushes its risk materially above average.")

    missing = [METRICS[f][0] for f in METRICS if pd.isna(row[f])]
    if missing:
        s.append("Not reported, so check manually: " + ", ".join(missing) + ".")

    if p < 1 / 3:
        s.append("On a first pass it screens as lower-risk and can move forward to detailed review.")
    elif p < 2 / 3:
        s.append("On a first pass the picture is mixed; resolve the concerns above before spending more time.")
    else:
        s.append("On a first pass it screens as high-risk; proceed only with a clear answer to the concerns above.")
    return s

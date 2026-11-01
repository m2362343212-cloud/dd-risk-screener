p = "alpha/evaluate_cn.py"
s = open(p).read()
if "studentized" in s:
    print("already patched"); raise SystemExit
start = s.index("def placebo(")
s = s[:start] + '''def placebo(p, n_iter=200, seed=0):
    """Permutation test (studentized, yearly shuffle)."""
    rng = np.random.default_rng(seed)
    q = p[["month", "code", "fiscal_year", "distress", "fwd_1m"]].copy()
    real_t = nw_mean_t(ic_series(q))[1]
    base = q[["fiscal_year", "code", "distress"]].drop_duplicates(["fiscal_year", "code"])
    sims = []
    for _ in range(n_iter):
        b = base.copy()
        b["shuf"] = b.groupby("fiscal_year")["distress"].transform(lambda s: rng.permutation(s.values))
        qq = q.merge(b[["fiscal_year", "code", "shuf"]], on=["fiscal_year", "code"], how="left")
        sims.append(nw_mean_t(ic_series(qq, col="shuf"))[1])
    sims = np.array(sims)
    return {"real_mean_IC": ic_series(q).mean(), "real_IC_t": real_t,
            "placebo_mean": sims.mean(), "placebo_std": sims.std(),
            "two_sided_p": (np.abs(sims) >= abs(real_t)).mean(), "n_iter": n_iter}
'''
open(p, "w").write(s)
r = "alpha/run.py"
t = open(r).read()
old = ('f"Real mean IC {plc[\'real_mean_IC\']:.4f} vs shuffled {plc[\'placebo_mean\']:.4f} "\n'
       '        f"+/- {plc[\'placebo_std\']:.4f}; share at least as extreme: {plc[\'two_sided_p\']:.3f} "')
new = ('f"Real IC t-stat {plc[\'real_IC_t\']:.2f} (mean IC {plc[\'real_mean_IC\']:.4f}) vs shuffled t-stats "\n'
       '        f"{plc[\'placebo_mean\']:.2f} +/- {plc[\'placebo_std\']:.2f}; share at least as extreme: {plc[\'two_sided_p\']:.3f} "')
assert t.count(old) == 1
t = t.replace(old, new)
open(r, "w").write(t)
print("patched OK")

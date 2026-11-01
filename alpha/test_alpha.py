"""Pipeline checks on fake prices (real score file). Run from repo root:
    python -m pytest -q alpha/test_alpha.py      (~1 minute)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pandas as pd
import evaluate_cn as E
from synthetic_cn import make_prices

SCORES = "alpha/output/scores_walkforward_tuned.csv"


def _run(effect):
    sc = pd.read_csv(SCORES, dtype={"code": str})
    px, be = make_prices(sc, effect=effect)
    p = E.add_score_rank(E.add_groups(E.build_panel(sc, px, be)))
    ls = E.group_returns(p)["LS_Q1_Q5"]
    return E.nw_mean_t(ls)[1], E.fama_macbeth(p, E.CONTROLS).loc["score_rank", "t_NW"]


def test_finds_planted_effect():
    _, t_fm = _run(0.004)
    assert t_fm < -3


def test_size_trap_raw_sort_fooled_but_fama_macbeth_not():
    t_ls, t_fm = _run(0.0)
    assert t_ls < -2          # raw sort: "risky stocks do better" (it's the small-cap premium)
    assert abs(t_fm) < 2      # with size control: nothing


def test_no_lookahead():
    sc = pd.read_csv(SCORES, dtype={"code": str})
    px, be = make_prices(sc, effect=0)
    p = E.build_panel(sc, px, be)
    assert (p["available"] <= p["month"]).all()
    assert p["month"].min() >= pd.Timestamp("2018-05-31")

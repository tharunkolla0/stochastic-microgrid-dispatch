"""Sanity tests: run with  python tests/test_microgrid.py   (or pytest tests)."""
import os, sys
from dataclasses import replace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.microgrid import *  # noqa


def _days(n, seed=0):
    return sample_days(WeatherSpec(), n, np.random.default_rng(seed))


def test_single_scenario_matches_wait_and_see_and_balance():
    """With S=1 the stochastic LP equals the perfect-information problem; recourse cost + DA cost must agree."""
    sy = System()
    pv, ld, _ = _days(3)
    for s in range(3):
        x, obj = solve_saa(pv[s:s + 1], ld[s:s + 1], sy, return_obj=True)
        total = second_stage_cost(x, pv[s], ld[s], sy) + float(DA_PRICE @ x)
        assert abs(total - obj) < 1e-4 * max(1, abs(obj)), (total, obj)


def test_energy_balance_and_soc_in_recourse():
    sy = System()
    pv, ld, _ = _days(1, 5)
    x = np.full(T, 250.0)
    _, v = second_stage_cost(x, pv[0], ld[0], sy, detail=True)
    # exact energy balance: x + u + (PV - w) + d + l = load + c + z
    assert np.allclose(x + v["u"] + (pv[0] - v["w"]) + v["d"] + v["l"], ld[0] + v["c"] + v["z"], atol=1e-6)
    assert v["e"].min() >= sy.soc_min * sy.batt_kwh - 1e-6 and v["e"].max() <= sy.batt_kwh + 1e-6
    assert abs(v["e"][-1] - sy.soc_init * sy.batt_kwh) < 1e-6          # cyclic state of charge


def test_no_battery_no_premium_gives_no_vss_and_known_cost():
    """With rt_premium = 1 and no battery, deviations are settled at the day-ahead price: stochastic ~ deterministic."""
    sy = System(rt_premium=1.0, batt_kwh=0.0, batt_kw=0.0, sell_price=0.0)
    pv, ld, _ = _days(300, 1)
    xs = solve_saa(pv, ld, sy); xd = solve_saa(pv.mean(0, keepdims=True), ld.mean(0, keepdims=True), sy)
    tp, tl, _ = _days(200, 9)
    ev = Evaluator(sy)
    assert abs(ev.costs(xs, tp, tl).mean() - ev.costs(xd, tp, tl).mean()) < 0.01 * ev.costs(xd, tp, tl).mean()


def test_stochastic_beats_deterministic_in_expectation():
    sy, ws = System(), WeatherSpec()
    h_pv, h_ld, _ = sample_days(ws, 365, np.random.default_rng(2))
    pv, ld = smoothed_bootstrap(h_pv, h_ld, 150, np.random.default_rng(3))
    xs = solve_saa(pv, ld, sy); xd = solve_saa(h_pv.mean(0, keepdims=True), h_ld.mean(0, keepdims=True), sy)
    tp, tl, _ = sample_days(ws, 400, np.random.default_rng(4))
    ev = Evaluator(sy)
    assert ev.costs(xs, tp, tl).mean() < ev.costs(xd, tp, tl).mean()


def test_cvar_trades_mean_for_tail_and_ws_bound():
    sy, ws = System(), WeatherSpec()
    h_pv, h_ld, _ = sample_days(ws, 365, np.random.default_rng(6))
    pv, ld = smoothed_bootstrap(h_pv, h_ld, 120, np.random.default_rng(7))
    tp, tl, _ = sample_days(ws, 400, np.random.default_rng(8))
    ev = Evaluator(sy)
    x0, x5 = solve_saa(pv, ld, sy, lam=0.0), solve_saa(pv, ld, sy, lam=0.5)
    r0, r5 = risk_summary(ev.costs(x0, tp, tl)), risk_summary(ev.costs(x5, tp, tl))
    assert r5["cvar90"] < r0["cvar90"] and r5["mean"] > r0["mean"]
    ws_ = wait_and_see(tp[:40], tl[:40], sy)
    assert (ev.costs(x0, tp[:40], tl[:40]) >= ws_ - 1e-6).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)

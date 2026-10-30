"""
Runs the full experimental pipeline; writes ALL results to ./results, ./figures, ./data.
Usage:  python -m src.run_experiments        (about 5-8 minutes on one CPU core)
All random streams are seeded -> fully reproducible.
"""
import json, os, sys, time
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.microgrid import *   # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG, DAT = (os.path.join(ROOT, d) for d in ("results", "figures", "data"))
for d in (RES, FIG, DAT):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
C = {"sto": "#1f77b4", "det": "#d62728", "ws": "#2ca02c", "grey": "#7f7f7f", "pv": "#ff9f1c"}
STATES = ["Sunny", "Partly cloudy", "Overcast"]
EF = 0.71   # kg CO2 per kWh grid import (assumed India average grid emission factor)

sy, ws = System(), WeatherSpec()
SUM = {}
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def setting(ws_, sy_, S=100, ntest=400, seed=0, nhist=365):
    """Controlled experiment: history + training scenarios + test days all from the SAME world."""
    PVh, LDh, _ = sample_days(ws_, nhist, np.random.default_rng(seed), sy_.pv_kwp)
    PV, LD = smoothed_bootstrap(PVh, LDh, S, np.random.default_rng(seed + 1), sy_.pv_kwp)
    x_s = solve_saa(PV, LD, sy_)
    x_d = solve_saa(PVh.mean(0)[None], LDh.mean(0)[None], sy_)
    Tp, Tl, _ = sample_days(ws_, ntest, np.random.default_rng(seed + 2), sy_.pv_kwp)
    ev = Evaluator(sy_)
    cs, cd = ev.costs(x_s, Tp, Tl), ev.costs(x_d, Tp, Tl)
    return dict(stoch=cs.mean(), det=cd.mean(), VSS=cd.mean() - cs.mean(), VSS_pct=100 * (cd.mean() - cs.mean()) / cd.mean(),
                std_stoch=cs.std(ddof=1), std_det=cd.std(ddof=1)), x_s


# ============================================================ 1. DATA =======
N_HIST = 730
PVh, LDh, state_h = sample_days(ws, N_HIST, np.random.default_rng(2026), sy.pv_kwp)
PVh, LDh = np.round(PVh, 1), np.round(LDh, 1)
dates = pd.date_range("2024-01-01", periods=N_HIST, freq="D")
long = pd.DataFrame({"date": np.repeat(dates, T), "hour": np.tile(HOURS, N_HIST),
                     "pv_kw": PVh.reshape(-1), "load_kw": LDh.reshape(-1)})
long.to_csv(os.path.join(DAT, "microgrid_history_hourly.csv"), index=False)
daily = pd.DataFrame({"date": dates, "weather_state": [STATES[i] for i in state_h],
                      "pv_kwh": PVh.sum(1), "load_kwh": LDh.sum(1), "peak_load_kw": LDh.max(1)})
daily.round({"pv_kwh": 1, "load_kwh": 1, "peak_load_kw": 1}).to_csv(os.path.join(DAT, "daily_summary.csv"), index=False)

rows = []
for k, nm in enumerate(STATES):
    m = state_h == k
    rows.append(dict(weather_state=nm, days=int(m.sum()), share_pct=100 * m.mean(), pv_kwh_mean=PVh[m].sum(1).mean(),
                     pv_kwh_std=PVh[m].sum(1).std(ddof=1), load_kwh_mean=LDh[m].sum(1).mean(),
                     peak_load_mean=LDh[m].max(1).mean()))
rows.append(dict(weather_state="ALL DAYS", days=N_HIST, share_pct=100.0, pv_kwh_mean=PVh.sum(1).mean(),
                 pv_kwh_std=PVh.sum(1).std(ddof=1), load_kwh_mean=LDh.sum(1).mean(), peak_load_mean=LDh.max(1).mean()))
t1 = pd.DataFrame(rows); t1.round(2).to_csv(os.path.join(RES, "table1_data_statistics.csv"), index=False)
SUM["data"] = dict(pv_load_corr=float(np.corrcoef(PVh.sum(1), LDh.sum(1))[0, 1]),
                   pv_cv=float(PVh.sum(1).std() / PVh.sum(1).mean()), load_cv=float(LDh.sum(1).std() / LDh.sum(1).mean()),
                   pv_share_of_load=float(PVh.sum(1).mean() / LDh.sum(1).mean()), stats=t1.round(2).to_dict(orient="records"))
par = pd.DataFrame({"parameter": ["PV capacity (kWp)", "Battery energy (kWh)", "Battery power (kW)", "One-way efficiency", "Min SOC",
                                  "Grid connection (kW)", "Real-time price multiplier", "Export / resale price (INR/kWh)",
                                  "Value of lost load (INR/kWh)", "Battery wear (INR/kWh)", "DA price min/max (INR/kWh)"],
                    "value": [sy.pv_kwp, sy.batt_kwh, sy.batt_kw, sy.eta, sy.soc_min, sy.grid_kw, sy.rt_premium, sy.sell_price,
                              sy.voll, sy.degr, f"{DA_PRICE.min():.1f} / {DA_PRICE.max():.1f}"]})
par.to_csv(os.path.join(RES, "table0_system_parameters.csv"), index=False)
log("data generated")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
for k, col in enumerate(["#ff9f1c", "#8ecae6", "#6c757d"]):
    ax[0].hist(PVh[state_h == k].sum(1), bins=25, alpha=.75, color=col, label=STATES[k])
ax[0].set_xlabel("daily PV energy (kWh)"); ax[0].set_ylabel("days"); ax[0].legend(fontsize=7); ax[0].set_title("Daily solar energy by weather")
for k, col in enumerate(["#ff9f1c", "#8ecae6", "#6c757d"]):
    idx = np.where(state_h == k)[0][:12]
    for j in idx:
        ax[1].plot(HOURS, PVh[j], color=col, alpha=.35, lw=.8)
ax[1].set_xlabel("hour of day"); ax[1].set_ylabel("PV output (kW)"); ax[1].set_title("Sample PV profiles")
ax[2].plot(HOURS, LDh.mean(0), color="k", lw=2, label="mean load")
ax[2].fill_between(HOURS, np.quantile(LDh, .05, 0), np.quantile(LDh, .95, 0), color=C["sto"], alpha=.25, label="5-95 % band")
ax[2].plot(HOURS, PVh.mean(0), color=C["pv"], lw=2, label="mean PV"); ax[2].set_xlabel("hour of day"); ax[2].set_ylabel("kW")
ax[2].legend(fontsize=7); ax[2].set_title("Load and PV bands")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_data_overview.png")); plt.close(fig)

# ================================================ 2. MAIN SOLVE =============
S_MAIN, N_TEST, N_WS = 300, 2000, 500
PV_tr, LD_tr = smoothed_bootstrap(PVh, LDh, S_MAIN, np.random.default_rng(11), sy.pv_kwp)
x_sto, obj_sto = solve_saa(PV_tr, LD_tr, sy, return_obj=True)
x_det = solve_saa(PVh.mean(0)[None], LDh.mean(0)[None], sy)
Tp, Tl, Tstate = sample_days(ws, N_TEST, np.random.default_rng(99), sy.pv_kwp)
ev = Evaluator(sy)
c_sto, d_sto = ev.costs(x_sto, Tp, Tl, want_detail=True)
c_det, d_det = ev.costs(x_det, Tp, Tl, want_detail=True)
log(f"main solved: E[cost] sto={c_sto.mean():.0f} det={c_det.mean():.0f}")
c_ws = wait_and_see(Tp[:N_WS], Tl[:N_WS], sy)
rs_s, rs_d = risk_summary(c_sto), risk_summary(c_det)
diff = c_det - c_sto
brng = np.random.default_rng(5)
boot = np.array([diff[brng.integers(0, N_TEST, N_TEST)].mean() for _ in range(2000)])
VSS = float(diff.mean()); VSS_ci = (float(np.quantile(boot, .025)), float(np.quantile(boot, .975)))
EVPI = float((c_sto[:N_WS] - c_ws).mean()); WS = float(c_ws.mean())
share_win = float((diff > 0).mean())
log("wait-and-see done")

sched = pd.DataFrame({"hour": HOURS, "da_price": DA_PRICE, "mean_load_kw": LDh.mean(0), "mean_pv_kw": PVh.mean(0),
                      "deterministic_purchase_kwh": x_det, "stochastic_purchase_kwh": x_sto,
                      "difference_kwh": x_sto - x_det})
sched.round(1).to_csv(os.path.join(RES, "table2_day_ahead_schedule.csv"), index=False)


def metrics(c, d, x):
    imp = x.sum() + d["rt_kwh"].mean() - d["resold"].mean()
    return dict(mean_cost=c.mean(), std=c.std(ddof=1), p50=np.median(c), p95=np.quantile(c, .95), cvar90=risk_summary(c)["cvar90"],
                da_kwh=float(x.sum()), rt_kwh=float(d["rt_kwh"].mean()), resold_kwh=float(d["resold"].mean()),
                curtailed_kwh=float(d["curtail"].mean()), shed_kwh=float(d["shed"].mean()),
                lolp_pct=100 * float((d["shed"] > 0.1).mean()), net_import_kwh=float(imp), co2_kg=float(EF * imp))


comp = pd.DataFrame([{"model": "Deterministic (mean-value)", **metrics(c_det, d_det, x_det)},
                     {"model": f"Stochastic (SAA, S={S_MAIN})", **metrics(c_sto, d_sto, x_sto)}])
comp.round(2).to_csv(os.path.join(RES, "table3_baseline_comparison.csv"), index=False)
val = pd.DataFrame([
    (f"Stochastic expected daily cost (out-of-sample, N={N_TEST})", c_sto.mean()),
    ("Deterministic expected daily cost (EEV, out-of-sample)", c_det.mean()),
    ("VSS = EEV - RP (INR/day saved)", VSS),
    ("VSS 95% bootstrap CI lower", VSS_ci[0]), ("VSS 95% bootstrap CI upper", VSS_ci[1]),
    ("VSS as % of deterministic cost", 100 * VSS / c_det.mean()),
    ("Share of days stochastic plan is cheaper (%)", 100 * share_win),
    (f"Wait-and-see expected cost (N={N_WS})", WS),
    (f"EVPI = RP - WS (INR/day, N={N_WS})", EVPI),
    ("In-sample SAA objective (S=300)", obj_sto),
    ("Annual saving from VSS (INR/yr)", 365 * VSS),
], columns=["metric", "value"])
val.round(2).to_csv(os.path.join(RES, "table4_value_of_stochastic_solution.csv"), index=False)
SUM["main"] = dict(x_sto=x_sto.tolist(), x_det=x_det.tolist(), sto=rs_s, det=rs_d, VSS=VSS, VSS_ci=VSS_ci, VSS_pct=100 * VSS / c_det.mean(),
                   EVPI=EVPI, WS=WS, share_win=share_win, obj_in=float(obj_sto), annual=float(365 * VSS),
                   met_sto=comp.iloc[1].drop("model").astype(float).to_dict(), met_det=comp.iloc[0].drop("model").astype(float).to_dict())

# cost by weather state
rows = []
for k, nm in enumerate(STATES):
    m = Tstate == k
    rows.append(dict(weather_state=nm, days=int(m.sum()), cost_det=c_det[m].mean(), cost_sto=c_sto[m].mean(),
                     saving=c_det[m].mean() - c_sto[m].mean(), shed_det=d_det["shed"][m].mean(), shed_sto=d_sto["shed"][m].mean()))
bs = pd.DataFrame(rows); bs.round(2).to_csv(os.path.join(RES, "table5_cost_by_weather.csv"), index=False)
SUM["by_weather"] = bs.round(2).to_dict(orient="records")

fig, ax = plt.subplots(figsize=(8, 3.3))
ax.step(HOURS, x_det, where="mid", color=C["det"], lw=2, label="deterministic purchase")
ax.step(HOURS, x_sto, where="mid", color=C["sto"], lw=2, label="stochastic purchase")
ax.plot(HOURS, LDh.mean(0), "k--", lw=1.2, label="mean load"); ax.plot(HOURS, LDh.mean(0) - PVh.mean(0), ":", color="k", lw=1.2, label="mean net load (load - PV)")
ax2 = ax.twinx(); ax2.step(HOURS, DA_PRICE, where="mid", color=C["grey"], alpha=.5); ax2.set_ylabel("DA price (INR/kWh)", color=C["grey"]); ax2.grid(False)
ax.set_xlabel("hour of day"); ax.set_ylabel("kWh purchased (day-ahead)"); ax.legend(fontsize=7, ncol=2, loc="upper left"); ax.set_ylim(0, 560)
ax.set_title("Day-ahead purchase schedules")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_day_ahead_schedule.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
bins = np.linspace(min(c_sto.min(), c_det.min()), max(c_sto.max(), c_det.max()), 60)
ax[0].hist(c_det, bins=bins, alpha=.55, color=C["det"], label="deterministic"); ax[0].hist(c_sto, bins=bins, alpha=.55, color=C["sto"], label="stochastic")
ax[0].axvline(c_det.mean(), color=C["det"], ls="--"); ax[0].axvline(c_sto.mean(), color=C["sto"], ls="--")
ax[0].set_xlabel("daily energy cost (INR), 2,000 simulated days"); ax[0].set_ylabel("days"); ax[0].legend(); ax[0].set_title("Daily cost distribution")
w = .35; xi = np.arange(3)
ax[1].bar(xi - w / 2, bs.cost_det, w, color=C["det"], label="deterministic"); ax[1].bar(xi + w / 2, bs.cost_sto, w, color=C["sto"], label="stochastic")
ax[1].set_xticks(xi); ax[1].set_xticklabels(STATES); ax[1].set_ylabel("mean daily cost (INR)"); ax[1].set_title("Cost by weather type"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_cost_distribution.png")); plt.close(fig)
log("main figures done")

# ================================================ 3. SAA CONVERGENCE ========
sizes, REP = [10, 25, 50, 100, 200, 400], 12
Tq = slice(0, 300)                                           # fixed test subset
PVb, LDb = smoothed_bootstrap(PVh, LDh, 1000, np.random.default_rng(77), sy.pv_kwp)   # fresh draws from the training model
rows = []
for S in sizes:
    for r in range(REP):
        P_, L_ = smoothed_bootstrap(PVh, LDh, S, np.random.default_rng(1000 * S + r), sy.pv_kwp)
        xs, o = solve_saa(P_, L_, sy, return_obj=True)
        rows.append(dict(S=S, rep=r, insample=o, oos_true=ev.costs(xs, Tp[Tq], Tl[Tq]).mean(),
                         oos_model=ev.costs(xs, PVb, LDb).mean(), x_total=xs.sum(), x_h21=xs[21], x_h22=xs[22]))
    log(f"convergence S={S}")
cv_raw = pd.DataFrame(rows); cv_raw.to_csv(os.path.join(RES, "saa_replications_raw.csv"), index=False)
cs = cv_raw.groupby("S").agg(insample=("insample", "mean"), oos_model=("oos_model", "mean"), oos_true=("oos_true", "mean"),
                             oos_true_std=("oos_true", "std"), total_purchase_std=("x_total", "std")).reset_index()
cs["optimism_gap"] = cs.oos_model - cs.insample        # for COST: in-sample is optimistic (too LOW)
cs.round(2).to_csv(os.path.join(RES, "table6_saa_convergence.csv"), index=False)
SUM["convergence"] = cs.round(2).to_dict(orient="records")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(cs.S, cs.insample, "o-", color=C["det"], label="in-sample SAA cost"); ax[0].plot(cs.S, cs.oos_model, "^-", color=C["ws"], label="out-of-sample (training model)")
ax[0].plot(cs.S, cs.oos_true, "s-", color=C["sto"], label="out-of-sample (true process)")
g = cv_raw.groupby("S")["oos_true"]; ax[0].fill_between(cs.S, g.quantile(.1).values, g.quantile(.9).values, color=C["sto"], alpha=.15)
ax[0].set_xscale("log"); ax[0].set_xlabel("number of scenarios S"); ax[0].set_ylabel("expected daily cost (INR)"); ax[0].legend(fontsize=7)
ax[0].set_title("SAA optimism shrinks with S (12 replications)")
ax[1].plot(cs.S, cs.oos_true_std, "o-", color=C["sto"], label="std of out-of-sample cost"); ax2 = ax[1].twinx()
ax2.plot(cs.S, cs.total_purchase_std, "s-", color=C["ws"]); ax2.set_ylabel("std of total purchase (kWh)", color=C["ws"]); ax2.grid(False)
ax[1].set_xscale("log"); ax[1].set_xlabel("number of scenarios S"); ax[1].set_ylabel("std of cost across replications (INR)"); ax[1].set_title("Stability of the solution")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_saa_convergence.png")); plt.close(fig)

# ================================================ 4. CVaR FRONTIER ==========
lams = [0.0, 0.25, 0.5, 0.75, 0.9]
PVc, LDc = smoothed_bootstrap(PVh, LDh, 200, np.random.default_rng(21), sy.pv_kwp)
rows = []
for lam in lams:
    xc = solve_saa(PVc, LDc, sy, lam=lam)
    cc, dc = ev.costs(xc, Tp[:1000], Tl[:1000], want_detail=True); r = risk_summary(cc)
    rows.append(dict(lam=lam, total_da_kwh=xc.sum(), mean_cost=r["mean"], std=r["std"], p95=r["p95"], cvar90=r["cvar90"],
                     max_cost=r["max"], shed_kwh=dc["shed"].mean()))
cvd = pd.DataFrame(rows); cvd.round(2).to_csv(os.path.join(RES, "table7_cvar_frontier.csv"), index=False)
SUM["cvar"] = cvd.round(2).to_dict(orient="records")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(cvd["std"], cvd.mean_cost, "o-", color=C["sto"])
for _, r in cvd.iterrows():
    ax[0].annotate(f"λ={r.lam:g}", (r["std"], r.mean_cost), textcoords="offset points", xytext=(4, 4), fontsize=7)
ax[0].set_xlabel("std of daily cost (INR)"); ax[0].set_ylabel("expected daily cost (INR)"); ax[0].set_title("Risk-cost frontier (mean-CVaR)")
ax[1].plot(cvd.lam, cvd.cvar90, "o-", color=C["det"], label="CVaR$_{90\\%}$ cost (worst 10 % days)"); ax[1].plot(cvd.lam, cvd.mean_cost, "s-", color=C["sto"], label="expected cost")
ax[1].set_xlabel("risk-aversion weight λ"); ax[1].set_ylabel("INR / day"); ax[1].legend(fontsize=7); ax[1].set_title("Trade-off as risk aversion grows")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_cvar_frontier.png")); plt.close(fig)
log("CVaR done")

# ================================================ 5. SENSITIVITY STUDIES ====
rows = []
for v in [0.05, 0.15, 0.25, 0.40]:
    r, _ = setting(replace(ws, p_overcast=v), sy, seed=10); rows.append(dict(factor="Overcast-day probability", value=v, **r))
for v in [0.0, 0.5, 1.0, 1.5, 2.0]:
    r, _ = setting(replace(ws, var_scale=v), sy, seed=20); rows.append(dict(factor="Weather variability multiplier", value=v, **r))
for v in [1.0, 1.5, 2.0, 3.0, 4.0]:
    r, _ = setting(ws, replace(sy, rt_premium=v), seed=30); rows.append(dict(factor="Real-time price multiplier", value=v, **r))
for v in [0, 200, 400, 800, 1200]:
    r, _ = setting(ws, replace(sy, batt_kwh=float(v), batt_kw=v / 4.0), seed=40); rows.append(dict(factor="Battery size (kWh)", value=v, **r))
sf = pd.DataFrame(rows); sf.round(2).to_csv(os.path.join(RES, "table8_sensitivity_factors.csv"), index=False)
SUM["sens"] = sf.round(2).to_dict(orient="records")
log("factor sweeps done")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
for a, fac, xl in zip(ax, ["Overcast-day probability", "Weather variability multiplier", "Real-time price multiplier"],
                      ["overcast-day probability", "weather variability multiplier", "real-time price multiplier"]):
    d = sf[sf.factor == fac]
    a.plot(d.value, d.VSS, "o-", color=C["ws"], lw=2); a.set_xlabel(xl); a.set_ylabel("VSS (INR/day)")
    a2 = a.twinx(); a2.plot(d.value, d.VSS_pct, "s--", color=C["grey"]); a2.set_ylabel("VSS (%)", color=C["grey"]); a2.grid(False)
    a.set_title(f"VSS vs {xl.split(' ')[0]} {xl.split(' ')[1]}", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_sensitivity_factors.png")); plt.close(fig)

# tornado: +/-20 % one at a time (stochastic plan re-optimised)
base, _ = setting(ws, sy, seed=50)
plist = [("RT price multiplier", lambda f: (ws, replace(sy, rt_premium=sy.rt_premium * f))),
         ("Export price", lambda f: (ws, replace(sy, sell_price=sy.sell_price * f))),
         ("Value of lost load", lambda f: (ws, replace(sy, voll=sy.voll * f))),
         ("Battery wear cost", lambda f: (ws, replace(sy, degr=sy.degr * f))),
         ("Battery efficiency", lambda f: (ws, replace(sy, eta=min(0.99, sy.eta * f)))),
         ("PV capacity", lambda f: (ws, replace(sy, pv_kwp=sy.pv_kwp * f))),
         ("Grid limit", lambda f: (ws, replace(sy, grid_kw=sy.grid_kw * f))),
         ("Battery size", lambda f: (ws, replace(sy, batt_kwh=sy.batt_kwh * f, batt_kw=sy.batt_kw * f))),
         ("Load level", lambda f: (replace(ws, load_shift=f), sy))]
rows = []
for nm, fn in plist:
    out = {}
    for f in (0.8, 1.2):
        w_, s_ = fn(f); r, _ = setting(w_, s_, seed=50); out[f] = r["stoch"] - base["stoch"]
    rows.append(dict(parameter=nm, low_minus20=out[0.8], high_plus20=out[1.2], swing=abs(out[1.2] - out[0.8])))
tor = pd.DataFrame(rows).sort_values("swing", ascending=False)
tor.round(2).to_csv(os.path.join(RES, "table9_sensitivity_parameters.csv"), index=False)
SUM["tornado"] = tor.round(2).to_dict(orient="records"); SUM["tornado_base_cost"] = float(base["stoch"])
t_ = tor.iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.barh(t_.parameter, t_.low_minus20, color=C["det"], alpha=.8, label="-20 %"); ax.barh(t_.parameter, t_.high_plus20, color=C["sto"], alpha=.8, label="+20 %")
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("change in expected daily cost vs base (INR/day)"); ax.legend(); ax.set_title("One-at-a-time parameter sensitivity")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_tornado.png")); plt.close(fig)
log("tornado done")

# distribution shift: plans trained on the base history, tested in other worlds, versus re-optimised oracle
worlds = {"Base (as trained)": ws,
          "Monsoon-heavy (35 % overcast)": replace(ws, p_overcast=0.35),
          "Clear season (3 % overcast)": replace(ws, p_overcast=0.03),
          "Load +10 %": replace(ws, load_shift=1.10), "Load -10 %": replace(ws, load_shift=0.90),
          "Weather variability x1.5": replace(ws, var_scale=1.5)}
rows = []
for i, (nm, w_) in enumerate(worlds.items()):
    Pw, Lw, _ = sample_days(w_, 600, np.random.default_rng(300 + i), sy.pv_kwp)
    Ph_, Lh_, _ = sample_days(w_, 365, np.random.default_rng(400 + i), sy.pv_kwp)
    Po, Lo = smoothed_bootstrap(Ph_, Lh_, 200, np.random.default_rng(500 + i), sy.pv_kwp)
    x_or = solve_saa(Po, Lo, sy)
    co, cs_, cd_ = ev.costs(x_or, Pw, Lw).mean(), ev.costs(x_sto, Pw, Lw).mean(), ev.costs(x_det, Pw, Lw).mean()
    rows.append(dict(world=nm, oracle=co, stochastic=cs_, deterministic=cd_, regret_stoch=cs_ - co, regret_det=cd_ - co, det_minus_stoch=cd_ - cs_))
ms = pd.DataFrame(rows); ms.round(2).to_csv(os.path.join(RES, "table10_distribution_shift.csv"), index=False)
SUM["shift"] = ms.round(2).to_dict(orient="records")
log("distribution shift done")

bsz = sf[sf.factor == "Battery size (kWh)"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(bsz.value, bsz.stoch, "o-", color=C["sto"], label="stochastic"); ax[0].plot(bsz.value, bsz.det, "s-", color=C["det"], label="deterministic")
ax[0].set_xlabel("battery energy (kWh)"); ax[0].set_ylabel("expected daily cost (INR)"); ax[0].legend(); ax[0].set_title("Value of storage")
xi = np.arange(len(ms)); w = .38
ax[1].bar(xi - w / 2, ms.regret_stoch, w, color=C["sto"], label="stochastic"); ax[1].bar(xi + w / 2, ms.regret_det, w, color=C["det"], label="deterministic")
ax[1].set_xticks(xi); ax[1].set_xticklabels([n.replace(" (", "\n(", 1).replace(" ", "\n", 1) if len(n) > 14 else n for n in ms.world], fontsize=6)
ax[1].set_ylabel("regret vs re-optimised (INR/day)"); ax[1].legend(fontsize=7); ax[1].set_title("Robustness to distribution shift")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_battery_and_shift.png")); plt.close(fig)

# ================================================ 6. SDG 9.5 impact =========
ms_, md_ = SUM["main"]["met_sto"], SUM["main"]["met_det"]
imp = pd.DataFrame([
    ("Net grid import (kWh/day)", md_["net_import_kwh"], ms_["net_import_kwh"]),
    ("CO2 from grid import (kg/day, EF=0.71)", md_["co2_kg"], ms_["co2_kg"]),
    ("PV energy curtailed (kWh/day)", md_["curtailed_kwh"], ms_["curtailed_kwh"]),
    ("Expected unserved energy (kWh/day)", md_["shed_kwh"], ms_["shed_kwh"]),
    ("Days with any load shedding (%)", md_["lolp_pct"], ms_["lolp_pct"]),
    ("Real-time (expensive) purchases (kWh/day)", md_["rt_kwh"], ms_["rt_kwh"]),
    ("Surplus energy resold at low price (kWh/day)", md_["resold_kwh"], ms_["resold_kwh"]),
    ("Expected energy cost (INR/day)", md_["mean_cost"], ms_["mean_cost"]),
], columns=["indicator", "deterministic", "stochastic"])
imp["change"] = imp.stochastic - imp.deterministic
imp.round(2).to_csv(os.path.join(RES, "table11_sdg_impact_metrics.csv"), index=False)
SUM["impact"] = imp.round(2).to_dict(orient="records")
SUM["config"] = dict(S_main=S_MAIN, N_test=N_TEST, N_ws=N_WS, N_hist=N_HIST, sizes=sizes, reps=REP, EF=EF,
                     system=dict(pv_kwp=sy.pv_kwp, batt_kwh=sy.batt_kwh, batt_kw=sy.batt_kw, eta=sy.eta, grid_kw=sy.grid_kw,
                                 rt_premium=sy.rt_premium, sell_price=sy.sell_price, voll=sy.voll, degr=sy.degr),
                     weather=dict(p=ws.probs().tolist(), kappa_mean=ws.kappa_mean, kappa_sd=ws.kappa_sd, hourly_sd=ws.hourly_sd))
with open(os.path.join(RES, "summary.json"), "w") as f:
    json.dump(SUM, f, indent=2, default=float)
log("ALL DONE - results written")

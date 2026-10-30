"""
Core module: weather/load uncertainty model, two-stage stochastic LP (SAA) for day-ahead
energy procurement of a solar + battery campus microgrid, CVaR risk-averse variant,
deterministic baseline, wait-and-see bound and exact second-stage (recourse) evaluation.

Problem: a campus microgrid (PV + battery + grid connection) must COMMIT to hourly
day-ahead grid purchases x_t BEFORE tomorrow's solar output and electricity load are known
(first stage). During the day, the real-time recourse is: buy extra at a premium price,
sell surplus cheaply, charge/discharge the battery, curtail PV or (last resort) shed load.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

T = 24
HOURS = np.arange(T)

# --------------------------------------------------------------------------- #
# System parameters
# --------------------------------------------------------------------------- #
DA_PRICE = np.array([5.0] * 6 + [6.5] * 4 + [6.0] * 8 + [9.5] * 4 + [6.0] * 2)   # INR/kWh time-of-use
LOAD_BASE = np.array([190, 180, 175, 172, 175, 190, 220, 270, 320, 350, 365, 375,
                      380, 375, 370, 365, 360, 370, 395, 410, 400, 360, 280, 215], dtype=float)  # kW
SHAPE = np.array([max(0.0, np.sin(np.pi * (h + 0.5 - 6) / 12)) ** 1.2 if 6 <= h < 18 else 0.0 for h in HOURS])


@dataclass(frozen=True)
class System:
    pv_kwp: float = 600.0          # PV capacity (kW)
    batt_kwh: float = 800.0        # battery energy (kWh)
    batt_kw: float = 200.0         # battery charge/discharge power (kW)
    eta: float = 0.92              # one-way efficiency
    soc_min: float = 0.10          # min state of charge (fraction)
    soc_init: float = 0.50         # initial = final state of charge (fraction)
    grid_kw: float = 450.0         # grid connection limit (kW)
    rt_premium: float = 2.0        # real-time price = premium * day-ahead price
    sell_price: float = 3.0        # INR/kWh paid for exported / resold energy
    voll: float = 60.0             # INR per kWh of unserved load (value of lost load)
    degr: float = 1.2              # INR per kWh discharged (battery wear)

    @property
    def rt_price(self):
        return self.rt_premium * DA_PRICE


# --------------------------------------------------------------------------- #
# Uncertainty model ("true" data-generating process)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WeatherSpec:
    p_sunny: float = 0.55
    p_partly: float = 0.30
    p_overcast: float = 0.15
    kappa_mean: tuple = (0.88, 0.60, 0.28)     # daily clearness index by state
    kappa_sd: tuple = (0.06, 0.12, 0.10)
    hourly_sd: tuple = (0.05, 0.25, 0.20)      # intra-day cloud noise by state (AR(1), phi=0.7)
    load_sd: float = 0.05                      # day-level load noise
    load_temp_coef: float = 0.06               # hot/sunny days -> higher cooling load
    load_shift: float = 1.0                    # global load multiplier (stress tests)
    var_scale: float = 1.0                     # scales kappa deviations (weather variability)

    def probs(self):
        p = np.array([self.p_sunny, self.p_partly, self.p_overcast])
        return p / p.sum()


def sample_days(ws: WeatherSpec, n: int, rng: np.random.Generator, pv_kwp: float = 600.0):
    """Return PV (n x 24 kW), LOAD (n x 24 kW), state (n,) drawn from the true process."""
    state = rng.choice(3, size=n, p=ws.probs())
    mu = np.array(ws.kappa_mean)[state]
    sd = np.array(ws.kappa_sd)[state] * ws.var_scale
    kappa = np.clip(mu + sd * rng.standard_normal(n), 0.05, 1.0)
    hsd = np.array(ws.hourly_sd)[state][:, None]
    eps = np.zeros((n, T)); e = rng.standard_normal(n) * hsd[:, 0]
    for t in range(T):
        e = 0.7 * e + np.sqrt(1 - 0.49) * rng.standard_normal(n) * hsd[:, 0]
        eps[:, t] = e
    pv = pv_kwp * SHAPE[None, :] * kappa[:, None] * (1 + eps)
    pv = np.clip(pv, 0, pv_kwp * SHAPE[None, :] * 1.05)
    f = (1 + ws.load_temp_coef * (kappa - 0.6) / 0.3) * np.exp(ws.load_sd * rng.standard_normal(n)) * ws.load_shift
    lz = np.zeros((n, T)); l = 0.03 * rng.standard_normal(n)
    for t in range(T):
        l = 0.6 * l + 0.8 * 0.03 * rng.standard_normal(n)
        lz[:, t] = l
    load = LOAD_BASE[None, :] * f[:, None] * (1 + lz)
    return pv, load, state


def smoothed_bootstrap(PVh: np.ndarray, LDh: np.ndarray, n: int, rng: np.random.Generator, pv_kwp: float = 600.0):
    """Scenario generator used by the optimiser: resample historical days with small multiplicative jitter."""
    idx = rng.integers(0, len(PVh), n)
    pv = PVh[idx] * np.exp(0.04 * rng.standard_normal((n, 1)))
    pv = np.minimum(pv, pv_kwp * SHAPE[None, :] * 1.05)
    ld = LDh[idx] * np.exp(0.02 * rng.standard_normal((n, 1)))
    return pv, ld


# --------------------------------------------------------------------------- #
# LP building blocks
# --------------------------------------------------------------------------- #
# per-scenario variable layout (168): u, z, c, d, e, l, w   (24 each)
NB = 7 * T
iu, iz, ic, idd, ie, il, iw = (k * T for k in range(7))


def _block(sy: System):
    """Constraint matrices of ONE scenario block (second stage) plus coupling to x."""
    I = sp.identity(T, format="csr"); Z = sp.csr_matrix((T, T))
    # energy balance: x + u - w + d + l - c - z = load - pv
    # columns in order [u, z, c, d, e, l, w]
    Abal = sp.hstack([I, -I, -I, I, Z, I, -I]).tocsr()
    Bx = I.tocsr()                                         # coefficient of x_t in balance
    # SOC dynamics: e_t - e_{t-1} - eta*c_t + d_t/eta = 0   (e_{-1} = E0 -> moved to RHS at t=0)
    D = sp.identity(T, format="lil")
    for t in range(1, T):
        D[t, t - 1] = -1.0
    Asoc = sp.hstack([Z, Z, -sy.eta * I, I / sy.eta, D.tocsr(), Z, Z]).tocsr()
    # import limit: x + u <= G  ->  u_t <= G - x_t
    Aimp = sp.hstack([I, Z, Z, Z, Z, Z, Z]).tocsr()
    return Abal, Bx, Asoc, Aimp


def _cost_vec(sy: System):
    c = np.zeros(NB)
    c[iu:iu + T] = sy.rt_price
    c[iz:iz + T] = -sy.sell_price
    c[idd:idd + T] = sy.degr
    c[il:il + T] = sy.voll
    return c


def _bounds_block(sy: System, pv: np.ndarray, load: np.ndarray):
    E, E0 = sy.batt_kwh, sy.soc_init * sy.batt_kwh
    lb = np.zeros(NB); ub = np.full(NB, np.inf)
    ub[ic:ic + T] = sy.batt_kw; ub[idd:idd + T] = sy.batt_kw
    lb[ie:ie + T] = sy.soc_min * E; ub[ie:ie + T] = E
    lb[ie + T - 1] = ub[ie + T - 1] = E0                      # cyclic SOC
    ub[il:il + T] = load
    ub[iw:iw + T] = pv
    return lb, ub


def second_stage_cost(x: np.ndarray, pv: np.ndarray, load: np.ndarray, sy: System, detail: bool = False):
    """Exact recourse cost (INR) of ONE scenario given day-ahead purchases x (kWh per hour)."""
    Abal, Bx, Asoc, Aimp = _block(sy)
    Aeq = sp.vstack([Abal, Asoc]).tocsr()
    E0 = sy.soc_init * sy.batt_kwh
    rhs_soc = np.zeros(T); rhs_soc[0] = E0
    beq = np.concatenate([load - pv - x, rhs_soc])
    bub = sy.grid_kw - x
    lb, ub = _bounds_block(sy, pv, load)
    r = linprog(_cost_vec(sy), A_ub=Aimp, b_ub=bub, A_eq=Aeq, b_eq=beq,
                bounds=np.column_stack([lb, np.where(np.isinf(ub), None, ub)]), method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    if detail:
        v = r.x
        return r.fun, dict(u=v[iu:iu + T], z=v[iz:iz + T], c=v[ic:ic + T], d=v[idd:idd + T],
                           e=v[ie:ie + T], l=v[il:il + T], w=v[iw:iw + T])
    return r.fun


class Evaluator:
    """Fast repeated recourse evaluation: block matrices are built once."""
    def __init__(self, sy: System):
        self.sy = sy
        Abal, Bx, Asoc, Aimp = _block(sy)
        self.Aeq = sp.vstack([Abal, Asoc]).tocsr(); self.Aimp = Aimp
        self.c = _cost_vec(sy); self.E0 = sy.soc_init * sy.batt_kwh

    def costs(self, x, PV, LD, want_detail=False):
        sy = self.sy; n = len(PV)
        out = np.empty(n); shed = np.zeros(n); rtbuy = np.zeros(n); curt = np.zeros(n); sold = np.zeros(n)
        rhs_soc = np.zeros(T); rhs_soc[0] = self.E0
        for s in range(n):
            lb, ub = _bounds_block(sy, PV[s], LD[s])
            r = linprog(self.c, A_ub=self.Aimp, b_ub=sy.grid_kw - x, A_eq=self.Aeq,
                        b_eq=np.concatenate([LD[s] - PV[s] - x, rhs_soc]),
                        bounds=np.column_stack([lb, np.where(np.isinf(ub), None, ub)]), method="highs")
            out[s] = r.fun
            v = r.x; shed[s] = v[il:il + T].sum(); rtbuy[s] = v[iu:iu + T].sum()
            curt[s] = v[iw:iw + T].sum(); sold[s] = v[iz:iz + T].sum()
        total = out + float(DA_PRICE @ x)
        if want_detail:
            return total, dict(shed=shed, rt_kwh=rtbuy, curtail=curt, resold=sold)
        return total


# --------------------------------------------------------------------------- #
# Two-stage stochastic LP (extensive form / SAA) with optional CVaR(cost)
# --------------------------------------------------------------------------- #
def solve_saa(PV: np.ndarray, LD: np.ndarray, sy: System, lam: float = 0.0, tail: float = 0.10,
              return_obj: bool = False):
    """
    min  (1-lam) E[cost] + lam CVaR_{1-tail}[cost]   over S scenarios (rows of PV, LD).
    lam = 0 : risk-neutral stochastic programme (SAA).  S = 1 with mean profiles : deterministic baseline.
    First-stage decision: day-ahead purchase x (24 hourly kWh). Returns x (and optimal SAA objective).
    """
    S = len(PV)
    use_cv = lam > 0
    nvar = T + S * NB + (1 + S if use_cv else 0)
    ieta = T + S * NB
    Abal, Bx, Asoc, Aimp = _block(sy)

    Eq_blocks = sp.block_diag([sp.vstack([Abal, Asoc])] * S, format="csr")
    Eq_x = sp.vstack([sp.vstack([Bx, sp.csr_matrix((T, T))])] * S, format="csr")
    A_eq = sp.hstack([Eq_x, Eq_blocks]).tocsr()
    E0 = sy.soc_init * sy.batt_kwh
    rhs_soc = np.zeros(T); rhs_soc[0] = E0
    b_eq = np.concatenate([np.concatenate([LD[s] - PV[s], rhs_soc]) for s in range(S)])
    if use_cv:
        A_eq = sp.hstack([A_eq, sp.csr_matrix((A_eq.shape[0], 1 + S))]).tocsr()

    Imp_blocks = sp.block_diag([Aimp] * S, format="csr")
    Imp_x = sp.vstack([sp.identity(T, format="csr")] * S, format="csr")   # x + u <= G
    A_ub = sp.hstack([Imp_x, Imp_blocks]).tocsr()
    b_ub = np.full(S * T, sy.grid_kw)
    if use_cv:
        A_ub = sp.hstack([A_ub, sp.csr_matrix((A_ub.shape[0], 1 + S))]).tocsr()

    cb = _cost_vec(sy)
    c = np.zeros(nvar)
    w = 1.0 - lam
    c[:T] = w * DA_PRICE                              # risk-neutral part; the CVaR part carries DA.x through the z_s constraints
    c[T:T + S * NB] = np.tile(cb, S) * w / S
    if use_cv:
        # cost_s = DA.x + cb.block_s ;  CVaR = eta + 1/(tail S) sum z_s ,  z_s >= cost_s - eta
        c[ieta] = lam
        c[ieta + 1:] = lam / (tail * S)
        rows = sp.lil_matrix((S, nvar))
        for s in range(S):
            rows[s, :T] = DA_PRICE
            rows[s, T + s * NB:T + (s + 1) * NB] = cb
            rows[s, ieta] = -1.0
            rows[s, ieta + 1 + s] = -1.0
        A_ub = sp.vstack([A_ub, rows.tocsr()]).tocsr()
        b_ub = np.concatenate([b_ub, np.zeros(S)])
    lbs, ubs = [np.zeros(T)], [np.full(T, sy.grid_kw)]
    for s in range(S):
        lb, ub = _bounds_block(sy, PV[s], LD[s]); lbs.append(lb); ubs.append(ub)
    if use_cv:
        lbs.append(np.array([-np.inf])); ubs.append(np.array([np.inf]))
        lbs.append(np.zeros(S)); ubs.append(np.full(S, np.inf))
    lb = np.concatenate(lbs); ub = np.concatenate(ubs)
    bounds = np.column_stack([np.where(np.isinf(lb), None, lb), np.where(np.isinf(ub), None, ub)])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x[:T]
    return (x, res.fun) if return_obj else x


def wait_and_see(PV, LD, sy: System):
    """Perfect-information cost per scenario (x chosen after seeing the day)."""
    out = np.empty(len(PV))
    for s in range(len(PV)):
        out[s] = solve_saa(PV[s:s + 1], LD[s:s + 1], sy, return_obj=True)[1]
    return out


def risk_summary(cost: np.ndarray, tail: float = 0.10) -> dict:
    c = np.sort(np.asarray(cost)); k = max(1, int(np.ceil(tail * len(c))))
    return dict(mean=float(c.mean()), std=float(c.std(ddof=1)), p50=float(np.median(c)),
                p95=float(np.quantile(c, .95)), max=float(c.max()), cvar90=float(c[-k:].mean()))

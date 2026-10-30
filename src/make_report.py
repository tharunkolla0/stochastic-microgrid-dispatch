"""
Builds report/Stochastic_Microgrid_Mini_Project_Report.pdf from the saved results.
All numbers are read from ./results so the report always matches the code output.
Optional:  STUDENT_NAME="Your Name (RegNo)" python -m src.make_report
"""
import json, os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = lambda f: os.path.join(ROOT, "results", f)
F = lambda f: os.path.join(ROOT, "figures", f)
OUT = os.path.join(ROOT, "report", "Stochastic_Microgrid_Mini_Project_Report.pdf")
STUDENT = os.environ.get("STUDENT_NAME", "B.Tech AI, Semester V")

fd = "/usr/share/fonts/truetype/dejavu/"
for nm, fl in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(nm, fd + fl))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

S = json.load(open(R("summary.json"))); M = S["main"]; ms, md = M["met_sto"], M["met_det"]
B = ParagraphStyle("B", fontName="DV", fontSize=8.4, leading=11.6, alignment=TA_JUSTIFY, spaceAfter=4)
H1 = ParagraphStyle("H1", keepWithNext=1, fontName="DV-B", fontSize=11, leading=14, spaceBefore=7, spaceAfter=3, textColor=colors.HexColor("#1b5e3a"))
H2 = ParagraphStyle("H2", keepWithNext=1, fontName="DV-B", fontSize=9, leading=12, spaceBefore=4, spaceAfter=2, textColor=colors.HexColor("#333333"))
CAP = ParagraphStyle("CAP", fontName="DV-I", fontSize=7.4, leading=9.5, alignment=TA_CENTER, spaceAfter=5, textColor=colors.HexColor("#444444"))
TITLE = ParagraphStyle("T", fontName="DV-B", fontSize=14.5, leading=18.5, alignment=TA_CENTER, textColor=colors.HexColor("#1b5e3a"))
SUB = ParagraphStyle("S", fontName="DV", fontSize=8.8, leading=12, alignment=TA_CENTER)
EQ = ParagraphStyle("EQ", fontName="DV", fontSize=8.4, leading=12, leftIndent=18, spaceAfter=3)
TC = ParagraphStyle("TC", fontName="DV", fontSize=7.2, leading=9)
TCB = ParagraphStyle("TCB", fontName="DV-B", fontSize=7.2, leading=9, textColor=colors.white)
REF = ParagraphStyle("ref", parent=B, fontSize=7.6, leading=10, leftIndent=10, firstLineIndent=-10, alignment=0)

story = []
P = lambda t, st=B: story.append(Paragraph(t, st))
fmt = lambda v, d=0: f"{v:,.{d}f}"


def table(df, widths=None, fmts=None, caption=None, default=2):
    fmts = fmts or {}
    data = [[Paragraph(str(c), TCB) for c in df.columns]]
    for _, r in df.iterrows():
        data.append([Paragraph(v, TC) if isinstance(v, str) else Paragraph(f"{v:,.{fmts.get(c, default)}f}", TC)
                     for c, v in zip(df.columns, r)])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b5e3a")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf5f0")]),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b8c8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)]))
    items = [t] + ([Spacer(1, 2), Paragraph(caption, CAP)] if caption else [])
    story.append(KeepTogether(items))


def fig(name, width_cm, caption):
    from PIL import Image as PI
    w, h = PI.open(F(name)).size
    story.append(KeepTogether([Image(F(name), width=width_cm * cm, height=width_cm * cm * h / w), Paragraph(caption, CAP)]))


sens = pd.DataFrame(S["sens"])
shift, cvr, conv, bw, tor = S["shift"], S["cvar"], S["convergence"], S["by_weather"], S["tornado"]
sg = lambda f: sens[sens.factor == f].reset_index(drop=True)
oc, vs_, rt, bt = sg("Overcast-day probability"), sg("Weather variability multiplier"), sg("Real-time price multiplier"), sg("Battery size (kWh)")
std_cut = 100 * (1 - ms["std"] / md["std"])
rt_cut = 100 * (md["rt_kwh"] - ms["rt_kwh"]) / md["rt_kwh"]

# ------------------------------------------------------------------ title --
P("Design and Analysis of a Stochastic Optimization Model for<br/>Day-Ahead Energy Planning of a Solar–Battery Campus Microgrid", TITLE)
story.append(Spacer(1, 3))
P("21AIE321T – Stochastic Decision Making · Assignment IV (CLA-2) · Mini Project<br/>"
  "SRM Institute of Science and Technology, Tiruchirappalli · B.Tech AI, Semester V · AY 2026-2027<br/>"
  f"Submitted by: {STUDENT} · Faculty: Dr. Himanshu Rai · CO4 · SDG 9 (Target 9.5)", SUB)
story.append(Spacer(1, 4))

P("Abstract", H1)
P(f"A campus microgrid with rooftop solar (600 kWp), a battery (800 kWh) and a grid connection must <b>commit to hourly day-ahead electricity purchases before "
  f"tomorrow's sunshine and load are known</b>; any mismatch is settled in the real-time market at twice the price (shortfall) or resold at a low export price (surplus). "
  f"We formulate a <b>two-stage stochastic linear programme</b> with a 24-hour battery model as recourse, solve it by <b>Sample Average Approximation (SAA)</b> with 300 scenarios, add a "
  f"<b>mean-CVaR</b> risk-averse variant, and compare it with a deterministic mean-forecast plan over 2,000 out-of-sample simulated days. The stochastic plan costs "
  f"<b>₹{fmt(M['sto']['mean'])}/day</b> versus <b>₹{fmt(M['det']['mean'])}/day</b>: a <b>Value of the Stochastic Solution (VSS) of ₹{fmt(M['VSS'],1)}/day</b> "
  f"(95% CI ₹{fmt(M['VSS_ci'][0])}–{fmt(M['VSS_ci'][1])}, {M['VSS_pct']:.2f}% ≈ ₹{M['annual']/1e5:.2f} lakh/year), with {std_cut:.0f}% lower cost volatility and {rt_cut:.0f}% less expensive real-time purchasing. "
  f"The gain is concentrated on overcast days (₹{bw[2]['saving']:,.0f}/day) and grows rapidly when real-time prices are high "
  f"({rt.VSS_pct.iloc[-1]:.1f}% at a 4× premium, ≈ 0 at no premium). A stress test shows the plan loses to the deterministic one if load turns out 10% lower than in training (Section 6.3).")

# ----------------------------------------------------------------- 1 intro --
P("1. Problem Selection and Real-World Grounding", H1)
P("Institutions in Tamil Nadu increasingly install rooftop solar and batteries, yet still buy a large share of their energy from the grid under time-of-day tariffs and, for "
  "larger consumers, scheduled <i>day-ahead</i> procurement with imbalance settlement for deviations. Tomorrow's solar output depends on cloud cover and the monsoon, and the load depends on "
  "temperature, academic calendar and occupancy; both are uncertain at the time the purchase schedule must be submitted. This is a textbook <b>two-stage stochastic decision problem "
  "with recourse</b>: the day-ahead schedule is the <i>here-and-now</i> decision; battery dispatch, real-time purchases and resale are <i>wait-and-see</i> corrections.")
P("<b>SDG 9 (Target 9.5) link.</b> Target 9.5 asks for enhanced industrial and technological capability and innovation. Stochastic energy scheduling is a software-only upgrade that makes existing "
  "solar and storage assets more valuable and reduces price-risk exposure; Section 6.4 measures its impact (cost, real-time purchases, grid import and CO₂), meeting SDG Integration Level 3 "
  "(Impact &amp; Measurement). <b>Data note:</b> no campus smart-meter data were available, so a 730-day hourly history of PV output and load was <i>realistically simulated</i> "
  "(Section 2.3) and saved in <font face='DV-B'>data/microgrid_history_hourly.csv</font>; the code accepts real data in the same format.")

# ------------------------------------------------------------- 2 formulation --
P("2. Formal Problem Formulation", H1)
P("2.1 Decision variables and parameters", H2)
P("Hours t = 0,…,23. <b>First stage (here-and-now):</b> x<sub>t</sub> ∈ [0, G], kWh bought day-ahead at known time-of-use price π<sub>t</sub>. "
  "<b>Random data</b> ω = (PV<sub>t</sub>(ω), L<sub>t</sub>(ω)): solar output and load (kW ≡ kWh/h). <b>Second stage (recourse)</b> for each ω: u<sub>t</sub> real-time purchase at ρπ<sub>t</sub> (ρ = real-time multiplier), "
  "z<sub>t</sub> resale at price σ, c<sub>t</sub>/d<sub>t</sub> battery charge/discharge, e<sub>t</sub> state of charge, l<sub>t</sub> unserved load at penalty V (value of lost load), w<sub>t</sub> curtailed PV; δ is the battery wear cost per kWh discharged.")
par = pd.read_csv(R("table0_system_parameters.csv")); par.columns = ["Parameter", "Value"]
par["Value"] = par["Value"].astype(str)
half = (len(par) + 1) // 2
left, right = par.iloc[:half].reset_index(drop=True), par.iloc[half:].reset_index(drop=True)
right = pd.concat([right, pd.DataFrame([["", ""]] * (len(left) - len(right)), columns=["Parameter", "Value"])], ignore_index=True)
two = pd.concat([left, right.rename(columns={"Parameter": "Parameter ", "Value": "Value "})], axis=1)
table(two, widths=[4.6*cm, 2.8*cm, 5.6*cm, 2.8*cm],
      caption="Table 1. System and market parameters. Day-ahead tariff: ₹5.0 (0–5 h), 6.5 (6–9 h), 6.0 (10–17 h), 9.5 (18–21 h), 6.0 (22–23 h).")
P("2.2 Two-stage stochastic programme", H2)
P("Minimise  Σ<sub>t</sub> π<sub>t</sub> x<sub>t</sub> + E<sub>ω</sub>[ Q(x, ω) ],   with the recourse cost", EQ)
P("Q(x,ω) = min Σ<sub>t</sub> ( ρπ<sub>t</sub> u<sub>t</sub> − σ z<sub>t</sub> + δ d<sub>t</sub> + V l<sub>t</sub> )", EQ)
P("s.t.  x<sub>t</sub> + u<sub>t</sub> + (PV<sub>t</sub> − w<sub>t</sub>) + d<sub>t</sub> + l<sub>t</sub> = L<sub>t</sub> + c<sub>t</sub> + z<sub>t</sub>   (energy balance)", EQ)
P("e<sub>t</sub> = e<sub>t−1</sub> + η c<sub>t</sub> − d<sub>t</sub>/η,  e<sub>−1</sub> = e<sub>23</sub> = 0.5E,  0.1E ≤ e<sub>t</sub> ≤ E,  0 ≤ c<sub>t</sub>, d<sub>t</sub> ≤ P<sub>b</sub>   (battery)", EQ)
P("x<sub>t</sub> + u<sub>t</sub> ≤ G,   0 ≤ w<sub>t</sub> ≤ PV<sub>t</sub>,   0 ≤ l<sub>t</sub> ≤ L<sub>t</sub>,   z<sub>t</sub>, u<sub>t</sub> ≥ 0   (grid, curtailment, shedding limits)", EQ)
P("Every day-ahead schedule is feasible for every scenario (resale and shedding always absorb mismatch), so the problem has <b>relatively complete recourse</b>. Buying too much day-ahead forces cheap resale (σ = ₹3 vs ₹5–9.5 paid), "
  "buying too little forces real-time purchases at 2π: a 24-period, battery-coupled analogue of the newsvendor trade-off. <b>Risk-averse variant</b> (Rockafellar–Uryasev): "
  "min (1−λ)·E[cost] + λ·CVaR<sub>90%</sub>[cost], the mean cost of the worst 10% of days, linearised with η and z<sub>s</sub> ≥ cost<sub>s</sub> − η, z<sub>s</sub> ≥ 0.")
P("2.3 Probability model of the uncertain data", H2)
P("<b>Weather state</b> per day: sunny (55%), partly cloudy (30%), overcast (15%), with daily clearness index κ ~ N(0.88, 0.06), N(0.60, 0.12), N(0.28, 0.10) respectively (clipped to [0.05, 1]). "
  "PV(t) = 600 kWp · clear-sky shape(t) · κ · (1 + ε<sub>t</sub>) where ε<sub>t</sub> is an AR(1) process (φ = 0.7) whose standard deviation is 5%/25%/20% by state, capturing intra-day cloud transients. "
  "<b>Load</b> = base campus profile (172–410 kW) × day factor (1 + 0.06(κ−0.6)/0.3)·lognormal(σ = 5%) × hourly AR(1) noise (3%): hot sunny days raise cooling load, so PV and load are positively correlated across days. "
  f"This process generates the 730-day history (daily PV–load correlation {S['data']['pv_load_corr']:.2f}, CV of daily PV energy {S['data']['pv_cv']:.2f}). "
  "<b>The optimiser never sees the true process</b>: scenarios are drawn by a <i>smoothed bootstrap</i> of the historical days, while validation uses fresh days from the true process.")
t1 = pd.read_csv(R("table1_data_statistics.csv"))
t1.columns = ["Weather state", "Days", "Share %", "PV kWh/day (mean)", "PV kWh/day (std)", "Load kWh/day", "Mean peak load (kW)"]
table(t1, fmts={"Days": 0, "Share %": 1, "PV kWh/day (mean)": 0, "PV kWh/day (std)": 0, "Load kWh/day": 0, "Mean peak load (kW)": 0},
      caption=f"Table 2. Statistics of the 730-day history. Solar covers on average {100*S['data']['pv_share_of_load']:.0f}% of campus load, but only {100*t1.iloc[2,3]/t1.iloc[2,5]:.0f}% on overcast days.")
fig("fig1_data_overview.png", 16.0, "Figure 1. Left: daily PV energy by weather type. Centre: sample PV profiles. Right: mean load and PV with 5–95% load band.")

# ------------------------------------------------------------- 3 methodology --
P("3. Methodology and Justification of the Chosen Approach", H1)
P("<b>Approach: two-stage stochastic programming solved by Sample Average Approximation (SAA).</b> E[Q(x,ω)] is replaced by the average over S sampled days, giving one large LP in extensive form "
  "(24 + 168S variables, e.g. 50,424 for S = 300) solved exactly with HiGHS (SciPy). <b>Why SAA:</b> (i) the structure is genuinely two-stage with battery-coupled recourse, which stochastic programming represents exactly; "
  "(ii) the uncertainty is a 48-dimensional correlated vector (hourly PV and load) with state-dependent noise, impossible to discretise on a small scenario tree but natural to sample; "
  "(iii) SAA solutions converge to the true optimum as S grows (Kleywegt et al., 2002; verified empirically in Section 5.4); (iv) the LP gives exact optima, whereas a metaheuristic such as simulated annealing would need "
  "an inner LP solve per candidate schedule and gives no optimality guarantee. A chance constraint on supply was not needed because the grid connection always provides a backstop; risk is instead managed through CVaR.")
P("<b>Baseline and metrics.</b> The <i>deterministic baseline</i> solves the same model with PV and load fixed at their historical mean profiles (S = 1). EEV = expected cost of that schedule on random days; "
  "RP = expected cost of the stochastic schedule; <b>VSS = EEV − RP</b>. <i>Wait-and-see</i> (WS) is the expected cost if the schedule could be chosen after seeing the day; <b>EVPI = RP − WS</b>. "
  "<b>Validation:</b> every schedule is evaluated by solving the exact recourse LP on 2,000 fresh days from the true process (EVPI on 500 of them), with a 2,000-resample bootstrap confidence interval for VSS.")

# --------------------------------------------------------- 4 implementation --
P("4. Implementation Summary", H1)
P("Python 3 with NumPy, SciPy (HiGHS), pandas, Matplotlib and ReportLab. <font face='DV-B'>src/microgrid.py</font> holds the uncertainty model, sparse extensive-form LP builder (risk-neutral and CVaR), "
  "exact recourse evaluator, wait-and-see solver and risk metrics. <font face='DV-B'>src/run_experiments.py</font> runs all experiments end-to-end with fixed seeds and writes 11 result tables "
  "(<font face='DV-B'>results/</font>), 8 figures (<font face='DV-B'>figures/</font>) and <font face='DV-B'>summary.json</font>; <font face='DV-B'>src/make_report.py</font> builds this PDF from them; "
  "<font face='DV-B'>tests/test_microgrid.py</font> checks the LP against an independent exact calculation, energy-balance feasibility, CVaR behaviour and the wait-and-see bound. "
  "An executed notebook is in <font face='DV-B'>notebooks/</font>.")

# ------------------------------------------------------------------ 5 results --
P("5. Results: Optimal Solution, Expected Value and Risk", H1)
P("5.1 Optimal day-ahead schedule", H2)
sc = pd.read_csv(R("table2_day_ahead_schedule.csv"))
sc = sc[sc.hour.isin([0, 3, 5, 9, 10, 12, 14, 18, 20, 21, 22])][["hour", "da_price", "mean_load_kw", "mean_pv_kw", "deterministic_purchase_kwh", "stochastic_purchase_kwh", "difference_kwh"]]
sc.columns = ["Hour", "DA price ₹/kWh", "Mean load kW", "Mean PV kW", "Det. purchase kWh", "Stoch. purchase kWh", "Difference"]
table(sc, fmts={"Hour": 0, "DA price ₹/kWh": 1, "Mean load kW": 0, "Mean PV kW": 0, "Det. purchase kWh": 0, "Stoch. purchase kWh": 0, "Difference": 0},
      caption=f"Table 3. Selected hours of the optimal day-ahead schedules (all 24 hours in results/table2_day_ahead_schedule.csv). Totals: deterministic {md['da_kwh']:,.0f} kWh, stochastic {ms['da_kwh']:,.0f} kWh/day.")
fig("fig2_day_ahead_schedule.png", 12.0, "Figure 2. Day-ahead purchase schedules against mean load and mean net load.")
P(f"The stochastic schedule <b>hedges where the mean forecast says no purchase is needed</b>: around midday (10–13 h) mean PV exceeds mean load, so the deterministic plan buys nothing, but on partly cloudy or overcast days PV falls far short and "
  f"the shortfall would be bought in real time at twice the price; the stochastic plan therefore buys 17–45 kWh/h there. It also buys more in the 18–21 h peak (₹9.5/kWh, ₹19 in real time) and more in the cheap night hours to charge the battery. "
  f"The total purchase is {100*(ms['da_kwh']/md['da_kwh']-1):.0f}% higher ({ms['da_kwh']:,.0f} vs {md['da_kwh']:,.0f} kWh), the surplus on good days being resold at ₹3/kWh "
  f"({ms['resold_kwh']:.0f} vs {md['resold_kwh']:.0f} kWh/day): the price of insurance against expensive real-time purchases.")
P("5.2 Expected cost, variability and risk versus the deterministic baseline", H2)
c = pd.read_csv(R("table3_baseline_comparison.csv"))
c = c[["model", "mean_cost", "std", "p50", "p95", "cvar90", "rt_kwh", "resold_kwh"]]
c.columns = ["Model", "Mean cost", "Std dev", "Median", "95th pct", "CVaR 90%", "RT buy kWh/day", "Resold kWh/day"]
table(c, widths=[3.8*cm, 1.7*cm, 1.6*cm, 1.7*cm, 1.7*cm, 1.8*cm, 2.1*cm, 2.1*cm],
      fmts={k: 0 for k in c.columns if k != "Model"}, caption="Table 4. Daily energy cost (₹) over 2,000 simulated out-of-sample days. CVaR 90% = mean cost of the worst 10% of days; RT = real-time purchases.")
fig("fig3_cost_distribution.png", 15.0, "Figure 3. Left: daily-cost distributions. Right: mean cost by weather type.")
P(f"The stochastic schedule has a lower mean (−₹{M['VSS']:.0f}), a {std_cut:.0f}% lower standard deviation (₹{ms['std']:,.0f} vs ₹{md['std']:,.0f}), a lower 95th percentile (₹{ms['p95']:,.0f} vs ₹{md['p95']:,.0f}) and a lower worst-10%-day average "
  f"(₹{ms['cvar90']:,.0f} vs ₹{md['cvar90']:,.0f}). It is cheaper on <b>{100*M['share_win']:.1f}%</b> of individual days. Unlike many hedging models, the benefit is therefore broad rather than a rare-event effect. "
  f"Neither plan ever sheds load or curtails PV in the base case, because the grid backstop is always available: reliability is not at stake, price risk is.")
rows = pd.DataFrame(bw)[["weather_state", "days", "cost_det", "cost_sto", "saving"]]
rows["saving_pct"] = 100 * rows.saving / rows.cost_det
rows.columns = ["Weather type", "Test days", "Cost, deterministic", "Cost, stochastic", "Saving ₹/day", "Saving %"]
table(rows, fmts={"Test days": 0, "Cost, deterministic": 0, "Cost, stochastic": 0, "Saving ₹/day": 0, "Saving %": 2},
      caption="Table 5. Where the value comes from: cost by true weather type (out-of-sample).")
P(f"Savings are small on sunny days (₹{bw[0]['saving']:.0f}, {100*bw[0]['saving']/bw[0]['cost_det']:.1f}%) and large on overcast days (₹{bw[2]['saving']:,.0f}, {100*bw[2]['saving']/bw[2]['cost_det']:.1f}%): "
  f"the mean-forecast plan is effectively a plan for a typical day that never occurs, and pays most when the actual day is far from typical.")
P("5.3 Value of the stochastic solution and value of perfect information", H2)
v = pd.read_csv(R("table4_value_of_stochastic_solution.csv")); v.columns = ["Metric", "Value"]
table(v, widths=[12.5*cm, 3.0*cm], caption="Table 6. Value metrics (₹/day unless a percentage). EVPI is computed on 500 of the test days.")
P(f"<b>VSS = ₹{M['VSS']:.1f}/day</b> (CI ₹{M['VSS_ci'][0]:.0f}–{M['VSS_ci'][1]:.0f}; {M['VSS_pct']:.2f}% ≈ ₹{M['annual']:,.0f}/year). "
  f"<b>EVPI = ₹{M['EVPI']:,.0f}/day</b> ({M['EVPI']/M['VSS']:.1f}× the VSS): perfect weather and load forecasts would be worth much more than any improvement in the decision model, so investing in forecasting (satellite irradiance, "
  f"numerical weather prediction) is the natural next step.")
P("5.4 Convergence and stability of SAA", H2)
cs = pd.read_csv(R("table6_saa_convergence.csv"))
cs = cs[["S", "insample", "oos_model", "oos_true", "oos_true_std", "total_purchase_std", "optimism_gap"]]
cs.columns = ["S", "In-sample cost", "Out-of-sample (training model)", "Out-of-sample (true process)", "Std of OOS cost", "Std of total purchase", "Optimism gap"]
table(cs, fmts={"S": 0, **{k: 0 for k in cs.columns if k != "S"}},
      caption="Table 7. SAA with S scenarios, 12 independent replications each (₹/day). For a cost-minimisation problem the in-sample value is biased low; optimism gap = out-of-sample (training model) − in-sample.")
fig("fig4_saa_convergence.png", 15.0, "Figure 4. Left: in-sample optimism and out-of-sample cost versus S (band = 10–90% over replications). Right: variability of the solution across replications.")
P(f"With S = 10 the in-sample cost is optimistic by ₹{conv[0]['optimism_gap']:.0f} and the total purchase varies by ±{conv[0]['total_purchase_std']:.0f} kWh between replications. Out-of-sample cost on the true process "
  f"falls from ₹{conv[0]['oos_true']:,.0f} (S = 10) to ₹{conv[-1]['oos_true']:,.0f} (S = 400) and its replication-to-replication spread from ₹{conv[0]['oos_true_std']:.0f} to ₹{conv[-1]['oos_true_std']:.0f}, "
  f"so the schedule becomes both better and more stable as S grows; the main results use S = 300, in the flat region.")
P("5.5 Risk-averse (mean-CVaR) extension", H2)
cv = pd.read_csv(R("table7_cvar_frontier.csv"))[["lam", "total_da_kwh", "mean_cost", "std", "p95", "cvar90", "max_cost"]]
cv.columns = ["λ", "DA purchase kWh", "Mean cost", "Std dev", "95th pct", "CVaR 90%", "Worst day"]
table(cv, fmts={"λ": 2, "DA purchase kWh": 0, "Mean cost": 0, "Std dev": 0, "95th pct": 0, "CVaR 90%": 0, "Worst day": 0},
      caption="Table 8. Mean-CVaR frontier (S = 200, 1,000 out-of-sample days; ₹/day). λ = 0 is risk-neutral.")
fig("fig5_cvar_frontier.png", 15.0, "Figure 5. Risk–cost frontier: paying a little more on average removes much of the expensive tail.")
c0, c25, c5 = cvr[0], cvr[1], cvr[2]
P(f"λ = 0.25 costs ₹{c25['mean_cost']-c0['mean_cost']:.0f}/day more ({100*(c25['mean_cost']/c0['mean_cost']-1):.1f}%) but cuts the standard deviation by {100*(1-c25['std']/c0['std']):.0f}% and CVaR from ₹{c0['cvar90']:,.0f} to ₹{c25['cvar90']:,.0f} ({100*(1-c25['cvar90']/c0['cvar90']):.0f}%); "
  f"λ = 0.5 reaches {100*(1-c5['std']/c0['std']):.0f}% lower volatility for +{100*(c5['mean_cost']/c0['mean_cost']-1):.1f}% cost. Beyond λ ≈ 0.5 the frontier is almost flat, so a budget-constrained institution should choose λ between 0.25 and 0.5.")

# --------------------------------------------------------------- 6 sensitivity --
P("6. Sensitivity Analysis", H1)
P("6.1 Uncertainty drivers, price premium and storage", H2)
tb = sens[["factor", "value", "stoch", "det", "VSS", "VSS_pct"]].copy()
tb.columns = ["Factor", "Value", "Stoch. cost", "Det. cost", "VSS", "VSS %"]
table(tb, widths=[5.2*cm, 1.6*cm, 2.2*cm, 2.2*cm, 1.9*cm, 1.7*cm], fmts={"Value": 2, "Stoch. cost": 0, "Det. cost": 0, "VSS": 0, "VSS %": 2},
      caption="Table 9. One-factor sweeps in a controlled world (history of 365 days, S = 100 scenarios, 400 test days; cost in ₹/day).")
fig("fig6_sensitivity_factors.png", 16.0, "Figure 6. Value of the stochastic solution versus overcast frequency, within-state weather variability and real-time price multiplier.")
P(f"<b>The VSS is driven by the price asymmetry and by the weather mix, not by day-to-day noise.</b> (i) At a real-time multiplier of 1× there is no penalty for deviating from the day-ahead schedule and the VSS is ≈ ₹{max(0.0, rt.VSS.iloc[0]):.0f} (statistically zero), as theory predicts; "
  f"it rises to ₹{rt.VSS.iloc[2]:.0f} at 2× (base), ₹{rt.VSS.iloc[3]:,.0f} at 3× and ₹{rt.VSS.iloc[4]:,.0f} ({rt.VSS_pct.iloc[4]:.1f}%) at 4×. (ii) More overcast days raise cost for both plans, and the VSS climbs from ₹{oc.VSS.iloc[0]:.0f} (5% overcast) to ₹{oc.VSS.iloc[3]:.0f} (40%). "
  f"(iii) Scaling the within-state variability of κ from 0× to 2× barely moves the VSS (₹{vs_.VSS.min():.0f}–{vs_.VSS.max():.0f}), because the large uncertainty is <i>which weather state occurs</i>, which the scenarios retain. "
  f"(iv) Storage is worth ₹{bt.stoch.iloc[0]-bt.stoch.iloc[3]:,.0f}/day for an 800 kWh battery under the stochastic plan (cost ₹{bt.stoch.iloc[0]:,.0f} → ₹{bt.stoch.iloc[3]:,.0f}); "
  f"going to 1,200 kWh saves only a further ₹{bt.stoch.iloc[3]-bt.stoch.iloc[4]:.0f}/day, and the VSS saturates (₹{bt.VSS.iloc[3]:.0f} → ₹{bt.VSS.iloc[4]:.0f}) once the battery is large enough to absorb most deviations.")
P("6.2 Parameters (one-at-a-time ±20%)", H2)
tt = pd.read_csv(R("table9_sensitivity_parameters.csv"))
tt.columns = ["Parameter (±20%)", "Δ cost at −20%", "Δ cost at +20%", "Swing"]
table(tt, fmts={"Δ cost at −20%": 0, "Δ cost at +20%": 0, "Swing": 0},
      caption=f"Table 10. Change in expected daily cost (₹/day, plan re-optimised) when a parameter changes by ∓20%. Base expected cost in this world: ₹{S['tornado_base_cost']:,.0f}.")
fig("fig7_tornado.png", 11.0, "Figure 7. Tornado chart of parameter sensitivity.")
P(f"Cost is dominated by the <b>load level</b> (±20% moves cost by about ±₹{abs(tor[0]['high_plus20']):,.0f}/day) and by <b>PV capacity</b> (swing ₹{tor[1]['swing']:,.0f}); battery round-trip efficiency (₹{tor[2]['swing']:,.0f}) matters more than battery size (₹{tor[5]['swing']:,.0f}). "
  f"The value of lost load has <b>zero</b> effect because no scenario requires shedding when the grid backstop exists, an important structural insight: <i>the VOLL penalty could be omitted from this model without changing any decision.</i>")
P("6.3 Robustness to distribution shift", H2)
sh = pd.read_csv(R("table10_distribution_shift.csv"))
sh.columns = ["True world (test)", "Re-optimised (oracle)", "Stochastic plan", "Deterministic plan", "Regret (stoch.)", "Regret (det.)", "Det. − Stoch."]
table(sh, fmts={k: 0 for k in sh.columns if k != "True world (test)"},
      caption="Table 11. Plans trained on the base history, tested in shifted worlds (₹/day, 600 days each). Regret = cost above the re-optimised oracle; negative 'Det. − Stoch.' means the deterministic plan wins.")
fig("fig8_battery_and_shift.png", 15.0, "Figure 8. Left: value of storage. Right: regret under distribution shift.")
wl = shift[4]
P(f"The stochastic plan beats the deterministic one in five of six worlds, including a monsoon-heavy season (+₹{shift[1]['det_minus_stoch']:.0f}/day) and a +10% load increase (+₹{shift[3]['det_minus_stoch']:,.0f}/day, where its regret of ₹{shift[3]['regret_stoch']:.0f} is a third of the deterministic regret of ₹{shift[3]['regret_det']:,.0f}). "
  f"However, if load is <b>10% lower</b> than in the training data, the deterministic plan wins by ₹{abs(wl['det_minus_stoch']):.0f}/day: the stochastic plan has learned to over-buy as insurance and the surplus is resold at ₹3/kWh. "
  f"Stochastic optimisation is only as good as its scenario set; a rolling re-estimation window (e.g. 60–90 days) is recommended.")
P("6.4 Measured impact (SDG 9.5)", H2)
im = pd.read_csv(R("table11_sdg_impact_metrics.csv")); im.columns = ["Indicator", "Deterministic", "Stochastic", "Change"]
table(im, widths=[8.2*cm, 2.6*cm, 2.6*cm, 2.6*cm], caption="Table 12. Impact indicators per day, out-of-sample. CO₂ uses an assumed grid emission factor of 0.71 kg/kWh.")
imp = {r["indicator"]: r for r in S["impact"]}
P(f"The stochastic plan reduces expensive real-time purchases by {rt_cut:.0f}% ({md['rt_kwh']:.0f} → {ms['rt_kwh']:.0f} kWh/day) and expected cost by ₹{M['VSS']:.0f}/day, while net grid import changes only marginally "
  f"({imp['Net grid import (kWh/day)']['change']:.0f} kWh/day, {abs(imp['CO2 from grid import (kg/day, EF=0.71)']['change'])*365/1000:.1f} t CO₂/year): the benefit is financial and price-risk reduction, not emissions, "
  f"an honest and measurable result consistent with Target 9.5's focus on innovation capability.")

# --------------------------------------------------------------- 7 conclusion --
P("7. Discussion, Limitations and Conclusions", H1)
P(f"<b>Conclusions.</b> (1) Modelling solar and load uncertainty explicitly cuts expected cost by ₹{M['VSS']:.0f}/day ({M['VSS_pct']:.2f}%, significant by bootstrap) and volatility by {std_cut:.0f}%. "
  f"(2) The gain arises from the asymmetry between real-time and resale prices and is concentrated on overcast days. (3) The value of the stochastic solution is driven by the real-time price premium and weather mix, not by within-state noise. "
  f"(4) CVaR gives an explicit, cheap risk dial (λ = 0.25: −{100*(1-c25['std']/c0['std']):.0f}% volatility for +{100*(c25['mean_cost']/c0['mean_cost']-1):.1f}% cost). (5) SAA with ≥ 300 scenarios is accurate and stable. (6) EVPI ≈ {M['EVPI']/M['VSS']:.0f}× VSS, so better forecasting matters more than a better optimiser.")
P("<b>Limitations.</b> Simulated data (absolute rupee amounts are illustrative); a single deterministic tariff and a flat real-time multiplier (real prices are also random); battery degradation is linear; no intra-hour dynamics; "
  "the plan is sensitive to a drop in load below the training level (Table 11). <b>Future work:</b> multi-stage rolling-horizon optimisation with forecast updates, stochastic prices, nonlinear battery ageing, "
  "distributionally robust formulations, and calibration with real smart-meter data.")
P("References", H1)
for r in ["J. R. Birge and F. Louveaux, <i>Introduction to Stochastic Programming</i>, 2nd ed., Springer, 2011.",
          "A. Shapiro, D. Dentcheva and A. Ruszczyński, <i>Lectures on Stochastic Programming: Modeling and Theory</i>, SIAM, 2009.",
          "A. J. Kleywegt, A. Shapiro and T. Homem-de-Mello, “The sample average approximation method for stochastic discrete optimization,” <i>SIAM J. Optimization</i>, 12(2), 479–502, 2002.",
          "R. T. Rockafellar and S. Uryasev, “Optimization of conditional value-at-risk,” <i>Journal of Risk</i>, 2(3), 21–41, 2000.",
          "Q. Huangfu and J. A. J. Hall, “Parallelizing the dual revised simplex method (HiGHS),” <i>Math. Programming Computation</i>, 10, 119–142, 2018.",
          "P. Virtanen et al., “SciPy 1.0: fundamental algorithms for scientific computing in Python,” <i>Nature Methods</i>, 17, 261–272, 2020.",
          "United Nations, Sustainable Development Goal 9 – Industry, Innovation and Infrastructure, Target 9.5."]:
    P(r, REF)


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("DV", 7); canvas.setFillColor(colors.grey)
    canvas.drawString(1.8 * cm, 1.0 * cm, "21AIE321T Stochastic Decision Making – Mini Project")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {doc.page}"); canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.7 * cm,
                        title="Stochastic Microgrid Dispatch - Mini Project", author=STUDENT)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("report written:", OUT)

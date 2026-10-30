# Stochastic Day-Ahead Energy Planning for a Solar–Battery Campus Microgrid

**Course:** 21AIE321T – Stochastic Decision Making · **Assignment IV (CLA-2) Mini Project**
**Institute:** SRM Institute of Science and Technology, Tiruchirappalli · B.Tech AI, Semester V · AY 2026-27
**SDG:** 9 – Industry, Innovation and Infrastructure (Target 9.5) · **CO4** · Level K4 (Analyse)

## Problem
A campus microgrid (600 kWp rooftop PV, 800 kWh battery, 450 kW grid connection) must submit **hourly day-ahead electricity
purchases before tomorrow's sunshine and load are known**. Any mismatch is settled in real time: shortfalls are bought at
**2× the day-ahead price**, surpluses are resold at only ₹3/kWh. The battery, grid limit and a value of lost load complete the model.

## Approach
| Component | Method |
|---|---|
| Model | **Two-stage stochastic linear programme** with a 24-hour battery model as recourse |
| Uncertainty | Weather-state mixture (sunny / partly cloudy / overcast) → hourly PV with AR(1) cloud noise; temperature-correlated load. 730-day history is simulated (seeded) |
| Solution method | **Sample Average Approximation (SAA)**, smoothed bootstrap of historical days, extensive-form LP (50,424 variables at S = 300) solved exactly with HiGHS |
| Risk | **Mean-CVaR** (Rockafellar–Uryasev), risk-aversion weight λ |
| Baseline | Deterministic **mean-forecast** plan; metrics **VSS** and **EVPI** |
| Validation | 2,000 fresh out-of-sample days from the *true* process, exact recourse LP per day, bootstrap CI for VSS |
| Sensitivity | Overcast frequency, weather variability, real-time price multiplier, battery size, ±20 % on 9 parameters, distribution shift, SAA sample size |

## Headline results (all produced by the code in this repo)
| Metric | Deterministic | Stochastic (SAA) |
|---|---|---|
| Expected daily energy cost | ₹34,043 | **₹33,548** |
| Std. dev. of daily cost | ₹7,017 | **₹6,249** |
| Worst-10 %-day average cost (CVaR) | ₹49,184 | **₹47,641** |
| Real-time purchases | 405 kWh/day | **250 kWh/day (−38 %)** |

* **VSS = ₹494.9/day** (95 % CI ₹450–541, +1.45 %, ≈ ₹1.8 lakh/year); the stochastic plan is cheaper on 63 % of days.
* Gains concentrate on **overcast days (₹1,549/day)**; VSS ≈ 0 with no real-time price premium and reaches 10 % at a 4× premium.
* **EVPI = ₹3,281/day** (6.6× VSS) → better forecasting is worth more than a better optimiser.
* Honest limitation: if load falls 10 % below the training level, the deterministic plan wins by ₹438/day (report §6.3).

Exact tables are in `results/`; the narrative is in `report/Stochastic_Microgrid_Mini_Project_Report.pdf`.

## Repository layout
```
.
├── README.md  ├── requirements.txt  ├── LICENSE  └── .gitignore
├── data/
│   ├── microgrid_history_hourly.csv      # 730 days x 24 h of PV (kW) and load (kW)
│   └── daily_summary.csv                 # per-day weather label, PV and load energy
├── src/
│   ├── microgrid.py                      # uncertainty model, SAA / CVaR LP, recourse evaluator, wait-and-see
│   ├── run_experiments.py                # runs every experiment -> results/, figures/
│   └── make_report.py                    # builds the PDF report from results/
├── tests/test_microgrid.py               # 5 correctness tests
├── notebooks/stochastic_microgrid_analysis.ipynb   # executed walkthrough with outputs
├── results/                              # 11 result tables (CSV), raw replications, summary.json
├── figures/                              # 8 figures (PNG)
└── report/Stochastic_Microgrid_Mini_Project_Report.pdf
```

## How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.run_experiments        # ~8 min on one core: regenerates data/, results/, figures/
python tests/test_microgrid.py       # or: python -m pytest tests
STUDENT_NAME="Your Name (RegNo)" python -m src.make_report   # rebuilds the PDF with your name
jupyter notebook notebooks/stochastic_microgrid_analysis.ipynb
```
All random streams are seeded, so every number is reproducible.

## Using real data
Replace `data/microgrid_history_hourly.csv` (columns `date, hour, pv_kw, load_kw`) with smart-meter and inverter data and edit
`System` / tariff arrays in `src/microgrid.py`. Fitting, optimisation and reporting are unchanged.

## Data disclosure
No campus meter data were available. The history is **simulated** from a documented process (`WeatherSpec`); absolute rupee
amounts are illustrative, while the methodology and relative conclusions are what the project demonstrates.

## References
Birge & Louveaux (2011); Shapiro, Dentcheva & Ruszczyński (2009); Kleywegt, Shapiro & Homem-de-Mello (2002) SIAM J. Optim. 12(2);
Rockafellar & Uryasev (2000) J. Risk 2(3); Huangfu & Hall (2018) HiGHS; Virtanen et al. (2020) SciPy.

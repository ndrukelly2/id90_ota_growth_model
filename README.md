# ID90 Growth & Revenue Simulator

This repo simulates weekly active users, bookers, product purchases, and total take (revenue) for two user groups (`partners`, `non_partners`) across four LOBs (`flights`, `hotels`, `cars`, `cruises`). It supports calibration, cohort‑based early retention, mature churn, product mix tuning, and scenario deltas vs a baseline.

* **Entrypoint:** `python run.py`
* **Config:** `config.yaml`
* **Output:** `simulation_output.csv` + console summary

---

## Quick Start

1. Ensure Python 3.x and dependencies (pandas, numpy, pyyaml).
2. Put your inputs in **`config.yaml`** (examples below).
3. Run:

   ```bash
   python run.py
   ```
4. Open **`simulation_output.csv`** and chart whatever you need (e.g., weekly take, bookings, churn).

---

## Concepts & Flow (High-level)

Each simulated week *t*:

1. **Active WAU** = mature pool + onboarding presence (weeks 0–12) driven by cohort presence `W_k`.
2. **Bookers** by group = WAU × weekly booker rate (`p_book[g]`).
3. **Bookings by LOB** = bookers × `(purchases_per_booker_per_year[g][lob] / 52)` (flights suppressed if gate is off).
4. **Take** by LOB using LOB economics (flights = PNRS × take\_per\_PNR; others = bookings × take\_per\_booking).
5. **Graduate** onboarding age‑12 to mature with `p_grad`, then **apply mature churn** to the mature pool.

---

## Configuration (what goes in `config.yaml`)

Below is a complete reference with examples. You don’t need every section—use what’s relevant.

### Simulation duration

```yaml
weeks: 52   # number of weeks to simulate
```

### Initial active users (WAU)

```yaml
initial_wau_by_group:
  partners: 150000
  non_partners: 150000
```

If omitted, the loader can split `initial_wau_total` by share:

```yaml
initial_wau_total: 300000
user_groups:
  active_user_share:
    partners: 0.5
    non_partners: 0.5
```

---

### New accounts (inflow)

There are **three** possible sources; they’re checked in this order:

1. **`historical_series.weekly_new_accounts_by_group` — overrides everything**
2. `new_accounts.mode: series` → `new_accounts.series_by_group`
3. `new_accounts.mode: baseline_growth` → parametric baseline + weekly growth

**Important precedence rule:**
If `historical_series.weekly_new_accounts_by_group` is present, it **wins**, and `new_accounts.mode` is **ignored**.

#### Option A — Use a week‑by‑week series

```yaml
new_accounts:
  mode: series
  series_by_group:
    partners:      [1200, 1180, 1235, ...]          # length ≥ weeks (extra is truncated)
    non_partners:  [2200, 2190, 2250, ...]
```

> If you also have the top‑level `historical_series.weekly_new_accounts_by_group`, delete it or it will override this series.

#### Option B — Use parametric baseline growth

```yaml
new_accounts:
  mode: baseline_growth
  baseline_per_week_by_group:
    partners: 1200
    non_partners: 2200
  growth_rate_weekly_by_group:
    partners: 0.0070       # ≈ +0.70% per week
    non_partners: 0.0070
```

#### (Overrides) Top‑level historical series

```yaml
historical_series:
  weekly_new_accounts_by_group:
    partners:      [ ... ]
    non_partners:  [ ... ]
```

> If present, this is used **regardless** of `new_accounts.mode`.

---

### Early‑life retention (onboarding presence 0–12 weeks)

You can provide a **cohort CSV** to derive weekly presence `W_k` and the **graduation probability** `p_grad`:

```yaml
cohort_csv_path: path/to/your_cohort.csv
retention_hazard_scale: 1.0   # >1.0 boosts daily presence; <1.0 reduces
```

**Accepted CSV formats:**

* **Long form:** columns `day` (0..90) and `p_active_day` (0..1).
* **Matrix form:** a non‑numeric id column (e.g. `cohort_date`) followed by numeric day offset columns `"0"`..`"90"`. Each row is normalized by day‑0 and averaged across cohorts.

> The **hazard transform** `p' = 1 − (1 − p)^scale` is applied **only** when a **cohort CSV** is used.

If you **don’t** provide a CSV (or retention points), the model falls back to default `W_k` and `p_grad` (a simple shaped presence; default `p_grad = 0.02`).

> There is also a `retention_points` option if you prefer to hand‑specify shape; if used, CSV is ignored. Hazard scale is not applied in that path.

---

### Mature churn (post‑90d)

Provide either **weekly** rates (preferred) or **monthly** rates (we convert to weekly via compounding). Then you can scale per group.

```yaml
# Preferred direct weekly input
estimated_churn_from_12m_inactivity:
  weekly_mature_churn_rate_by_group:
    partners: 0.000576   # e.g., ≈0.0576% weekly
    non_partners: 0.000691

# Or, monthly rates (converted to weekly)
monthly_mature_churn_rate_by_group:
  partners: 0.0025
  non_partners: 0.0030

# Scenario knob (NEW): per‑group mature churn scaling
mature_churn_scale_by_group:
  partners: 1.00      # 0.90 => reduce partners churn 10%
  non_partners: 1.00  # 1.10 => increase non‑partners churn 10%
```

---

### Booker conversion & calibration

You can run with a fixed weekly booker rate per group, or let the simulator **calibrate** to a target.

```yaml
# Use calibration or provide explicit rates
calibration_mode: target_annual_take   # one of: none | target_annual_take | observed_bookings
target_annual_take_usd: 22359972.58
calibration_shared_rate_only: true     # calibrates a single shared weekly rate for both groups
calibration_max_weekly_booker_rate: 1.0
weeks_to_use: 52                       # optional: limit calibration to first N weeks

# If calibration_mode: none, set rates explicitly:
booker_rate_weekly_by_group:
  partners: 0.012
  non_partners: 0.010

# (Optional) Observed bookings mode expects a CSV and schema you define; if provided:
bookings_csv_path: path/to/bookings.csv
```

> The `target_annual_take` calibration uses linearity: it runs a unit simulation and scales a **shared** weekly booker rate (bounded by `calibration_max_weekly_booker_rate`) to hit your target.

---

### Product mix & LOB economics

Base **purchases per booker per year** by group and LOB:

```yaml
purchases_per_booker_per_year_by_lob_by_group:
  partners:
    flights: 2.1      # PNRS/booker/year (flights)
    hotels:  1.0      # BOOKINGS/booker/year (hotels)
    cars:    0.2
    cruises: 0.05
  non_partners:
    flights: 1.9
    hotels:  0.9
    cars:    0.2
    cruises: 0.05
```

**Product‑mix scaler (NEW):** multiply those frequencies to test mix shifts.

```yaml
product_mix_scale_by_lob_by_group:
  flights:                # single number or per‑group mapping
    partners: 1.10        # +10% PNRS/booker for partners
    non_partners: 1.00
  hotels: 1.05            # +5% bookings/booker for both groups
  cruises: 0.90           # -10%
  cars: 1.00
```

**LOB economics:**

```yaml
lobs:
  flights:
    take_per_pnr_usd: 4.75
  hotels:
    take_per_booking_usd: 45.00
  cars:
    take_per_booking_usd: 5.50
  cruises:
    take_per_booking_usd: 200.00
```

**Flights gate:** suppress flights demand (no redistribution) for a group.

```yaml
flights_gate:
  partners: true
  non_partners: false
```

---

### Baseline (for quick scenario deltas)

Add a **baseline** to print deltas in the console after each run:

```yaml
baseline_total_take_usd: 22359972.58
```

Console will show:

```
Total take_usd_total: 22,423,100.00
Vs baseline $22,359,972.58: Δ $63,127.42 (+0.28%)
```

---

## Output schema (CSV)

`simulation_output.csv` includes:

* `week`
* **Active & bookers**

  * `active_partners`, `active_non_partners`
  * `bookers_partners`, `bookers_non_partners`
* **Bookings by LOB**

  * `bookings_flights`, `bookings_hotels`, `bookings_cars`, `bookings_cruises`
* **Take by LOB and total**

  * `take_usd_flights`, `take_usd_hotels`, `take_usd_cars`, `take_usd_cruises`
  * `bookings_total`, `take_usd_total`
* **New accounts (inflow)**

  * `new_accounts_partners`, `new_accounts_non_partners`
* **(NEW) Weekly churn**

  * `churn_rate_partners`, `churn_rate_non_partners`  *(weekly mature churn rates applied)*
  * `churned_partners`, `churned_non_partners`, `churned_total`
    *(counts, computed on mature pool **after** this week’s graduates join)*

> The CSV uses one row per week. Column order is stable; if you remove the explicit list in `run.py` it will follow insertion order from the simulator.

---

## Running & Console Summary

`python run.py` will:

* Generate `simulation_output.csv`
* Print a summary:

  * Sum of key columns (pandas `describe`/`sum` flavor)
  * Estimated/calibrated weekly booker rate(s)
  * **Total take** and **Δ vs baseline** (if baseline provided)
  * **Avg new users/week**
  * **Avg churned users/week** and **avg weekly mature churn rate** by group

---

## Modeling Tips

* **New accounts precedence:** If you want to drive scenarios with a **series**, put the data under `new_accounts.series_by_group` and remove `historical_series.weekly_new_accounts_by_group`. If you want a **smooth increase**, use `baseline_growth`. Don’t keep both—`historical_series` overrides.
* **Retention hazard scale:** Only applied when a **cohort CSV** is used. It warps daily presence as `1 − (1 − p)^scale` before weekly union presence is computed. It does **not** apply in the default `W_k` fallback.
* **Flights gate:** When `false`, we **suppress** flights demand for that group with no redistribution to other LOBs (deliberately conservative).
* **Mature churn scale:** Use `mature_churn_scale_by_group` to test scenario reductions/increases to churn **after** you supply the baseline weekly/monthly rates.
* **Product mix:** `product_mix_scale_by_lob_by_group` multiplies your **per‑booker frequencies**. If you want a separate “share of bookers” control vs “orders per purchasing booker,” we can add it, but in practice the single scaler is usually sufficient for scenario planning.

---

## Minimal Example `config.yaml`

```yaml
weeks: 52

initial_wau_by_group:
  partners: 150000
  non_partners: 150000

# New accounts (pick ONE source)
new_accounts:
  mode: baseline_growth
  baseline_per_week_by_group:
    partners: 1200
    non_partners: 2200
  growth_rate_weekly_by_group:
    partners: 0.0070
    non_partners: 0.0070
# historical_series:
#   weekly_new_accounts_by_group: {partners: [...], non_partners: [...]}
# new_accounts:
#   mode: series
#   series_by_group: {partners: [...], non_partners: [...]}

# Early retention (optional)
# cohort_csv_path: path/to/cohort.csv
retention_hazard_scale: 1.0

# Mature churn
estimated_churn_from_12m_inactivity:
  weekly_mature_churn_rate_by_group:
    partners: 0.000576
    non_partners: 0.000691
mature_churn_scale_by_group:
  partners: 1.0
  non_partners: 1.0

# Calibration
calibration_mode: target_annual_take   # none | target_annual_take | observed_bookings
target_annual_take_usd: 22359972.58
calibration_shared_rate_only: true
calibration_max_weekly_booker_rate: 1.0
weeks_to_use: 52

# Purchases per booker per year (base)
purchases_per_booker_per_year_by_lob_by_group:
  partners:      {flights: 2.1, hotels: 1.0, cars: 0.2, cruises: 0.05}
  non_partners:  {flights: 1.9, hotels: 0.9, cars: 0.2, cruises: 0.05}

# Product mix scaler (NEW)
product_mix_scale_by_lob_by_group:
  flights: {partners: 1.10, non_partners: 1.00}
  hotels: 1.05
  cruises: 0.90
  cars: 1.00

# LOB economics
lobs:
  flights: {take_per_pnr_usd: 4.75}
  hotels:  {take_per_booking_usd: 45.00}
  cars:    {take_per_booking_usd: 5.50}
  cruises: {take_per_booking_usd: 200.00}

# Flights gate
flights_gate:
  partners: true
  non_partners: false

# Baseline for % delta printout (NEW)
baseline_total_take_usd: 22359972.58
```

---

## Changelog

* **\[NEW] Weekly churn in CSV:**
  `churn_rate_partners`, `churn_rate_non_partners`, `churned_partners`, `churned_non_partners`, `churned_total`.
* **\[NEW] Per‑group mature churn scale:**
  `mature_churn_scale_by_group` multiplies computed weekly churn per group.
* **\[NEW] Product mix scaler:**
  `product_mix_scale_by_lob_by_group` multiplies per‑booker frequencies by LOB (per‑group optional).
* **\[NEW] Baseline deltas:**
  `baseline_total_take_usd` prints Δ and % vs baseline in console.
* Documentation clarified for **new accounts precedence** and **cohort hazard scale** behavior.

---

## Troubleshooting

* **My hazard scale doesn’t change results.**
  Ensure `cohort_csv_path` points to a real CSV in one of the supported formats. Hazard only applies in the cohort‑CSV path.
* **Changing `new_accounts.mode` does nothing.**
  You probably still have `historical_series.weekly_new_accounts_by_group` present. Remove it or it will override.
* **Flights scaling has no effect for non‑partners.**
  Check `flights_gate.non_partners`. If it’s `false`, flights demand is suppressed for that group.



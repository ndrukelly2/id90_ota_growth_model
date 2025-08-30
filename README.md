# Weekly Travel Finance Simulator (Active → Bookers → Bookings → Take)

This repo contains a lean, data‑anchored simulator that projects from **weekly active users (WAU)** → **bookers** → **bookings by line of business (LOB)** → **take (\$)**. It models **partners** and **non‑partners** separately and enforces a **flights gate** (non‑partners cannot book flights, by design).

---

## How it works (exact to code)

Each simulated **week *t*** does the following for each group `g ∈ {partners, non_partners}`:

1. **Compute WAU (active)**
   `active[g,t] = mature_active[g,t] + Σ_{k=0..12} onboarding_age[g,k,t] × W[k]`

* `W[k]` are weekly “seen at least once” presence weights for onboarding ages **0..12 weeks** derived from (in order of precedence):

  * a cohort CSV, or
  * four **retention\_points** anchors (days 1/7/30/90), or
  * defaults (see Config Reference).
* A **graduation fraction `p_grad`** (the mean of the last four `W` entries) moves age‑12 onboarding into the mature pool at the end of the week.

2. **Bookers**
   `bookers[g,t] = active[g,t] × p_book[g]`
   `p_book[g]` is either provided or **calibrated** (see Calibration).

3. **Expected bookings by LOB**
   For each LOB ℓ in `{flights, hotels, cars, cruises}`:
   `bookings_ℓ[t] += Σ_g bookers[g,t] × (purchases_per_booker_per_year[g,ℓ] / 52)`

* **Flights gate**: if `flights_gate[g] == false`, that group’s flight demand is **suppressed** (not redistributed).

4. **Take (\$)**
   Flights use `take_per_pnr_usd`. Other LOBs use `take_per_booking_usd`. Totals are summed per week and across the run.

5. **Transition to t+1**

* **Graduation**: `graduates = onboarding_age[g,12,t] × p_grad`; add to `mature_active`.
* **Mature churn**: `mature_active[g,t+1] = (mature_active[g,t] + graduates) × (1 − weekly_churn[g])`.
* **Age onboarding**: shift ages 0→12; set new age‑0 to next week’s **new\_accounts\[g,t]**.

Initial state puts all `initial_wau_by_group` into the mature pool; onboarding ages start at 0.

---

## Calibration

Two modes are implemented; choose in `config.yaml → calibration.mode`:

* **`observed_bookings`** (regression):

  1. Run a unit‑rate simulation (`p_book=1` for both groups) to get each group’s **potential bookings** under your WAU/retention/frequencies and flights gate.
  2. Regress **observed total bookings** (sum of LOBs) on the two group potentials to estimate weekly `p_book` (clamped to `[0, max_weekly_booker_rate]`).
     Options: `shared_rate_only: true` fits a single shared coefficient; `weeks_to_use` limits the fitting window.

* **`target_annual_take`**:

  1. Run unit‑rate simulation to compute **take with p\_book=1**.
  2. Scale by `target_annual_take_usd / take_with_unit` to get a **single scalar booker rate** applied to both groups (bounded by `max_weekly_booker_rate`).

If `mode: none`, or inputs for a chosen mode are missing, the model uses any provided `booker_rate_weekly_by_group` (filling missing with `0`).

**Observed bookings CSV** expectations:

* Must have a `week` column and one or more LOB columns named (case‑insensitive): `flights, hotels, cars, cruises`.
* Weeks are aligned from the top; the first `weeks_to_use` rows are used.

---

## Running the simulator

```bash
python run.py
```

This reads `config.yaml`, writes `simulation_output.csv`, then prints a sum of numeric columns (excluding `week`).

### Output columns (per week)

* `week`
* `active_partners`, `active_non_partners`
* `bookers_partners`, `bookers_non_partners`
* `bookings_flights`, `bookings_hotels`, `bookings_cars`, `bookings_cruises`
* `take_usd_flights`, `take_usd_hotels`, `take_usd_cars`, `take_usd_cruises`
* `bookings_total`, `take_usd_total`
* `new_accounts_partners`, `new_accounts_non_partners`

---

## Config reference (every key)

> All keys live in `config.yaml`. Relative file paths are resolved relative to the config file’s directory. Unknown keys are ignored.

### Global

* **`time_grain`**: string, currently informational. Only `weekly` is implemented.
* **`weeks`**: integer simulation horizon; default `52`.

### Groups & initialization

* **`user_groups.active_user_share`**: `{partners: float, non_partners: float}` used **only** to split `initial_wau_total` when `initial_wau_by_group` is not provided. Defaults to `{0.5, 0.5}`.
* **`initial_wau_total`**: integer WAU total to split by `active_user_share`. Default `300000` if both this and `initial_wau_by_group` are absent.
* **`initial_wau_by_group`**: dict of per‑group WAU; when provided, overrides `initial_wau_total`.
* **`dau_wau_ratio`**: optional float for **sanity only** (no effect on simulation). Presently not printed by `run.py`—keep as a reference.

### New accounts (signup inflow)

Two mutually exclusive ways to provide weekly new accounts by group:

1. **Historical series (preferred if present)**
   `historical_series.weekly_new_accounts_by_group`: arrays of length ≥ `weeks` for each group. The simulator truncates/zero‑pads to `weeks`.
2. **`new_accounts.mode`**

   * `mode: baseline_growth`

     * `baseline_per_week_by_group`: starting weekly signups per group.
     * `growth_rate_weekly_by_group`: multiplicative WoW growth per group (e.g., `0.01` for +1%/wk).
       The generated series is `baseline × (1+growth)^t` for `t=0..weeks-1`.
   * `mode: series`

     * `series_by_group`: explicit arrays per group (length ≥ `weeks` recommended); truncated/zero‑padded if shorter.

### Early‑life presence (0–90d into WAU)

Provide **one** of the following;

* **`cohort_csv_path`**: path to a cohort matrix (ID column + day‑offset numeric columns `0..90`) **or** a long‑form table with `day` and `p_active_day` columns. The loader normalizes by day‑0, averages across cohorts, then builds weekly “union presence” weights `W[0..12]`.
  Optional: **`retention_hazard_scale`** (float ≥0) scales daily presence via `p' = 1 − (1 − p)^scale` before computing `W`.
* **`retention_points`**: dictionary with anchors `{day_1, day_7, day_30, day_90}` (interpreted as daily presence/survival). A smooth curve is built in log‑space between anchors, then `W` and `p_grad` are derived as above.

If neither is provided, defaults are used:
`W = [0.25, 0.18, 0.18, 0.18, 0.12, 0.12, 0.12, 0.12, 0.08, 0.08, 0.08, 0.08, 0.08]` and `p_grad = 0.02`.

> **Note on graduation**: `p_grad` is the **average of `W[9..12]`** (the last four onboarding weeks), not simply day‑90 presence. This matches the code.

### Mature churn (post‑90d)

* Preferred direct weekly input: **`estimated_churn_from_12m_inactivity.weekly_mature_churn_rate_by_group`** `{partners, non_partners}`.
* Otherwise, provide **`monthly_mature_churn_rate_by_group`**; the simulator converts to weekly via compounding:
  `weekly = 1 − (1 − monthly)^(1/4.345)`.
* If neither is supplied, the model falls back to a monthly `0.03` and converts it.

### Booker rate (conversion)

* **`booker_rate_weekly_by_group`**: `{partners, non_partners}`. Used only when `calibration.mode` is `none` or calibration inputs are missing.

### Calibration block

* **`calibration.mode`**: one of `observed_bookings`, `target_annual_take`, `none` (code default is `none`).
* **`calibration.bookings_csv_path`**: path to observed weekly bookings by LOB (columns: `week, flights, hotels, cars, cruises`). Required for `observed_bookings`.
* **`calibration.weeks_to_use`**: integer; default `52`.
* **`calibration.shared_rate_only`**: boolean; fit a single shared `p_book` across groups (stability option).
* **`calibration.max_weekly_booker_rate`**: float cap; default `1.0`.
* **`calibration.target_annual_take_usd`**: positive float; required for `target_annual_take`.

### Frequencies (intensity) & economics

* **`purchases_per_booker_per_year_by_lob_by_group`**: for each group and LOB, the average purchases per booker per year. These are treated as **fixed** in the current model; booker rate calibration does **not** change them.
* **`lobs`**: per‑LOB economics:

  * `flights.take_per_pnr_usd`
  * `hotels.take_per_booking_usd`
  * `cars.take_per_booking_usd`
  * `cruises.take_per_booking_usd`

### Flights gate

* **`flights_gate`**: `{partners: true|false, non_partners: true|false}`; if `false` for a group, that group’s **flights** demand is set to **0** (suppression; not redistributed). Default `{partners: true, non_partners: false}`.

---

## Assumptions & scope

* No seasonality, no randomness (all expectations).
* Two groups (`partners`, `non_partners`) and four LOBs are hard‑coded.
* Initial WAU starts fully mature; future extension could seed onboarding history.

## Tips

* When switching to `observed_bookings`, ensure the CSV week indexing aligns with the model’s week 1..`weeks`.
* Use `shared_rate_only: true` initially if your observed data are noisy; relax later.

## Repository map

* `model.py` — dataclasses, loaders, retention processing, simulator core, calibration helpers.
* `run.py` — CLI runner that writes `simulation_output.csv` and prints totals.
* `config.yaml` — your assumptions & data paths (commented example filled in).

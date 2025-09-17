# OTA Growth Model

A simple simulation to project weekly active users, bookers, bookings by line of business (LOB), and take (net revenue) over a configurable horizon.

* Input: `config.yaml`
* Output: `simulation_output.csv`
* Entry point: `python run.py`

> **Important change**: All calibration settings now live **under a nested `calibration:` block** (see below). The code no longer reads top‑level calibration keys like `calibration_mode:`.

---

## Quick start

```bash
pip install -r requirements.txt
# or: pip install pandas numpy pyyaml

python run.py
# writes simulation_output.csv and prints a run summary
```

---

## Configuration (`config.yaml`)

### Top-level keys

```yaml
# number of weeks to simulate
weeks: 52

# (optional) RNG seed for reproducible runs
seed: 42

# (optional) used only for console Δ reporting at the end of a run
baseline_total_take_usd: 0

# enable/disable product access per user group
flights_gate:
  partners: true
  non_partners: true

# LOB unit economics (used to compute take from bookings)
lobs:
  flights:
    take_per_pnr_usd: 9.50
  hotels:
    take_per_booking_usd: 14.00
  cars:
    take_per_booking_usd: 6.00
  cruises:
    take_per_booking_usd: 25.00

# Average purchases per booker per year, by LOB
# These drive weekly booking volume as: bookers * (purchases_per_year / 52)
purchases_per_booker_per_year_by_lob:
  flights: 1.6
  hotels: 1.8
  cars: 0.7
  cruises: 0.2

# New-account inflow; either provide a weekly baseline (constant) and optional growth,
# or provide your own per-week series (advanced, see notes).
new_accounts:
  baseline_per_week_by_group:
    partners: 1200
    non_partners: 800
  growth_rate_weekly_by_group:   # fractional; e.g., 0.01 == +1% per week compounding
    partners: 0.00
    non_partners: 0.00

# Weekly mature churn rates by group (see notes for monthly → weekly conversion)
weekly_mature_churn_rate_by_group:
  partners: 0.006
  non_partners: 0.011
```

### Calibration (nested block)

All calibration-related inputs must be placed under the `calibration:` key.

```yaml
calibration:
  # "none" disables calibration; "target_annual_take" tunes booker rate(s)
  mode: none  # or: target_annual_take

  # If mode == target_annual_take, the simulator will adjust booker rate(s)
  # to match this total take (USD) across the horizon.
  target_annual_take_usd: 22500000

  # Optional: path to a weekly bookings CSV used for calibration heuristics
  bookings_csv_path: data/bookings_history.csv

  # Optional: limit historical weeks consumed from the CSV for calibration
  weeks_to_use: 52

  # If true, calibrate a single shared weekly booker rate for all groups/LOBs
  shared_rate_only: false

  # Upper bound on the weekly booker rate during calibration (safety valve)
  max_weekly_booker_rate: 0.15
```

**Notes**

* If `mode: none`, no calibration occurs and your explicit config values are used as-is.
* If `mode: target_annual_take`, the simulator iteratively adjusts weekly booker rate(s) (bounded by `max_weekly_booker_rate`) to hit `target_annual_take_usd`. If `shared_rate_only: true`, a single rate is learned; otherwise rates may be learned per group/LOB.

### Churn rates: weekly vs monthly

The simulator expects **weekly** churn inputs via `weekly_mature_churn_rate_by_group`. If you only have **monthly** churn, convert to weekly before placing it in the config using:

$weekly$ = `1 - (1 - monthly)^(1/4.345)`

> Example: `monthly = 0.025` ⇒ `weekly ≈ 0.00581`.

---

## What the simulator writes

`simulation_output.csv` with the following columns (one row per week):

* `week` — 1..N
* `active_partners`, `active_non_partners`, `active_total`
* `bookers_partners`, `bookers_non_partners`, `bookers_total`
* `bookings_flights`, `bookings_hotels`, `bookings_cars`, `bookings_cruises`, `bookings_total`
* `take_flights_usd`, `take_hotels_usd`, `take_cars_usd`, `take_cruises_usd`, `take_total_usd`
* `new_accounts_partners`, `new_accounts_non_partners`
* `churn_rate_partners`, `churn_rate_non_partners`
* `churned_partners`, `churned_non_partners`, `churned_total`

**Relationships (at a glance)**

* `bookings_<lob> = bookers_total_or_group × (purchases_per_booker_per_year_by_lob[<lob>] / 52)`
* `take_<lob>_usd = bookings_<lob> × lobs[<lob>].take_per_*_usd`
* If both `flights_gate.partners` and `flights_gate.non_partners` are `false`, `bookings_flights` will be zero.

**Console summary**
At the end of a run, the script prints: total take, Δ vs `baseline_total_take_usd` (if provided), estimated booker conversion, and averages for new users and churn.

---

## Examples

### Minimal `config.yaml`

```yaml
weeks: 52
seed: 7
baseline_total_take_usd: 0

flights_gate:
  partners: true
  non_partners: true

lobs:
  flights: { take_per_pnr_usd: 9.5 }
  hotels:  { take_per_booking_usd: 14 }
  cars:    { take_per_booking_usd: 6 }
  cruises: { take_per_booking_usd: 25 }

purchases_per_booker_per_year_by_lob:
  flights: 1.6
  hotels: 1.8
  cars: 0.7
  cruises: 0.2

new_accounts:
  baseline_per_week_by_group:
    partners: 1200
    non_partners: 800
  growth_rate_weekly_by_group:
    partners: 0.00
    non_partners: 0.00

weekly_mature_churn_rate_by_group:
  partners: 0.006
  non_partners: 0.011

calibration:
  mode: none
  target_annual_take_usd: 0
  bookings_csv_path: null
  weeks_to_use: 52
  shared_rate_only: false
  max_weekly_booker_rate: 0.15
```

---

## Troubleshooting

* **`config.yaml` not found**: Ensure the file is in the project root (same folder as `run.py`).
* **Missing LOB economics**: Define `take_per_pnr_usd` for `flights` and `take_per_booking_usd` for `hotels`, `cars`, and `cruises`.
* **Take looks too high/low**: Recheck `purchases_per_booker_per_year_by_lob` and LOB take values.
* **Churn looks off**: Confirm you converted monthly rates to weekly before placing them under `weekly_mature_churn_rate_by_group`.
* **Calibration overshoots**: Lower `max_weekly_booker_rate`, enable `shared_rate_only`, or feed more representative history via `bookings_csv_path` and `weeks_to_use`.

---

## Dev notes

* The simulator reads only `config.yaml`. A previous mention of `config.py` is obsolete.
* Calibration keys are **strictly nested** under `calibration:`.
* CSV schema remains stable; downstream notebooks or dashboards can rely on the columns listed above.

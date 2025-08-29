
# Weekly Travel Finance Simulator (Active → Bookers)

This repo contains a **lean, data-anchored simulator** that projects from **Active Users (WAU)** to **Bookers**, to **Bookings by LOB**, to **Take ($)**. It aligns with your chosen simplifications:

- Keep **New Accounts** as an acquisition lever
- Initialize **Active** from **WAU**
- Use **90-day cohort CSV** for early-life presence (0–90d) into WAU
- Fit **booker rate `p_book`** (Mode A) while keeping per‑LOB per‑booker **frequencies fixed**
- **Flights gate**: **Partners** can book flights; **Non‑partners** cannot (suppression; no redistribution)
- **DAU/WAU** is used as a **sanity check** only

---

## Files

- `model.py` — dataclasses, loaders, retention processing, simulator core, calibration helper.
- `run.py` — runs the sim and writes `simulation_output.csv`.
- `config.yaml` — assumptions & data paths (with detailed comments).

---

## How to run

```bash
python run.py
```

Output: `/mnt/data/simulation_output.csv` with weekly rows:

- `week`
- `active_partners`, `active_non_partners`
- `bookers_partners`, `bookers_non_partners`
- `bookings_flights`, `bookings_hotels`, `bookings_cars`, `bookings_cruises`
- `take_usd_flights`, `take_usd_hotels`, `take_usd_cars`, `take_usd_cruises`
- `bookings_total`, `take_usd_total`
- `new_accounts_partners`, `new_accounts_non_partners`

---

## Key mechanics

### 1) Active stock (WAU)

We split **Active** into two **non-overlapping** parts per group:

- **Onboarding presence (0–90d)**: From the **cohort CSV**, we compute 13 **weekly presence weights** `W_k` (k=0..12). Each new weekly cohort contributes `W_k` to WAU as it ages. After week 13 (≈ day 90), a fraction `p_grad` (the CSV’s day‑90 presence) **graduates** to the mature pool.
- **Mature active**: Users beyond 90 days. The pool decays each week by the **weekly mature churn rate** converted from your **monthly** input.

Transitions (per group):
```
mature[t+1] = (mature[t] + onboarding_age[12][t] * p_grad) * (1 - c_w)
onboarding_age[k+1][t+1] = onboarding_age[k][t],  k=0..11
onboarding_age[0][t+1] = new_accounts[t]
active[t] = mature[t] + Σ_k onboarding_age[k][t] * W_k
```

### 2) Bookers & bookings

- **Bookers**: `bookers[g,t] = active[g,t] * p_book[g]`
- **Expected bookings** per LOB: `bookers[g,t] * (freq_per_booker_per_year[g,lob] / 52)`
- **Flights gate**: set **non‑partner flights** demand to **0** (suppression).
- **Take**: multiply by per‑LOB economics (flights use `take_per_segment × avg_segments_per_pnr`).

### 3) Calibration of `p_book` (Mode A)

Two modes are supported; the config currently defaults to **target_annual_take** for convenience:

- **Observed bookings mode**: Provide a CSV of weekly bookings by LOB, the model builds a design matrix and fits `p_book[partners]`, `p_book[non_partners]` by least squares (non-negative clamp).
- **Target annual take mode** (current default): We compute a single `p_book` such that the **total annual take** equals your provided target (e.g., $23M), then assign it to both groups.

You can switch to `observed_bookings` later to get per‑group `p_book` when you have the weekly bookings CSV available.

---

## Tuning levers

- **Acquisition**: `new_accounts` baseline & growth by group.
- **Retention (mature)**: `monthly_mature_churn_rate_by_group` (converted to weekly via compounding).
- **Early-life presence**: comes from your **cohort CSV**; you can apply a `retention_hazard_scale` later for “what‑if” goals.
- **Conversion**: `p_book` (calibrated); you can override directly.
- **Intensity / Mix**: `purchases_per_booker_per_year_by_lob_by_group`.
- **Economics**: `lobs.*` take assumptions.
- **Flights gate**: `flights_gate` (non‑partners fixed to false per your instruction).

---

## DAU/WAU sanity check

If you set `dau_wau_ratio`, the runner prints the implied average active days per WAU.
Use this qualitatively to check whether your churn + `p_book` feel plausible.

---

## Notes & assumptions

- The cohort CSV is interpreted as **counts of active/returning users by day offset** per signup date.
  We normalize by day‑0 to get per‑day activity probabilities, average across cohorts,
  then map to weekly presence weights `W_k` via a union‑of‑days approximation.
- Graduation to mature at day‑90 uses `p_grad = p_active_day[90]`. You can refine this once you agree on the exact graduation rule.
- Flights demand for **non‑partners** is **suppressed** (not redistributed).
- No seasonality or randomness is included yet; hooks can be added later.

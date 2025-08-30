from __future__ import annotations
import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import yaml
import pandas as pd
import numpy as np

# -----------------------------
# Helpers
# -----------------------------

GROUPS = ("partners", "non_partners")
LOBS = ("flights", "hotels", "cars", "cruises")


def monthly_to_weekly_rate(c_m: float) -> float:
    """Convert a monthly churn rate to an equivalent weekly rate using compounding."""
    c_m = max(0.0, min(1.0, float(c_m)))
    return 1.0 - (1.0 - c_m) ** (1.0 / 4.345)


def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class LOBEconomics:
    take_per_pnr_usd: float = 0.0
    take_per_booking_usd: float = 0.0


@dataclass
class Params:
    weeks: int
    # Initialization
    initial_wau_by_group: Dict[str, float]
    # Inflows
    new_accounts_mode: str  # 'series' or 'baseline_growth'
    new_accounts_series_by_group: Dict[str, List[float]] = field(default_factory=dict)
    signups_weekly_baseline_by_group: Dict[str, float] = field(default_factory=dict)
    signup_growth_rate_weekly_by_group: Dict[str, float] = field(default_factory=dict)
    acct_create_rate: float = 1.0

    # Cohort presence + graduation
    cohort_use_csv: bool = False
    cohort_csv_path: Optional[str] = None
    retention_points: Optional[Dict[str, float]] = None
    retention_hazard_scale: float = 1.0
    W: List[float] = field(default_factory=lambda: [0.0]*13)  # weekly presence weights for ages 0..12
    p_grad: float = 0.0  # graduation fraction

    # Mature churn
    weekly_mature_churn_rate_by_group: Dict[str, float] = field(default_factory=dict)

    # Booker
    booker_rate_weekly_by_group: Dict[str, Optional[float]] = field(default_factory=dict)
    calibration_mode: str = "none"  # 'none' | 'target_annual_take' | 'observed_bookings'
    target_annual_take_usd: Optional[float] = None
    bookings_csv_path: Optional[str] = None
    weeks_to_use: int = 52
    calibration_shared_rate_only: bool = False
    calibration_max_weekly_booker_rate: float = 1.0

    # LOB frequencies (per-booker per year) and economics
    purchases_per_booker_per_year_by_lob_by_group: Dict[str, Dict[str, float]] = field(default_factory=dict)
    lobs: Dict[str, LOBEconomics] = field(default_factory=dict)

    # Flights gate
    flights_gate: Dict[str, bool] = field(default_factory=lambda: {"partners": True, "non_partners": False})

    # Sanity info
    dau_wau_ratio: Optional[float] = None


@dataclass
class State:
    mature_active: Dict[str, float]
    onboarding_age: Dict[str, List[float]]  # per-group 13-element array


# -----------------------------
# Loading + derivation
# -----------------------------

def load_params(config_path: str) -> Params:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    weeks = int(cfg.get("weeks", 52))
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # 1) Initialization: WAU split
    init_wau = safe_get(cfg, "initial_wau_by_group", default=None)
    if not init_wau or not init_wau.get("partners"):
        # Fallback: split a provided WAU total if present in notes; otherwise default 50/50 of 300k
        wau_total = cfg.get("initial_wau_total", 300_000)
        share = safe_get(cfg, "user_groups", "active_user_share", default={"partners": 0.5, "non_partners": 0.5})
        init_wau = {g: wau_total * float(share.get(g, 0.5)) for g in GROUPS}

    # 2) New accounts inflow (prefer historical series if present)
    new_accounts_series_by_group = safe_get(cfg, "historical_series", "weekly_new_accounts_by_group", default=None)
    if new_accounts_series_by_group:
        na_mode = "series"
        na_series = {g: [float(x) for x in new_accounts_series_by_group.get(g, [0.0]*weeks)][:weeks] for g in GROUPS}
        for g in GROUPS:
            if len(na_series[g]) < weeks:
                na_series[g].extend([0.0]*(weeks - len(na_series[g])))
    else:
        na_mode = safe_get(cfg, "new_accounts", "mode", default="baseline_growth")
        na_series = {}
        if na_mode == "series":
            series = safe_get(cfg, "new_accounts", "series_by_group", default={})
            for g in GROUPS:
                na_series[g] = [float(x) for x in series.get(g, [0.0] * weeks)][:weeks]
                if len(na_series[g]) < weeks:
                    na_series[g].extend([0.0] * (weeks - len(na_series[g])))
        else: # baseline_growth
            base = safe_get(cfg, "new_accounts", "baseline_per_week_by_group", default={})
            growth = safe_get(cfg, "new_accounts", "growth_rate_weekly_by_group", default={})
            for g in GROUPS:
                b = float(base.get(g, 0.0))
                r = float(growth.get(g, 0.0))
                na_series[g] = [b * ((1.0 + r) ** t) for t in range(weeks)]

    # 3) Mature churn weekly rate (prefer estimated weekly from config)
    weekly_churn: Dict[str, float] = {}
    est_weekly = safe_get(cfg, "estimated_churn_from_12m_inactivity", "weekly_mature_churn_rate_by_group", default={})
    if est_weekly and all(g in est_weekly for g in GROUPS):
        weekly_churn = {g: float(est_weekly[g]) for g in GROUPS}
    else:
        monthly = safe_get(cfg, "monthly_mature_churn_rate_by_group", default={})
        if monthly:
            weekly_churn = {g: monthly_to_weekly_rate(float(monthly.get(g, 0.0))) for g in GROUPS}
        else:
            weekly_churn = {g: monthly_to_weekly_rate(0.03) for g in GROUPS}

    # 4) Cohort weights W_k and graduation p_grad
    cohort_path = cfg.get("cohort_csv_path")
    if cohort_path and not os.path.isabs(cohort_path):
        cohort_path = os.path.join(config_dir, cohort_path)
    hazard_scale = float(cfg.get("retention_hazard_scale", 1.0))
    retention_points = cfg.get("retention_points")

    if cohort_path and os.path.exists(cohort_path):
        W, p_grad = _derive_presence_from_cohort_csv(cohort_path, hazard_scale)
        cohort_use_csv = True
    elif retention_points:
        W, p_grad = _derive_presence_from_retention_points(retention_points)
        cohort_use_csv = False
    else:
        W = [0.25] + [0.18] * 3 + [0.12] * 4 + [0.08] * 5
        p_grad = 0.02
        cohort_use_csv = False

    # 5) Booker rates + calibration mode
    booker_cfg = cfg.get("booker_rate_weekly_by_group", {})
    p_book = {g: booker_cfg.get(g) for g in GROUPS}

    calib_mode = safe_get(cfg, "calibration", "mode", default="none")
    target_take = safe_get(cfg, "calibration", "target_annual_take_usd", default=None)
    bookings_csv_path = safe_get(cfg, "calibration", "bookings_csv_path", default=None)
    if bookings_csv_path and not os.path.isabs(bookings_csv_path):
        bookings_csv_path = os.path.join(config_dir, bookings_csv_path)
    weeks_to_use = safe_get(cfg, "calibration", "weeks_to_use", default=52)
    shared_rate_only = bool(safe_get(cfg, "calibration", "shared_rate_only", default=False))
    max_weekly_booker_rate = float(safe_get(cfg, "calibration", "max_weekly_booker_rate", default=1.0))

    if calib_mode == "target_annual_take" and target_take is not None:
        calibration_mode = "target_annual_take"
        target_annual_take_usd = float(target_take)
    elif calib_mode == "observed_bookings" and bookings_csv_path is not None:
        calibration_mode = "observed_bookings"
        target_annual_take_usd = None
    else:
        calibration_mode = "none"
        target_annual_take_usd = None

    # 6) Frequencies + economics
    freq = safe_get(cfg, "purchases_per_booker_per_year_by_lob_by_group", default={})
    lobs_cfg = safe_get(cfg, "lobs", default={})

    lobs: Dict[str, LOBEconomics] = {}
    for lob in LOBS:
        lc = lobs_cfg.get(lob, {}) or {}
        lobs[lob] = LOBEconomics(
            take_per_pnr_usd=float(lc.get("take_per_pnr_usd", 0.0) or 0.0),
            take_per_booking_usd=float(lc.get("take_per_booking_usd", 0.0) or 0.0),
        )

    flights_gate = safe_get(cfg, "flights_gate", default={"partners": True, "non_partners": False})
    dau_wau_ratio = cfg.get("dau_wau_ratio")

    return Params(
        weeks=weeks,
        initial_wau_by_group={g: float(init_wau.get(g, 0.0)) for g in GROUPS},
        new_accounts_mode=na_mode,
        new_accounts_series_by_group=na_series,
        signups_weekly_baseline_by_group=safe_get(cfg, "new_accounts", "baseline_per_week_by_group", default={}),  # type: ignore[arg-type]
        signup_growth_rate_weekly_by_group=safe_get(cfg, "new_accounts", "growth_rate_weekly_by_group", default={}),  # type: ignore[arg-type]
        acct_create_rate=1.0,
        cohort_use_csv=cohort_use_csv,
        cohort_csv_path=cohort_path,
        retention_points=retention_points,
        retention_hazard_scale=hazard_scale,
        W=W,
        p_grad=float(p_grad),
        weekly_mature_churn_rate_by_group=weekly_churn,
        booker_rate_weekly_by_group=p_book,
        calibration_mode=calibration_mode,
        target_annual_take_usd=target_annual_take_usd,
        bookings_csv_path=bookings_csv_path,
        weeks_to_use=int(weeks_to_use),
        calibration_shared_rate_only=shared_rate_only,
        calibration_max_weekly_booker_rate=max_weekly_booker_rate,
        purchases_per_booker_per_year_by_lob_by_group=freq,
        lobs=lobs,
        flights_gate={g: bool(flights_gate.get(g, g == "partners")) for g in GROUPS},
        dau_wau_ratio=float(dau_wau_ratio) if dau_wau_ratio is not None else None,
    )


def _derive_presence_from_cohort_csv(csv_path: str, hazard_scale: float = 1.0) -> Tuple[List[float], float]:
    """
    Expect either:
      A) long form with columns ['day', 'p_active_day'] (fractions 0..1)
      B) a cohort matrix where a non-numeric id column (e.g. 'cohort_date') is followed by numeric
         day-offset columns '0'..'90'. We normalize each row by day-0 and average across cohorts.
    Returns (W[0..12], p_grad).
    """
    df = pd.read_csv(csv_path)
    p_active_day = None

    # Case A: explicit columns
    cols_lower = [str(c).strip().lower() for c in df.columns]
    if "day" in cols_lower and ("p_active_day" in cols_lower or "p_active" in cols_lower):
        day_col = df.columns[cols_lower.index("day")]
        p_col = df.columns[cols_lower.index("p_active_day")] if "p_active_day" in cols_lower else df.columns[cols_lower.index("p_active")]
        tmp = df[[day_col, p_col]].dropna()
        tmp.columns = ["day", "p"]
        tmp = tmp[(tmp["day"] >= 0) & (tmp["day"] <= 90)]
        tmp = tmp.groupby("day", as_index=False)["p"].mean()
        p_active_day = {int(d): float(p) for d, p in zip(tmp["day"], tmp["p"])}

    # Case B: cohort matrix with numeric day columns in the HEADER
    if p_active_day is None:
        day_cols = []
        for c in df.columns:
            try:
                d = int(float(str(c).strip()))
                if 0 <= d <= 90:
                    day_cols.append((d, c))
            except Exception:
                continue
        day_cols.sort(key=lambda x: x[0])  # sort by day integer
        if day_cols:
            if not any(d == 0 for d, _ in day_cols):
                # if no day-0 column, we can't normalize; fall back below
                day_cols = []
        if day_cols:
            series = []
            # normalize each cohort row by its day-0 column
            day0_label = [c for d, c in day_cols if d == 0][0]
            for _, row in df.iterrows():
                try:
                    base = float(row[day0_label])
                except Exception:
                    base = 0.0
                if base and base > 0:
                    s = {}
                    for d, col in day_cols:
                        try:
                            s[int(d)] = max(0.0, float(row[col]) / base)
                        except Exception:
                            s[int(d)] = 0.0
                    series.append(s)
            if series:
                p_active_day = {d: float(sum(s.get(d, 0.0) for s in series) / max(1, len(series))) for d, _ in day_cols}

    # Fallback: mild exponential presence
    if p_active_day is None:
        p_active_day = {d: 0.05 * math.exp(-d / 30.0) for d in range(0, 91)}
        p_active_day[0] = 1.0

    # Hazard scaling
    if hazard_scale != 1.0:
        for d in list(p_active_day.keys()):
            p = max(0.0, min(1.0, float(p_active_day[d])))
            p_active_day[d] = 1.0 - (1.0 - p) ** hazard_scale

    # Weekly union presence W_k: probability seen at least once in week k (days 7k..7k+6)
    W = []
    for k in range(13):
        days = [7 * k + i for i in range(7) if 7 * k + i <= 90]
        prod = 1.0
        for d in days:
            p = max(0.0, min(1.0, p_active_day.get(d, 0.0)))
            prod *= (1.0 - p)
        W_k = 1.0 - prod
        W.append(W_k)

    # Graduation fraction: average presence across the last 4 onboarding weeks (ages 9..12)
    tail = W[9:13] if len(W) >= 13 else W[-4:]
    p_grad = float(np.mean(tail)) if tail else float(W[-1] if W else 0.0)
    p_grad = max(0.0, min(1.0, p_grad))
    return W, p_grad


def _derive_presence_from_retention_points(points: Dict[str, float]) -> Tuple[List[float], float]:
    """
    points expects keys like day_1, day_7, day_30, day_90 representing survival or presence.
    We'll build a crude daily curve and then compute W_k and p_grad similarly to the cohort path.
    """
    # Minimal monotone interpolation across key points
    anchors = {
        0: 1.0,
        1: float(points.get("day_1", 0.2)),
        7: float(points.get("day_7", 0.06)),
        30: float(points.get("day_30", 0.03)),
        90: float(points.get("day_90", 0.008)),
    }
    # Build per-day linearly in log space (avoid negatives)
    daily = {}
    keys = sorted(anchors.keys())
    for i in range(len(keys) - 1):
        d0, d1 = keys[i], keys[i + 1]
        y0, y1 = max(1e-9, anchors[d0]), max(1e-9, anchors[d1])
        for d in range(d0, d1 + 1):
            t = (d - d0) / (d1 - d0) if d1 > d0 else 0.0
            y = math.exp(math.log(y0) * (1 - t) + math.log(y1) * t)
            daily[d] = y
    # Compute W and p_grad
    W = []
    for k in range(13):
        days = [7 * k + i for i in range(7) if 7 * k + i <= 90]
        prod = 1.0
        for d in days:
            p = max(0.0, min(1.0, daily.get(d, 0.0)))
            prod *= (1.0 - p)
        W_k = 1.0 - prod
        W.append(W_k)
    tail = W[9:13] if len(W) >= 13 else W[-4:]
    p_grad = float(np.mean(tail)) if tail else float(W[-1] if W else 0.0)
    p_grad = max(0.0, min(1.0, p_grad))
    return W, p_grad


# -----------------------------
# Simulation
# -----------------------------

def _init_state(p: Params) -> State:
    # Start with all WAU in the mature pool unless the caller provides a history to seed onboarding (future extension).
    mature = {g: float(p.initial_wau_by_group.get(g, 0.0)) for g in GROUPS}
    onboarding = {g: [0.0] * 13 for g in GROUPS}
    return State(mature_active=mature, onboarding_age=onboarding)


def _simulate_weeks(p: Params, p_book_override: Optional[Dict[str, float]] = None) -> Tuple[List[Dict[str, Any]], float]:
    """
    Returns (rows, total_take) for the configured number of weeks.
    If p_book_override is provided, use those booker rates instead of p.booker_rate_weekly_by_group.
    """
    state = _init_state(p)
    rows: List[Dict[str, Any]] = []
    total_take = 0.0

    # Resolve booker rates to use
    if p_book_override is not None:
        p_book = p_book_override
    else:
        p_book = {g: (p.booker_rate_weekly_by_group.get(g) or 0.0) for g in GROUPS}

    # Pre-pull frequencies
    freq = p.purchases_per_booker_per_year_by_lob_by_group

    for t in range(p.weeks):
        # 1) Active (WAU) this week
        active = {}
        for g in GROUPS:
            onboarding_presence = sum(state.onboarding_age[g][k] * p.W[k] for k in range(13))
            active[g] = state.mature_active[g] + onboarding_presence

        # 2) Bookers
        bookers = {g: active[g] * float(p_book.get(g, 0.0)) for g in GROUPS}

        # 3) Expected bookings by LOB with flights gate
        bookings_by_lob = {lob: 0.0 for lob in LOBS}
        for g in GROUPS:
            for lob in LOBS:
                fpy = float(freq.get(g, {}).get(lob, 0.0))
                weekly_freq = fpy / 52.0
                demand = bookers[g] * weekly_freq
                if lob == "flights" and not p.flights_gate.get(g, g == "partners"):
                    demand = 0.0  # suppression (no redistribution)
                bookings_by_lob[lob] += demand

        # 4) Take
        take_by_lob = {
            "flights": bookings_by_lob["flights"] * p.lobs["flights"].take_per_pnr_usd,
            "hotels": bookings_by_lob["hotels"] * p.lobs["hotels"].take_per_booking_usd,
            "cars": bookings_by_lob["cars"] * p.lobs["cars"].take_per_booking_usd,
            "cruises": bookings_by_lob["cruises"] * p.lobs["cruises"].take_per_booking_usd,
        }
        take_total = sum(take_by_lob.values())
        total_take += take_total

        # 5) Emit row
        row = {
            "week": t + 1,
            "active_partners": active["partners"],
            "active_non_partners": active["non_partners"],
            "bookers_partners": bookers["partners"],
            "bookers_non_partners": bookers["non_partners"],
            "bookings_flights": bookings_by_lob["flights"],
            "bookings_hotels": bookings_by_lob["hotels"],
            "bookings_cars": bookings_by_lob["cars"],
            "bookings_cruises": bookings_by_lob["cruises"],
            "take_usd_flights": take_by_lob["flights"],
            "take_usd_hotels": take_by_lob["hotels"],
            "take_usd_cars": take_by_lob["cars"],
            "take_usd_cruises": take_by_lob["cruises"],
            "bookings_total": sum(bookings_by_lob.values()),
            "take_usd_total": take_total,
            "new_accounts_partners": p.new_accounts_series_by_group["partners"][t] if p.new_accounts_series_by_group.get("partners") else 0.0,
            "new_accounts_non_partners": p.new_accounts_series_by_group["non_partners"][t] if p.new_accounts_series_by_group.get("non_partners") else 0.0,
        }
        rows.append(row)

        # 6) Transition to next week
        next_state = State(
            mature_active=state.mature_active.copy(),
            onboarding_age={g: state.onboarding_age[g][:] for g in GROUPS},
        )
        for g in GROUPS:
            # Graduate age-12 into mature (using p_grad), then apply mature churn
            graduates = state.onboarding_age[g][12] * p.p_grad
            next_state.mature_active[g] += graduates
            # Apply churn
            cw = float(p.weekly_mature_churn_rate_by_group.get(g, 0.0))
            next_state.mature_active[g] *= (1.0 - cw)

            # Age onboarding
            for k in range(12, 0, -1):
                next_state.onboarding_age[g][k] = state.onboarding_age[g][k - 1]
            # New cohort enters age 0
            influx = float(p.new_accounts_series_by_group.get(g, [0.0] * p.weeks)[t])
            next_state.onboarding_age[g][0] = influx

        state = next_state

    return rows, total_take


def _calibrate_booker_rates_for_target_take(p: Params) -> Dict[str, float]:
    """Find a single scalar booker rate applied to both groups that hits target annual take.
    We use linearity: run once with p_book=1.0 to get per-unit take, then scale.
    """
    if not p.target_annual_take_usd or p.target_annual_take_usd <= 0:
        # Fallback to mild defaults
        return {g: (p.booker_rate_weekly_by_group.get(g) or 0.01) for g in GROUPS}

    unit_rates = {g: 1.0 for g in GROUPS}
    _, take_with_unit = _simulate_weeks(p, p_book_override=unit_rates)
    if take_with_unit <= 0:
        scalar = 0.01
    else:
        scalar = p.target_annual_take_usd / take_with_unit
    # Bound
    scalar = max(0.0, min(p.calibration_max_weekly_booker_rate, scalar))
    return {g: scalar for g in GROUPS}


def _get_potential_bookings_by_group(p: Params) -> pd.DataFrame:
    """Helper to run a unit simulation and get potential bookings."""
    unit_rates = {g: 1.0 for g in GROUPS}
    rows, _ = _simulate_weeks(p, p_book_override=unit_rates)
    
    potential_bookings = []
    for row in rows:
        active_partners = row['active_partners']
        active_non_partners = row['active_non_partners']
        
        potential = {'week': row['week']}
        for g in GROUPS:
            active = active_partners if g == 'partners' else active_non_partners
            total_potential_bookings = 0.0
            for lob in LOBS:
                fpy = float(p.purchases_per_booker_per_year_by_lob_by_group.get(g, {}).get(lob, 0.0))
                weekly_freq = fpy / 52.0
                demand = active * weekly_freq
                if lob == "flights" and not p.flights_gate.get(g, g == "partners"):
                    demand = 0.0
                total_potential_bookings += demand
            potential[g] = total_potential_bookings
        potential_bookings.append(potential)
        
    return pd.DataFrame(potential_bookings)


def _calibrate_booker_rates_from_observed_bookings(p: Params) -> Dict[str, float]:
    """
    Fit p_book by regressing observed total bookings against potential bookings from each group.
    Supports a shared-rate option for numerical stability.
    """
    if not p.bookings_csv_path or not os.path.exists(p.bookings_csv_path):
        raise FileNotFoundError(f"Bookings CSV not found at {p.bookings_csv_path}")

    # 1. Potential bookings under unit booker rate
    potential_df = _get_potential_bookings_by_group(p)
    
    # 2. Observed bookings
    observed_df = pd.read_csv(p.bookings_csv_path)
    observed_df.columns = [c.lower() for c in observed_df.columns]
    # Expect columns: week, flights, hotels, cars, cruises (case-insensitive handled above)
    lob_cols = [c for c in observed_df.columns if c in LOBS]
    if not lob_cols:
        raise ValueError("Observed bookings CSV must contain LOB columns: flights, hotels, cars, cruises")
    observed_df['bookings_total'] = observed_df[lob_cols].sum(axis=1)

    # 3. Align weeks
    weeks = min(p.weeks_to_use, len(potential_df), len(observed_df))
    merged = pd.merge(potential_df.head(weeks), observed_df.head(weeks), on='week')

    # 4. Regression
    max_rate = float(p.calibration_max_weekly_booker_rate)
    if p.calibration_shared_rate_only:
        X = (merged['partners'] + merged['non_partners']).values.reshape(-1, 1)
        y = merged['bookings_total'].values
        # closed-form least squares for single feature
        denom = float(np.dot(X[:,0], X[:,0]))
        coeff = float(np.dot(X[:,0], y) / denom) if denom > 0 else 0.0
        coeff = max(0.0, min(max_rate, coeff))
        return {"partners": coeff, "non_partners": coeff}
    else:
        X = merged[['partners', 'non_partners']].values
        y = merged['bookings_total'].values
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        c_partners = max(0.0, min(max_rate, float(coeffs[0])))
        c_non_partners = max(0.0, min(max_rate, float(coeffs[1])))
        return {"partners": c_partners, "non_partners": c_non_partners}


# -----------------------------
# Public API
# -----------------------------

def run_sim(config_path: str) -> List[Dict[str, Any]]:
    """Run the weekly simulator and return a list of rows (dicts)."""
    params = load_params(config_path)

    # Calibrate p_book if requested
    if params.calibration_mode == "target_annual_take":
        p_book = _calibrate_booker_rates_for_target_take(params)
    elif params.calibration_mode == "observed_bookings":
        p_book = _calibrate_booker_rates_from_observed_bookings(params)
    else:
        # Use provided (fill Nones with 0)
        p_book = {g: (params.booker_rate_weekly_by_group.get(g) or 0.0) for g in GROUPS}

    rows, _ = _simulate_weeks(params, p_book_override=p_book)
    return rows

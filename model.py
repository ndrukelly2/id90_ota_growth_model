from __future__ import annotations
import yaml, random
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List
import numpy as np

def _interp_retention_curve(points: Dict[str, float]) -> Tuple[List[float], float]:
    # Build weekly retention array for ages 0..12 (~days 1,7,14,...,84) via linear interpolation
    # from provided points at day_1, day_7, day_30, day_90.
    # Returns (weekly_ret[0..12], ret_at_day90).
    d_keys = ["day_1","day_7","day_30","day_90"]
    for k in d_keys:
        if k not in points:
            raise ValueError(f"retention_curve missing {k}")
    xs = np.array([1,7,30,90], dtype=float)
    ys = np.array([float(points["day_1"]), float(points["day_7"]), float(points["day_30"]), float(points["day_90"])], dtype=float)
    ys = np.clip(ys, 0.0, 1.0)
    sample_days = np.array([1 + 7*i for i in range(13)], dtype=float)  # 1..84 in steps of 7
    weekly = np.interp(sample_days, xs, ys)
    ret90 = float(ys[-1])
    return weekly.tolist(), ret90


LOBs = ["flights","hotels","cars","cruises"]
GROUPS = ["partners","non_partners"]

def monthly_to_weekly(x: float) -> float:
    return x / 4.345

@dataclass
class LOBParams:
    take_per_booking_usd: float = 0.0
    take_per_segment_usd: float = 0.0
    avg_segments_per_pnr: float = 1.0

    def expected_segments(self) -> float:
        return self.avg_segments_per_pnr

@dataclass
class Rates:
    acct_create_rate: float
    new_user_retention_90d: float
    booker_share_weekly: float
    weekly_engaged_churn: float

@dataclass
class UserGroupMix:
    active_user_share: Dict[str, float]
    signup_share: Dict[str, float]

@dataclass
class Params:
    time_grain: str
    commercial_flights_enabled: bool
    lobs: Dict[str, LOBParams]
    active_users_12mo: int
    purchasers_12mo: int
    monthly_new_signups_total: int
    monthly_churn_users: int
    rates: Rates
    initial_mature_engaged_share: float
    historical_purchase_mix_users: Dict[str, float]
    weeks: int
    rng: random.Random
    funnels_by_group: Dict[str, Dict[str, float]] = field(default_factory=dict)
    purchases_per_booker_per_year_by_lob_by_group: Dict[str, Dict[str, float]] = field(default_factory=dict)
    signups_weekly_baseline_by_group: Dict[str, float] = field(default_factory=dict)
    signup_growth_rate_weekly_by_group: Dict[str, float] = field(default_factory=dict)

@dataclass
class State:
    active_users_by_group: Dict[str, float]
    mature_engaged_by_group: Dict[str, float]
    cohort_bases_by_group: Dict[str, List[float]]

def load_params(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    rng = random.Random(cfg.get("random_seed", 123))

    lobs = {}
    fl_cfg = cfg["lobs"]["flights"]
    lobs["flights"] = LOBParams(
        take_per_booking_usd=0.0,
        take_per_segment_usd=float(fl_cfg.get("take_per_segment_usd", 0.0)),
        avg_segments_per_pnr=float(fl_cfg.get("avg_segments_per_pnr", 1.0)),
    )
    for lob in ["hotels","cars","cruises"]:
        lobs[lob] = LOBParams(take_per_booking_usd=float(cfg["lobs"][lob].get("take_per_booking_usd",0.0)))

    rates = Rates(
        acct_create_rate=float(cfg["rates"]["acct_create_rate"]),
        new_user_retention_90d=float(cfg["rates"]["new_user_retention_90d"]),
        booker_share_weekly=float(cfg["rates"]["booker_share_weekly"]),
        weekly_engaged_churn=float(cfg["rates"]["weekly_engaged_churn"]),
    )

    ug = UserGroupMix(
        active_user_share={k.replace("-", "_"): float(v) for k,v in cfg["user_groups"]["active_user_share"].items()},
        signup_share={k.replace("-", "_"): float(v) for k,v in cfg["user_groups"]["signup_share"].items()},
    )

    p = Params(
        time_grain=cfg["time_grain"],
        commercial_flights_enabled=bool(cfg.get("commercial_flights_enabled", False)),
        lobs=lobs,
        active_users_12mo=int(cfg["active_users_12mo"]),
        purchasers_12mo=int(cfg["purchasers_12mo"]),
        monthly_new_signups_total=int(cfg["monthly_new_signups_total"]),
        monthly_churn_users=int(cfg["monthly_churn_users"]),
        rates=rates,
        initial_mature_engaged_share=float(cfg.get("initial_mature_engaged_share", 0.5)),
        historical_purchase_mix_users={k: float(v) for k, v in cfg["historical_purchase_mix_users"].items()},
        weeks=int(cfg.get("weeks", 26)),
        rng=rng,
        funnels_by_group=cfg.get("funnels_by_group", {}),
        purchases_per_booker_per_year_by_lob_by_group={g: {k: float(v) for k,v in d.items()} for g, d in cfg.get("purchases_per_booker_per_year_by_lob_by_group", {}).items()},
        signups_weekly_baseline_by_group={g: float(v) for g,v in cfg.get("signups_weekly_baseline_by_group", {}).items()},
        signup_growth_rate_weekly_by_group={g: float(v) for g,v in cfg.get("signup_growth_rate_weekly_by_group", {}).items()},
    )
    # Optional: new user retention curve toggle
    use_curve = bool(cfg.get("use_new_user_retention_curve", False)) and ("retention_curve" in cfg)
    setattr(p, "use_new_user_retention_curve", use_curve)
    if use_curve:
        weekly_ret, ret90 = _interp_retention_curve(cfg["retention_curve"])
        setattr(p, "weekly_retention_curve", weekly_ret)  # length 13, days ~1..84
        setattr(p, "retention_day90", ret90)
    else:
        setattr(p, "weekly_retention_curve", None)
        setattr(p, "retention_day90", None)

    return p, ug

def derive_initial_state(p: Params, ug: UserGroupMix) -> State:
    active = {"partners": p.active_users_12mo * ug.active_user_share["partners"],
              "non_partners": p.active_users_12mo * ug.active_user_share["non_partners"]}
    mature = {g: p.initial_mature_engaged_share * active[g] for g in active}
    cohorts = {g: [0.0]*13 for g in GROUPS}
    return State(active_users_by_group=active, mature_engaged_by_group=mature, cohort_bases_by_group=cohorts)

def expected_flight_take_per_booking(p: Params) -> float:
    fl = p.lobs["flights"]
    return fl.take_per_segment_usd * fl.expected_segments()

def allocate_product_mix(p: Params) -> Dict[str, float]:
    combos = p.historical_purchase_mix_users
    lob_hits = dict(flights=0.0, hotels=0.0, cars=0.0, cruises=0.0)
    for combo, share in combos.items():
        if combo == "none": continue
        for lob in lob_hits:
            if lob in combo: lob_hits[lob] += share
    total = sum(lob_hits.values())
    return {lob: (lob_hits[lob]/total if total>0 else 1/4) for lob in lob_hits}

def funnel_strength_for(p: Params, lob: str, group: str) -> float:
    st = p.funnels_by_group.get(group, {})
    return float(st.get(lob,0))

def weekly_signups_for_group(p: Params, group: str, week_index_zero_based: int) -> float:
    base = float(p.signups_weekly_baseline_by_group.get(group, 0.0))
    gr = float(p.signup_growth_rate_weekly_by_group.get(group, 0.0))
    return base * ((1.0 + gr) ** week_index_zero_based)

def get_group_lob_freq(p: Params, group: str, lob: str) -> float:
    grp = p.purchases_per_booker_per_year_by_lob_by_group.get(group, {})
    return float(grp.get(lob, 0.0))

def step_week(p: Params, ug: UserGroupMix, s: State, week_index_zero_based: int=0):
    out = {"metrics": {}, "by_lob": {}}

    # Signups from weekly baselines + growth
    signups = {g: weekly_signups_for_group(p, g, week_index_zero_based) for g in GROUPS}

    # Churn (keep monthly->weekly, split by active mix)
    weekly_churn_total = monthly_to_weekly(p.monthly_churn_users)
    tot_active = s.active_users_by_group["partners"] + s.active_users_by_group["non_partners"]
    partners_share = (s.active_users_by_group["partners"]/tot_active) if tot_active>0 else 0.5
    churn = {"partners": weekly_churn_total * partners_share,
             "non_partners": weekly_churn_total * (1 - partners_share)}

    # Update active base
    for g in GROUPS:
        s.active_users_by_group[g] = max(0.0, s.active_users_by_group[g] + signups[g] - churn[g])

    # Engaged dynamics
    # Engaged dynamics
    engaged_totals = {}
    use_curve = getattr(p, "use_new_user_retention_curve", False) and (getattr(p, "weekly_retention_curve", None) is not None)
    for g in GROUPS:
        if use_curve:
            # Shift cohort bases and add new base for this week
            bases = s.cohort_bases_by_group[g]
            transfer_base = bases[-1]
            bases = [signups[g]*p.rates.acct_create_rate] + bases[:-1]
            s.cohort_bases_by_group[g] = bases
            # Onboarding engaged (sum of cohort bases weighted by age-specific retention)
            weekly_ret = getattr(p, "weekly_retention_curve")
            onboarding = sum(bases[age] * weekly_ret[age] for age in range(len(weekly_ret)))
            # Transfer matured cohort into mature pool at 90d level
            ret90 = float(getattr(p, "retention_day90"))
            matured_inflow = transfer_base * ret90
            # Mature stock decays + gets new inflow
            s.mature_engaged_by_group[g] = s.mature_engaged_by_group[g]*(1.0 - p.rates.weekly_engaged_churn) + matured_inflow
            engaged_totals[g] = onboarding + s.mature_engaged_by_group[g]
        else:
            # Back-compat: simple aggregate stock
            s.mature_engaged_by_group[g] = s.mature_engaged_by_group[g]*(1.0 - p.rates.weekly_engaged_churn)
            s.mature_engaged_by_group[g] += signups[g]*p.rates.acct_create_rate*p.rates.new_user_retention_90d
            engaged_totals[g] = s.mature_engaged_by_group[g]

    # Bookers
    bookers = {g: engaged_totals[g]*p.rates.booker_share_weekly for g in GROUPS}

    # Allocate to LOBs
    base_mix = allocate_product_mix(p)
    bookings_by_lob = {lob: 0.0 for lob in LOBs}
    for g in GROUPS:
        raw = {}
        group_total_freq_week = 0.0
        for lob in LOBs:
            # flight gating for non-partners
            if (g == "non_partners") and (not p.commercial_flights_enabled) and lob == "flights":
                freq = 0.0
                raw[lob] = 0.0
            else:
                freq = get_group_lob_freq(p, g, lob)
                fs = funnel_strength_for(p, lob, g)
                raw[lob] = base_mix[lob]*fs*freq
            group_total_freq_week += freq/52.0
        rsum = sum(raw.values())
        weights = {lob: (raw[lob]/rsum if rsum>0 else 1/len(LOBs)) for lob in LOBs}
        group_total = bookers[g]*group_total_freq_week
        for lob in LOBs:
            bookings_by_lob[lob] += group_total*weights[lob]

    # Take
    take_per_booking = {"flights": expected_flight_take_per_booking(p),
                        "hotels": p.lobs["hotels"].take_per_booking_usd,
                        "cars": p.lobs["cars"].take_per_booking_usd,
                        "cruises": p.lobs["cruises"].take_per_booking_usd}
    take_usd_by_lob = {lob: bookings_by_lob[lob]*take_per_booking[lob] for lob in LOBs}

    out["metrics"] = {
        "weekly_signups_total": sum(signups.values()),
        "weekly_churn_total": weekly_churn_total,
        "active_users_total": sum(s.active_users_by_group.values()),
        "engaged_users_total": sum(engaged_totals.values()),
        "bookers_total": sum(bookers.values()),
        "bookings_total": sum(bookings_by_lob.values()),
    }
    out["by_group"] = {"active": s.active_users_by_group.copy(),
                       "engaged": engaged_totals.copy(),
                       "bookers": bookers}
    out["by_lob"] = {"bookings": bookings_by_lob,
                     "take_per_booking": take_per_booking,
                     "take_usd": take_usd_by_lob}
    return s, out

def run_sim(config_path: str):
    p, ug = load_params(config_path)
    s = derive_initial_state(p, ug)
    rows: List[Dict[str, Any]] = []
    for w in range(1, p.weeks + 1):
        s, out = step_week(p, ug, s, week_index_zero_based=w-1)
        rows.append({
            "week": w,
            **{f"active_{g}": out["by_group"]["active"][g] for g in GROUPS},
            **{f"engaged_{g}": out["by_group"]["engaged"][g] for g in GROUPS},
            **{f"bookers_{g}": out["by_group"]["bookers"][g] for g in GROUPS},
            **{f"bookings_{lob}": out["by_lob"]["bookings"][lob] for lob in LOBs},
            **{f"take_usd_{lob}": out["by_lob"]["take_usd"][lob] for lob in LOBs},
            "bookings_total": out["metrics"]["bookings_total"],
            "take_usd_total": sum(out["by_lob"]["take_usd"].values())
        })
    return rows

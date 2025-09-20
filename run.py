
import argparse
import os
import pandas as pd
import yaml
from model import run_sim, load_params

def main():
    rows = run_sim('config.yaml')
    df = pd.DataFrame(rows)
    out_path = 'simulation_output.csv'
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path}')

    # Set pandas display options for nicer formatting
    pd.options.display.float_format = '{:,.2f}'.format

    # Print onboarding graduation fraction (p_grad) derived from cohort/retention
    try:
        params = load_params('config.yaml')
        print(f"p_grad (onboarding graduation fraction after ~90d): {params.p_grad:.4f}")
    except Exception as e:
        print(f"Could not load p_grad: {e}")

    print("\nTotals from simulation_output.csv:")
    # The 'week' column is not a metric to be summed, so we drop it.
    #print(df.drop(columns=['week']).sum())
    rate_cols = [c for c in df.columns if c.startswith('churn_rate')]
    count_cols = [c for c in df.columns if c not in rate_cols + ['week']]
    print("Totals (counts/values):\n", df[count_cols].sum(numeric_only=True).apply('{:,.0f}'.format))
    # Churn rate means as percentages
    rate_means = df[rate_cols].mean(numeric_only=True)
    churn_rate_means = (rate_means * 100.0).map(lambda x: f"{x:.2f}%")
    print("Means (rates as %):\n", churn_rate_means)

        # Estimated booker conversion rates over the run (bookers / active)
    tot_active_p = df['active_partners'].sum()
    tot_active_np = df['active_non_partners'].sum()
    conv_p = (df['bookers_partners'].sum() / tot_active_p) if tot_active_p else 0.0
    conv_np = (df['bookers_non_partners'].sum() / tot_active_np) if tot_active_np else 0.0

    print("\nEstimated booker conversion rate(s) per week:")
    if abs(conv_p - conv_np) < 1e-12:
        print(f"  shared p_book: {conv_p:,.6f}")
    else:
        print(f"  partners:     {conv_p:,.6f}")
        print(f"  non_partners: {conv_np:,.6f}")

    # Print total_take
    total_take = df['take_usd_total'].sum()
    print(f"\nTotal take_usd_total: ${total_take:,.0f}")

    # If baseline_total_take_usd provided in config, print delta
    try:
        with open('config.yaml', 'r') as _f:
            _cfg = yaml.safe_load(_f) or {}
        _baseline = float(_cfg.get('baseline_total_take_usd', 0) or 0)
    except Exception:
        _baseline = 0.0
    if _baseline > 0:
        _delta = float(total_take) - _baseline
        _pct = (_delta / _baseline) * 100.0
        print(f"Vs baseline ${_baseline:,.2f}: Δ ${_delta:,.2f} ({_pct:+.2f}%)")

    # Print avg new_users/week
    avg_new_users = (df['new_accounts_partners'] + df['new_accounts_non_partners']).mean()
    print(f"Avg new users/week: {avg_new_users:,.0f}")

    # Print avg churn/week (if churn column exists)
    if 'churned_total' in df.columns:
        avg_churn = df['churned_total'].mean()
    elif {'churned_partners','churned_non_partners'}.issubset(df.columns):
        avg_churn = (df['churned_partners'] + df['churned_non_partners']).mean()
    else:
        avg_churn = float('nan')
    print(f"Avg mature churn/week: {avg_churn:,.0f}")



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--diagnose', action='store_true', help='Print calibration/data diagnostics and exit')
    ap.add_argument('--mode', type=str, help='Override calibration.mode')
    ap.add_argument('--source', type=str, help='Override calibration.bookings_csv_path')
    ap.add_argument('--schema', type=str, help='Override calibration.schema (auto|transactions|unique_bookers)')
    args = ap.parse_args()

    if args.mode or args.source or args.schema:
        # Shallow override by editing an in-memory copy of config and writing back
        import yaml, os
        with open('config.yaml','r') as f:
            cfg = yaml.safe_load(f) or {{}}
        cal = cfg.get('calibration', {{}}) or {{}}
        if args.mode:
            cal['mode'] = args.mode
        if args.source:
            cal['bookings_csv_path'] = args.source
        if args.schema:
            cal['schema'] = args.schema
        cfg['calibration'] = cal
        with open('config.yaml','w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    
if args.diagnose:
        # Print schema detection, required columns, rows ignored by gates, weeks used, WAU snapshot
        import pandas as pd, os
        from model import load_params
        try:
            p = load_params('config.yaml')
            print("Diagnosis for calibration:")
            print(f"  mode: {p.calibration_mode}")
            print(f"  schema_hint: {getattr(p,'calibration_schema_hint','auto')}")
            print(f"  source: {p.bookings_csv_path}")
            print(f"  weeks_to_use: {p.weeks_to_use}")
            print(f"  flights_gate: {p.flights_gate}")
            print(f"  WAU (partners, non_partners): {p.initial_wau_by_group.get('partners'):.0f}, {p.initial_wau_by_group.get('non_partners'):.0f}")
            if p.bookings_csv_path and os.path.exists(p.bookings_csv_path):
                df = pd.read_csv(p.bookings_csv_path)
                cols = [str(c).lower() for c in df.columns]
                print(f"  columns: {sorted(set(cols))}")
                has_unique = {'weekly_unique_bookers_est','lob'}.issubset(set(cols)) and (len({'week','week_number'} & set(cols))>0) and ('user_type' in cols)
                has_tx = any(c in cols for c in ['flights','hotels','cars','cruises']) and ('week' in cols)
                detected = 'unique_bookers' if has_unique else ('transactions' if has_tx else 'unknown')
                print(f"  detected_schema: {detected}")
                if has_unique and not p.flights_gate.get('non_partners', False):
                    try:
                        drop = df[(df['lob'].str.lower()=='flights') & (df['user_type'].str.lower().str.contains('non'))]
                        print(f"  note: would ignore {len(drop)} non-partner flight rows due to flights_gate")
                    except Exception:
                        pass
            else:
                print("  (no source file available)")
        except Exception as e:
            print("Diagnosis failed:", e)
else:
        main()

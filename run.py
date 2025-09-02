import pandas as pd
import yaml
from model import run_sim

def main():
    rows = run_sim('config.yaml')
    df = pd.DataFrame(rows)
    out_path = 'simulation_output.csv'
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path}')

    print("\nTotals from simulation_output.csv:")
    # The 'week' column is not a metric to be summed, so we drop it.
    print(df.drop(columns=['week']).sum())
        # Estimated booker conversion rates over the run (bookers / active)
    tot_active_p = df['active_partners'].sum()
    tot_active_np = df['active_non_partners'].sum()
    conv_p = (df['bookers_partners'].sum() / tot_active_p) if tot_active_p else 0.0
    conv_np = (df['bookers_non_partners'].sum() / tot_active_np) if tot_active_np else 0.0

    print("\nEstimated booker conversion rate(s) per week:")
    if abs(conv_p - conv_np) < 1e-12:
        print(f"  shared p_book: {conv_p:.6f}")
    else:
        print(f"  partners:     {conv_p:.6f}")
        print(f"  non_partners: {conv_np:.6f}")

    # Print total_take
    total_take = df['take_usd_total'].sum()
    print(f"\nTotal take_usd_total: {total_take:,.2f}")

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
    print(f"Avg new users/week: {avg_new_users:.2f}")

    # Print avg churn/week (if churn column exists)
    churn_cols = [col for col in df.columns if 'churn' in col.lower()]
    if churn_cols:
        avg_churn = df[churn_cols].sum(axis=1).mean()
        print(f"Avg churn/week: {avg_churn:.2f}")
    else:
        print("Avg churn/week: [No churn column found]")


if __name__ == '__main__':
    main()

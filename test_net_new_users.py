import pandas as pd
from pathlib import Path

"""
Simple diagnostic script to compute weekly and cumulative net new users from the
simulation output (new_accounts_partners + new_accounts_non_partners) minus
(churned_partners + churned_non_partners).

Outputs:
 - Prints first few weekly rows
 - Prints aggregate totals for new, churn, and net new
 - Saves a small CSV `net_new_users_summary.csv` with week-level metrics
"""

def main():
    csv_path = Path('simulation_output.csv')
    if not csv_path.exists():
        raise SystemExit('simulation_output.csv not found. Run run.py first.')

    df = pd.read_csv(csv_path)

    required_cols = {
        'week',
        'new_accounts_partners','new_accounts_non_partners',
        'churned_partners','churned_non_partners'
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f'Missing required columns in simulation_output.csv: {missing}')

    df['new_users_total'] = df['new_accounts_partners'] + df['new_accounts_non_partners']
    df['churned_total_calc'] = df['churned_partners'] + df['churned_non_partners']

    # Net new for each week
    df['net_new_users'] = df['new_users_total'] - df['churned_total_calc']
    df['cumulative_net_new'] = df['net_new_users'].cumsum()

    summary_cols = [
        'week','new_accounts_partners','new_accounts_non_partners',
        'new_users_total','churned_partners','churned_non_partners',
        'churned_total_calc','net_new_users','cumulative_net_new'
    ]

    print('\nWeekly net new users (head):')
    print(df[summary_cols].head(10).to_string(index=False, formatters={
        'net_new_users': '{:,.2f}'.format,
        'cumulative_net_new': '{:,.2f}'.format
    }))

    total_new = df['new_users_total'].sum()
    total_churn = df['churned_total_calc'].sum()
    total_net = df['net_new_users'].sum()

    print('\nTotals over period:')
    print(f'  Total new users:   {total_new:,.0f}')
    print(f'  Total churned:     {total_churn:,.0f}')
    print(f'  Net new users:     {total_net:,.0f}')
    print(f'  Final cumulative net new: {df["cumulative_net_new"].iloc[-1]:,.0f}')

    out_csv = Path('net_new_users_summary.csv')
    df[summary_cols].to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')

if __name__ == '__main__':
    main()

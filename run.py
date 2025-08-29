
import pandas as pd
from model import run_sim

def main():
    rows = run_sim('config.yaml')
    df = pd.DataFrame(rows, columns=[
        'week',
        'active_partners','active_non_partners',
        'bookers_partners','bookers_non_partners',
        'bookings_flights','bookings_hotels','bookings_cars','bookings_cruises',
        'take_usd_flights','take_usd_hotels','take_usd_cars','take_usd_cruises',
        'bookings_total','take_usd_total',
        'new_accounts_partners','new_accounts_non_partners'
    ])
    out_path = 'simulation_output.csv'
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path}')

    print("\nTotals from simulation_output.csv:")
    # The 'week' column is not a metric to be summed, so we drop it.
    print(df.drop(columns=['week']).sum())


if __name__ == '__main__':
    main()
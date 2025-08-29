import pandas as pd
from model import run_sim

if __name__ == '__main__':
    rows = run_sim('config.yaml')
    pd.DataFrame(rows).to_csv('simulation_output.csv', index=False)
    print('Wrote simulation_output.csv')

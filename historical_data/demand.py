import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from historical_data.series_processing import get_time
from stocha_processes.gaussian_markov_process import create_process, plot_process


if __name__ == "__main__":
    df = pd.read_csv('raw_data/total_load_DA_forecast_actual.csv').head(-1)

    # positive if short
    df['Imbalance [MW]'] = df['Actual Total Load [MW] - BZN|DE-LU'] - df[
        'Day-ahead Total Load Forecast [MW] - BZN|DE-LU']
    df['Imbalance [MW]'] = df['Imbalance [MW]'].fillna(df['Imbalance [MW]'].mean())

    # quarter of the product
    time_info = np.array(list(map(get_time, df['Time (CET)'])))
    df['Quarter'] = time_info[:, 2]

    plt.plot(df['Imbalance [MW]'])

    df.to_csv('processed_data/load_imbalance_quarter.csv')
    df.head()

    mu_0, sigma_0, mu_q, sigma_q, a_q, b_q = create_process(df, 'Imbalance [MW]', 'Quarter')
    plot_process(mu_0, sigma_0, 'Quarters', 'Imbalance shortage [MW/quarter] (marginal)')
    plot_process(mu_q, sigma_q, 'Quarters', 'Imbalance shortage [MW/quarter] (transition)')

    plt.show()

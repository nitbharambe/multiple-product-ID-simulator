import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from historical_data.series_processing import get_time
from stocha_processes.gaussian_markov_process import create_process, plot_process


if __name__ == "__main__":
    df = pd.read_csv('raw_data/day_ahead_prices.csv').head(-1)

    # quarter of the product
    time_info = np.array(list(map(get_time, df['MTU (CET)'])))
    df['Hours'] = time_info[:, 3]

    plt.plot(df['Day-ahead Price [EUR/MWh]'])

    df.to_csv('processed_data/day_ahead_prices.csv')
    df.head()

    mu_0, sigma_0, mu_q, sigma_q, a_q, b_q = create_process(df, 'Day-ahead Price [EUR/MWh]', 'Hours')
    plot_process(mu_0, sigma_0, 'Hours', 'Day-ahead Price [EUR/MWh] (marginal)')
    plot_process(mu_q, sigma_q, 'Hours', 'Day-ahead Price [EUR/MWh] (transition)')

    plt.show()

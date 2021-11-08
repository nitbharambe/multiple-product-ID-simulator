import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def product_plot(exp_path, NordPool_all):
    data_trader, data_p_no, data_time, data_volume = [], [], [], []
    for p_no, product in NordPool_all.products.items():
        for id, trader in product.traders.items():
            for t in range(len(product.trading_horizon) - 1):
                data_trader.append(id)
                data_p_no.append(p_no)
                data_time.append(product.trading_horizon[t + 1])
                data_volume.append(trader._cum_position[t + 1] - trader._cum_position[t])
    data_heatmap = pd.DataFrame({'trader': data_trader, 'p_no': data_p_no, 'volume': data_volume, 'time': data_time})
    data_heatmap.replace(np.nan, 0)
    data_heatmap['p_no'] = data_heatmap['p_no'] + 1
    for id, items in NordPool_all.products[0].traders.items():
        pivoted = data_heatmap[data_heatmap['trader'] == id].pivot(index="p_no", columns="time", values="volume")
        snsax = sns.heatmap(pivoted, center=0,  cmap="YlGnBu")
        snsax.set_title("%s Traded Volume" % (id))
        snsax.set_ylabel("Product number")
        snsax.set_xlabel("Time")
        snsax.set_xticklabels(data_heatmap['time'].dt.strftime('%H'))
        plt.savefig("%s/a_%s_%s%s" % (exp_path, "heatmap", id, ".png"), dpi=500, bbox_inches='tight')
        plt.show()


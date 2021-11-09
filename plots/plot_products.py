import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def product_plot(exp_path, NordPool_all):
    #sel_plot = '_cum_positions'
    sel_plot = 'all_positions'
    data_trader, data_p_no, data_time, data_volume = [], [], [], []
    # Position plot method selection: From all_positions or _cum_positions of traders
    if sel_plot == '_cum_positions':
        for p_no, product in NordPool_all.products.items():
            for id, trader in product.traders.items():
                for t in range(len(product.trading_horizon) - 1):
                    data_trader.append(id)
                    data_p_no.append(p_no)
                    data_time.append(product.trading_horizon[t])
                    data_volume.append(trader._cum_position[t + 1] - trader._cum_position[t])
    else:
        for t in NordPool_all.all_positions.keys():
            if t > NordPool_all.products[0].trading_horizon[0]:
                for id, items in NordPool_all.products[0].traders.items():
                    for p_no in range(len(NordPool_all.products)):
                        if t in NordPool_all.products[p_no].trading_horizon:
                            data_trader.append(id)
                            data_p_no.append(p_no)
                            data_time.append(t)
                            data_volume.append(NordPool_all.all_positions[t][id][p_no] - NordPool_all.all_positions[t-t.freq][id][p_no])
    data_heatmap = pd.DataFrame({'trader': data_trader, 'p_no': data_p_no, 'volume': data_volume, 'time': data_time})
    data_heatmap.replace(np.nan, 0)
    data_heatmap['p_no'] = data_heatmap['p_no'] + 1
    data_heatmap['time'] = pd.to_datetime(data_heatmap['time'])
    # Saving DF for quicker plots
    #data_heatmap.to_pickle("C:/Users/nitbh/OneDrive/Documents/repository/intraday-market-simulator-master/results/heatmapdata/data_heatmap.pkl")

    for id, items in NordPool_all.products[0].traders.items():
        pivoted = data_heatmap[data_heatmap['trader'] == id].pivot(index="p_no", columns="time", values="volume")
        plt.figure(figsize=(15,8))
        snsax = sns.heatmap(pivoted, center=0)
        snsax.set_title("%s Traded Volume" % (id))
        snsax.set_ylabel("Product Number")
        snsax.set_xlabel("Time")
        snsax.set_xticklabels(pivoted.columns.strftime('%H:%M'))
        plt.savefig("%s/a_%s_%s%s" % (exp_path, "heatmap", id, ".png"), dpi=500, bbox_inches='tight')
        #plt.show()


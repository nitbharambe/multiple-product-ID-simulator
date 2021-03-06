import json

from agents.Market_operator import MarketOperator
from agents.Market_operator import allmarket
from plots.plot_simulation import Plotter
from utils import get_exp_path, save_configs, load_configs, parse_args, set_up_logger
from matplotlib import pyplot as plt
from plots.plot_products import product_plot

if __name__ == "__main__":
    plt.close('all')
    args = parse_args()
    exp_path = get_exp_path(**args)
    config_traders, config_market_operator = load_configs(**args)
    save_configs(config_traders, config_market_operator, exp_path)
    set_up_logger(exp_path, args)

    NordPool_all = allmarket(config_traders, config_market_operator, exp_path, no_of_products=24)
    product_plot(exp_path, NordPool_all)

    plt.close('all')
    print('breakpoint')
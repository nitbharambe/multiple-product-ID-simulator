import os
import datetime 

import matplotlib.pyplot as plt
import numpy as np

from agents.Consumer import Consumer
from agents.RES_agent import RES
from agents.Storage import Storage
from agents.Conventional import Conventional
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates


class Plotter(object):

    def __init__(self, market_operator, path, show_plots):
        self.equilibriums = market_operator.equilibriums
        self.order_book = market_operator.order_book
        self.order_book_data = market_operator.order_book_data
        self.traders = market_operator.traders
        self.trading_horizon = market_operator.trading_horizon
        self.path = "%s/%s" % (path, 'plots')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.up_lim = market_operator.up_regulation_price+20
        self.down_lim = market_operator.dn_regulation_price-20
        self.show_plots = show_plots

        self.product_number = market_operator.product_number
        self.figdpi = 500
        fontP = FontProperties()
        fontP.set_size('xx-small')
        self.xformatter = mdates.DateFormatter('%H')


    def plot_positions(self):
        plt.figure()
        ax = plt.subplot(111)
        ax.set_title("Market Positions"+'_'+str(self.product_number))
        ax.set_xlabel("Time (hrs)")
        x_data = self.trading_horizon
        ax.xaxis.set_major_formatter(self.xformatter)
        ax.set_xlim([self.trading_horizon[0] - datetime.timedelta(hours=1), self.trading_horizon[-1] + datetime.timedelta(hours=1)])
        for id, traders in self.traders.items():
            ax.plot(x_data,traders._cum_position, label=id)
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        if self.show_plots:plt.show()
        plt.savefig("%s/%s_%s%s" % (self.path, "positions", self.product_number, ".png"), dpi=self.figdpi, bbox_inches='tight')

    def plot_variable_positions(self):
        for id, traders in self.traders.items():
            if isinstance(traders, RES) or isinstance(traders, Consumer):
                plt.figure()
                ax = plt.subplot(211)
                ax.set_title(id+('_')+str(self.product_number))
                ax.set_ylabel("Volume (MWh)")
                ax.plot(self.trading_horizon, traders._cum_position, label="Position")
                ax.plot(self.trading_horizon, traders.forecasts, label="Forecast")
                ax.axhline(traders.realization, color='r', linestyle="--", label="Realization")
                # ax.plot(self.trading_horizon, [traders.realization - traders.error_range(time) for time in
                #                                self.trading_horizon], label="Error lower bound")
                # ax.plot(self.trading_horizon, [traders.realization + traders.error_range(time) for time in
                #                                self.trading_horizon], label="Error upper bound")

                ax1 = plt.subplot(212)
                ax1.set_xlabel("Time (min)")
                ax1.set_ylabel("Price (€)")
                ax1.plot(self.trading_horizon, traders.limits_buy, label="Limit_buy")
                ax1.plot(self.trading_horizon, traders.limits_sell, label="Limit_sell")
                ax1.scatter(
                    [traders.submitted_orders[i]['timestamp'] for i in range(len(traders.submitted_orders)) if traders.submitted_orders[i]['side'].name == "ASK"],
                    [traders.submitted_orders[i]['price'] for i in range(len(traders.submitted_orders)) if traders.submitted_orders[i]['side'].name == "ASK"],
                    c="green", label="Sell")

                ax1.scatter(
                    [traders.submitted_orders[i]['timestamp'] for i in range(len(traders.submitted_orders)) if traders.submitted_orders[i]['side'].name != "ASK"],
                    [traders.submitted_orders[i]['price'] for i in range(len(traders.submitted_orders)) if traders.submitted_orders[i]['side'].name != "ASK"],
                    c="red", label="Buy")
                ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
                ax.set_xticklabels([])
                ax1.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
                #ax1.set_ylim([self.down_lim, self.up_lim])
                ax1.xaxis.set_major_formatter(self.xformatter)
                ax.xaxis.set_major_formatter(self.xformatter)
                plt.savefig("%s/%s_%s_%s%s" % (self.path, id, "private_info",self.product_number, ".png"), dpi=self.figdpi, bbox_inches='tight')
            elif isinstance(traders, Storage):
                plt.figure()

                ax = plt.subplot(211)
                ax.set_title(id+('_')+str(self.product_number))

                ax.set_ylabel("Volume (MWh)")
                ax.plot(self.trading_horizon, traders._cum_position, label="Position")
                ax.axhline(traders.capacity, color='r', linestyle="--", label="Capacity")
                ax1 = plt.subplot(212)
                ax1.set_xlabel("Time (min)")
                ax1.set_ylabel("Price (€)")
                ax1.plot(self.trading_horizon, traders.limits_buy, label="Limit_buy")
                ax1.plot(self.trading_horizon, traders.limits_sell, label="Limit_sell")
                ax1.scatter(
                    [traders.submitted_orders[i]['timestamp'] for i in range(len(traders.submitted_orders)) if
                     traders.submitted_orders[i]['side'].name == "ASK"],
                    [traders.submitted_orders[i]['price'] for i in range(len(traders.submitted_orders)) if
                     traders.submitted_orders[i]['side'].name == "ASK"],
                    c="green", label="Sell")

                ax1.scatter(
                    [traders.submitted_orders[i]['timestamp'] for i in range(len(traders.submitted_orders)) if
                     traders.submitted_orders[i]['side'].name != "ASK"],
                    [traders.submitted_orders[i]['price'] for i in range(len(traders.submitted_orders)) if
                     traders.submitted_orders[i]['side'].name != "ASK"],
                    c="red", label="Buy")
                ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
                ax.set_xticklabels([])
                ax1.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
                #ax1.set_ylim([self.down_lim, self.up_lim])
                ax.xaxis.set_major_formatter(self.xformatter)
                ax1.xaxis.set_major_formatter(self.xformatter)
                plt.savefig("%s/%s_%s_%s%s" % (self.path, id, "private_info", self.product_number, ".png"), dpi=self.figdpi, bbox_inches='tight')
            else:
                plt.figure()

                ax = plt.subplot(211)
                ax.set_title(id+('_')+str(self.product_number))

                ax.set_ylabel("Volume (MWh)")
                ax.plot(self.trading_horizon, traders._cum_position, label="Position")
                ax.axhline(traders.capacity, color='r', linestyle="--", label="Capacity")
                ax.axhline(traders.min_stable_load, color='k', linestyle="--", label="min_stable_load")
                try:
                    ax.axvline(traders.ramp_active_timeline[0], linestyle="--", label="ramp_active")
                except:
                    pass
                ax1 = plt.subplot(212)
                ax1.set_xlabel("Time (min)")
                ax1.set_ylabel("Price (€)")
                ax1.plot(self.trading_horizon, traders.limits_buy, label="Limit_buy")
                ax1.plot(self.trading_horizon, traders.limits_sell, label="Limit_sell")
                ax1.scatter(
                    [traders.submitted_orders[i]['timestamp'] for i in range(len(traders.submitted_orders)) if
                     traders.submitted_orders[i]['side'].name == "ASK"],
                    [traders.submitted_orders[i]['price'] for i in range(len(traders.submitted_orders)) if
                     traders.submitted_orders[i]['side'].name == "ASK"],
                    c="green", label="Sell")

                ax1.scatter(
                    [traders.submitted_orders[i]['timestamp'] for i in range(len(traders.submitted_orders)) if
                     traders.submitted_orders[i]['side'].name != "ASK"],
                    [traders.submitted_orders[i]['price'] for i in range(len(traders.submitted_orders)) if
                     traders.submitted_orders[i]['side'].name != "ASK"],
                    c="red", label="Buy")
                ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
                ax.set_xticklabels([])
                ax1.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
                #ax1.set_ylim([self.down_lim, self.up_lim])
                ax.xaxis.set_major_formatter(self.xformatter)
                ax1.xaxis.set_major_formatter(self.xformatter)
                plt.savefig("%s/%s_%s_%s%s" % (self.path, id, "private_info", self.product_number, ".png"), dpi=self.figdpi, bbox_inches='tight')


            plt.figure()

            ax = plt.subplot(211)
            ax.set_title(id+('_')+str(self.product_number))

            ax.plot(traders.aggressivenesses_buy, label="Aggressiveness_buy")
            ax.plot(traders.aggressivenesses_sell, label="Aggressiveness_sell")
            ax1 = plt.subplot(212)
            ax1.set_xlabel("Market event")
            ax1.set_ylabel("Price (€)")
            ax1.plot(traders.eqlbm_prices, label="Eqlbm_price")
            ax1.plot(traders.targets_buy, label="Target_buy")
            ax1.plot(traders.targets_sell, label="Target_sell")
            ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
            ax1.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
            ax1.set_ylim([self.down_lim, self.up_lim])
            plt.savefig("%s/%s_%s_%s%s" % (self.path, id, "traderAA_params", self.product_number, ".png"), dpi=self.figdpi, bbox_inches='tight')
            ax.set_xticklabels([])
        if self.show_plots:plt.show()

    def plot_market_evolution(self):
        plt.figure()
        plt.title("Market Evolution")
        plt.ylabel("Price (€)")
        plt.xlabel("Market event")
        plt.plot(self.order_book_data["best_ask"], label="best_ask")
        plt.plot(self.order_book_data["best_bid"], label="best_bid")
        #plt.plot(self.order_book_data["vwap_ask"], label="vwap_ask")
        #plt.plot(self.order_book_data["vwap_buy"], label="vwap_buy")
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        plt.savefig("%s/%s_%s%s" % (self.path,  "market_evolution",self.product_number, ".png"), dpi=self.figdpi, bbox_inches='tight')

        plt.figure()
        ax = plt.subplot(111)
        ax.set_title("Transaction price")
        ax.set_ylabel("Price (€)")
        dates = [transaction["resting_timestamp"] for transaction in self.order_book.trade_book]
        ax.set_xlim([self.trading_horizon[0] - datetime.timedelta(hours=1), self.trading_horizon[-1] + datetime.timedelta(hours=1)])
        ax.scatter(dates, [transaction["price"] for transaction in self.order_book.trade_book])
        #plt.autofmt_xdate()
        plt.gcf().axes[0].xaxis.set_major_formatter(self.xformatter)
        try:
            ax.axvline(len(self.trading_horizon)*0.7, linestyle="--", label="ramp_active")
        except:
            pass
        plt.savefig("%s/%s_%s%s" % (self.path, "transaction_price",self.product_number, ".png"), dpi=self.figdpi, bbox_inches='tight')

        colors = {}
        for num, agent in enumerate(self.traders.keys()):
            colors[agent] = num

        self.order_book.order_history.pop(0)
        self.order_book.order_history.pop(0)

        # prices = [i["price"] for i in self.order_book.order_history if i["type"] == 1]
        # times = [i["timestamp"] for i in self.order_book.order_history if i["type"] == 1]
        # agents = [i["trader_id"] for i in self.order_book.order_history if i["type"] == 1]

        prices = [i["price"] for i in self.order_book.order_history]
        times = [i["timestamp"] for i in self.order_book.order_history]
        agents = [i["trader_id"] for i in self.order_book.order_history]

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_title("Submitted Orders")
        ax.set_ylabel("Price (€)")
        plt.xlabel("Market event")
        self.order_book.order_history.pop(0)
        self.order_book.order_history.pop(0)

        scatter = ax.scatter(range(len(prices)), prices, c=[colors[a] for a in agents])
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="best", title="Agents")
        ax.plot(self.order_book_data["best_ask"], label="best_ask")
        ax.plot(self.order_book_data["best_bid"], label="best_bid")
        ax.add_artist(legend1)
        ax.set_ylim([self.down_lim, self.up_lim])

        plt.savefig("%s/%s_%s%s" % (self.path, "orders_sumbitted", self.product_number, ".png"), dpi=self.figdpi, bbox_inches='tight')

    def plot_order_book(self):
        prices_sell = np.array([key for key in sorted(self.order_book._ask_book.keys(), reverse=False) if
                                self.order_book._ask_book[key]["size"] > 0])

        prices_buy = np.array([key for key in sorted(self.order_book._bid_book.keys(), reverse=True) if
                               self.order_book._bid_book[key]["size"] > 0])

        volumes_sell = np.cumsum(np.array(
            [self.order_book._ask_book[key]["size"] for key in
             sorted(self.order_book._ask_book.keys(), reverse=False)
             if self.order_book._ask_book[key]["size"] > 0]))

        volumes_buy = np.cumsum(np.array(
            [self.order_book._bid_book[key]["size"] for key in
             sorted(self.order_book._bid_book.keys(), reverse=True) if
             self.order_book._bid_book[key]["size"] > 0]))

        plt.plot(prices_sell, volumes_sell)
        plt.plot(prices_buy, volumes_buy)
        plt.ylim([self.down_lim, self.up_lim])

        plt.savefig("%s/%s_%s%s" % (self.path, "order_book",self.product_number, ".png"), dpi=self.figdpi, bbox_inches='tight')

    def plot_equilibrium(self):
        for t, curves in self.equilibriums.items():
            limits_buy = curves["buy"]
            limits_sell = curves["sell"]
            plt.figure()
            plt.title(t)
            plt.step(np.cumsum([limits_sell[i] for i in sorted(limits_sell, reverse=False)]),
                     [i for i in sorted(limits_sell, reverse=False)], where="pre")
            plt.step(np.cumsum([limits_buy[i] for i in sorted(limits_buy, reverse=True)]),
                     [i for i in sorted(limits_buy, reverse=True)], where="pre")

    def plot_imbalance(self):
        plt.figure()
        plt.title("Imbalances" + '_' + str(self.product_number))
        ax = plt.subplot(211)
        for id, traders in self.traders.items():
            ax.plot(traders.all_imbalances, label=id)
        try:
            ax.axvline(len(self.all_imbalances)*0.4, linestyle="--", label="ramp_active")
        except:
            pass
        ax1 = plt.subplot(212)
        for id, traders in self.traders.items():
            if isinstance(traders, Conventional) or isinstance(traders, Storage):
                ax1.plot(traders.all_ramp_down_margin, label=id+'_dn')
                ax1.plot(traders.all_ramp_up_margin, label=id+'_up')

        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        ax1.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        if self.show_plots:plt.show()
        plt.savefig("%s/%s_%s%s" % (self.path, "imbalances", self.product_number, ".png"), dpi=self.figdpi, bbox_inches='tight')

    def prices_table(self):
        [transaction["price"] for transaction in self.order_book.trade_book]

    def generate_plots(self):
        # self.plot_equilibrium()
        self.plot_positions()
        self.plot_variable_positions()
        # self.plot_market_evolution()
        #self.plot_order_book()
        self.plot_imbalance()


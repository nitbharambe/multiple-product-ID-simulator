import logging
import random

from scipy.optimize import bisect

import pandas as pd
import numpy as np
from collections import defaultdict
from plots.plot_simulation import Plotter
import matplotlib.pyplot as plt

from mmabm.orderbook import Orderbook
from mmabm.shared import Side, OType
import mmabm.trader as trader
from agents import *

AGENT_TYPES = {RES.name(): RES, Consumer.name(): Consumer, Conventional.name(): Conventional,
               Storage.name(): Storage}


def fix_random_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)



class MarketOperator(object):

    def __init__(self, config_traders, config_market_operator, path, product_number=0):
        """
        Initialize the market operator

        :param config_traders: dict with the market participants configuration params
        :param config_market_operator: dict with the market operator configuration params
        """
        self.path = path
        self.config_traders = config_traders
        for key in config_market_operator:
            setattr(self, key, config_market_operator[key])

        self.product_number = product_number
        self.start_trade = pd.to_datetime(self.start_trade)
        self.top_of_book = dict()
        # self.all_positions = dict() TODO: remove
        self.end_trade = pd.to_datetime(self.end_trade)
        if self.product_number > 0:
        #if self.end_trade < self.start_trade:
            self.end_trade += pd.to_timedelta('1D')
        self.time_step = pd.to_timedelta(self.time_step)
        # TODO Fix end trade
        self.trading_horizon = pd.date_range(self.start_trade, self.end_trade,
                                             freq=self.time_step)
        if self.random_seed is not None: fix_random_seeds(self.random_seed)
        self.order_book = None
        self.traders = None
        self.order_book_data = None
        self.imbalance_penalty = None
        self.equilibriums = None
        self.regulation = None
        self.counter = 0

        self.reset()

    @staticmethod
    def name():
        return "MarketOperator"

    def reset(self):
        """
        Initilize the market setup
        :return: dict with the top of the book (bid, ask price and volumes)
        """
        self.order_book = Orderbook()  # instantiate the orderbook object
        self.traders = dict()
        self.buildTraders()  # instantiate the traders
        self.init_regulation_info()
        self.order_book.aon_traders = [id for id,trader in self.traders.items() if trader.aon_trader]
        self.seedOrderbook()
        self.order_book_data = {"best_ask": [], "best_bid": [], "vwap_ask": [],
                                "vwap_buy": []}  # Keep trach of the orderbook data
        self.imbalance_penalty = dict()
        self.equilibriums = dict()  # Keep track of the transactionshappened
        top_of_book = self.get_tob(self.trading_horizon[0])  # Get the initial top of the book (tob)
        self.initialize_agents_beliefs(top_of_book)

        return top_of_book

    def get_system_regulation(self, sign_imb):
        """
        Predefined system imbalance for the end of the trading session
        :return: sign of the system imbalance
        """
        k = sign_imb * self.influence_imbalance / 2.

        if self.system_regulation.upper() == "RANDOM":
            return np.random.choice([1, -1], p=[0.5 + k, 0.5 - k])
        elif self.system_regulation.upper() == "UP":
            return -1
        else:
            return 1

    def seedOrderbook(self):
        """
        Define the initial values for the order book data.
        :return: N/A
        """
        seed_provider = trader.Provider(9999, 1, 0.05, 0.025)
        ba = random.choice(range(200, 300, 5))
        bb = random.choice(range(0, 10, 5))
        qask = {'order_id': 1, 'trader_id': 9999, 'timestamp': 0, 'type': OType.ADD,
                'quantity': 1, 'side': Side.ASK, 'price': ba}
        qbid = {'order_id': 2, 'trader_id': 9999, 'timestamp': 0, 'type': OType.ADD,
                'quantity': 1, 'side': Side.BID, 'price': bb}
        seed_provider.local_book[1] = qask
        self.order_book.add_order_to_book(qask)
        self.order_book.add_order_to_history(qask)
        seed_provider.local_book[2] = qbid
        self.order_book.add_order_to_book(qbid)
        self.order_book.add_order_to_history(qbid)

    def buildTraders(self):
        """
        Instantiate the traders based on the config file
        :return: N/A
        """
        for agent_type, trade_ids in self.config_traders.items():
            if trade_ids:
                agent = AGENT_TYPES[agent_type]
                for i, characteristics in trade_ids.items():
                    self.traders[i] = agent(ID=i, trading_horizon=self.trading_horizon, product_number = self.product_number, **characteristics)

    def runMarket(self, current_time, all_positions):
        """
        Main function that simulates the trading session
        :return:N/A
        """

        # start the trading session with a DA clearing
        # self.day_ahead_clearing()

        # intraday session
        if current_time == self.start_trade:
            self.top_of_book = self.get_tob(self.start_trade)  # Get the initial top of the book (tob)
        logging.info(str(self.top_of_book))
        # Update the traders beliefs based on the tob
        ids = list(self.traders.keys())
        random.shuffle(ids)  # Randomize the order in which the traders submit their orders
        for id in ids:  # Loo through the agents

            self.update_agents_beliefs(current_time, self.top_of_book, None)  # First update their beliefs
            self.traders[id].process_signal(current_time,
                                            self.top_of_book)  # Based on the tob and their belief: 1) cancel previous orders if any 2) submit new orders in the market

            self.top_of_book, rew = self.step((self.traders[id].cancel_collector, self.traders[id].quote_collector),
                                              current_time)
            logging.info(str(self.top_of_book))
            # Save the final  position of the agent after processing all the orders
            self.traders[id].store_cum_position()
            self.print_agent_results()
            self.compute_equilibrium(current_time,
                                     self.traders)  # TODO use this to compare the transaction prices and the actual equilibrium at each time step
            # print("       ",id, self.traders[id]._position, self.traders[id]._cash_flow)
        # self.store_result(before_imbalance_flag=True)      #remove for multiple
        # self.imbalance_settlement()  # Perform the final imbalance settlement
        # self.store_result(before_imbalance_flag=False)

    def step(self, action, current_time):

        cancel_collector, quote_collector = action[0], action[1]
        reward = 0.
        top_of_book = self.get_tob(current_time)
        for c in cancel_collector:
            logging.info("Cancelling order: %s" % c)

            top_of_book = self._step(c, current_time)  # Get the tob

        for q in quote_collector:  # Loop through the orders that the agent submitted and process them one by one
            logging.info("Submitted order: %s" % q)

            top_of_book = self._step(q, current_time)

            if self.order_book.traded:  # in case there is a match
                logging.info("Match:: %s" % self.order_book.trade_book[-1])
                reward += self.confirmTrades()  # Process the orders in the local order book of the agents engaged in the transaction

        return top_of_book, reward

    def _step(self, order, current_time):
        self.order_book.process_order(order)
        top_of_book = self.get_tob(current_time)

        self.save_tob_data(top_of_book)
        logging.info(top_of_book)

        self.update_agents_beliefs(current_time, top_of_book,
                                   transactions=self.order_book.trade_book if self.order_book.traded else None)
        return top_of_book

    def get_tob(self, t):
        """
        Get the top of the book (bid ask price volumes and vwaps)
        :param t: Time step in the trading horizon
        :return: tob
        """
        # TODO Find alternative for vwap calculation inside this function
        self.counter += 1
        return self.order_book.report_top_of_book(t)

    def imbalance_settlement(self):
        """
        Perform the final imbalance settlement based on the mechanism used
        :return: N/A
        """
        total_imb = 0.
        for id, trader in self.traders.items():
            trader.compute_final_imbalance()  # Check if system is up regulating
            total_imb += trader.final_imbalance
        self.regulation = self.get_system_regulation(np.sign(total_imb))  # Define the type of final imbalance

        if self.dual_pricing:
            for id, trader in self.traders.items():
                if isinstance(trader, Consumer):
                    self.imbalance_settlement_single(id, trader)
                else:
                    self.imbalance_settlement_dual(id, trader)

        else:
            for id, trader in self.traders.items():
                self.imbalance_settlement_single(id, trader)

        logging.info("Imbalance settlement")
        self.print_agent_results()

    def print_agent_results(self):
        logging.info("Trader - Position - Revenue -  Limitsbuy - limitssell")
        [logging.info("%s - %s - %s - %s - %s" % (id, trader._position, trader._cash_flow, trader.limits_buy, trader.limits_sell)) for id, trader in
         self.traders.items()]

    def compute_vwap(self):
        """
        Function that computes the volume weighted average price for each side at each time instant
        :return: vwap
        """

        return (sum(i * self.order_book._bid_book[i]["size"] for i in self.order_book._bid_book.keys()) + sum(
            i * self.order_book._ask_book[i]["size"] for i in self.order_book._ask_book.keys())) / (
                       sum(self.order_book._bid_book[i]["size"] for i in self.order_book._bid_book.keys()) + sum(
                   self.order_book._ask_book[i]["size"] for i in self.order_book._ask_book.keys()))

    def save_tob_data(self, top_of_book):
        """
        Record the tob
        :param top_of_book: top of the orderbook
        :return: N/A
        """
        self.order_book_data["best_ask"].append(top_of_book["best_ask"])
        self.order_book_data["best_bid"].append(top_of_book["best_bid"])
        self.order_book_data["vwap_ask"].append(top_of_book["vwap_ask"])
        self.order_book_data["vwap_buy"].append(top_of_book["vwap_buy"])

    def doCancels(self, trader):
        """
        Process the cancel orders for each trader
        :param trader: trader
        :return: N/A
        """
        for c in trader.cancel_collector:
            logging.info("Cancelling order:", c)
            self.order_book.process_order(c)

    def confirmTrades(self):
        """
        Process the matched trades and update locally the trades
        :return: N/A
        """
        for party in self.order_book.confirm_trade_collector:
            for side in party:
                r = self.traders[side['trader']].confirm_trade_local(side)
        return r

    def initialize_agents_beliefs(self, top_of_book):
        """
        Initialize the params of the traders
        :param top_of_book: top of the order book
        :return: N/A
        """
        for id, trader in self.traders.items():
            trader.trader_behavior.initialize(top_of_book)  # how does this go to trader_AA???

    def update_agents_beliefs(self, time, tob, transactions):
        """
        Update the params of the agents
        :param time: time step
        :param tob: top of the book
        :param transactions: previous transactions happened (price, volumes etc)
        :return: N/A
        """
        for id, trader in self.traders.items():
            logging.info("Update beliefs Agent: %s" % id)
            trader.trader_behavior.respond(time, tob, transactions, verbose=True)
            trader.store_aggressiveness()
            trader.store_eqlbm()
            trader.store_targets()
            if trader.trader_behavior.name() == "Trader_AA": logging.info(trader.trader_behavior.target_buy)
            if trader.trader_behavior.name() == "Trader_AA": logging.info(trader.trader_behavior.target_sell)
            if transactions:
                #for transaction in transactions:
                #    if transaction['incoming_trader_id'] == id:
                #        if transaction['side'] == Side.BID:
                #            trader.own_bid_transactions.append(transaction['price'])
                #        else:
                #            trader.own_ask_transactions.append(transaction['price'])
                #TODO Change 1 and 2 to Side.ASK/Side.BID
                trader.own_bid_transactions = [transaction['price'] for transaction in transactions
                                               if transaction['incoming_trader_id'] == id and transaction['side'] == 1]
                trader.own_ask_transactions = [transaction['price'] for transaction in transactions
                                               if transaction['incoming_trader_id'] == id and transaction['side'] == 2]


    def imbalance_settlement_dual(self, id, trader):
        """
        Compute the imbalances and the penalty applied to each agent under dual pricing mechanism
        :param id: ID of each trader
        :param trader: trader object
        :return: N/A
        """

        if self.regulation > 0:  # but is should be the same value of system regulation for all the agents!!
            if trader.final_imbalance > 0:
                self.imbalance_penalty[id] = trader.final_imbalance * self.up_regulation_price
                trader._cash_flow += self.imbalance_penalty[id]
            else:
                self.imbalance_penalty[id] = trader.final_imbalance * self.spot_price
                trader._cash_flow += self.imbalance_penalty[id]
        elif self.regulation < 0:
            if trader.final_imbalance > 0:
                self.imbalance_penalty[id] = trader.final_imbalance * self.dn_regulation_price
                trader._cash_flow += self.imbalance_penalty[id]
            else:
                self.imbalance_penalty[id] = trader.final_imbalance * self.spot_price
                trader._cash_flow += self.imbalance_penalty[id]
            # else if system did not up or down regulate then
        else:
            self.imbalance_penalty[id] = trader.final_imbalance * self.spot_price
            trader._cash_flow += self.imbalance_penalty[id]

    def imbalance_settlement_single(self, id, trader):
        """
        Compute the imbalances and the penalty of each agent under single pricing
        :param id: ID of the trader
        :param trader: trader object
        :return: N/A
        """
        if self.regulation > 0:
            self.imbalance_penalty[id] = trader.final_imbalance * self.up_regulation_price
            trader._cash_flow += self.imbalance_penalty[id]
        else:
            self.imbalance_penalty[id] = trader.final_imbalance * self.dn_regulation_price
            trader._cash_flow += self.imbalance_penalty[id]

    def compute_equilibrium(self, time, traders):
        """
        Compute the "theoretical" equilibrium. Its for analysis of the market
        :param time: timestep
        :param traders: trader objects
        :return: N/A
        """
        limits_buy = {}
        limits_sell = {}
        for name, tr in traders.items():
            limits_buy[tr.limit_price_buy] = tr.ramp_down_margin
            limits_sell[tr.limit_price_sell] = tr.ramp_up_margin

        self.equilibriums[time] = {"buy": limits_buy, "sell": limits_sell}

    # def day_ahead_clearing(self):
    #     da_demand_q = []
    #     da_demand_p = []
    #     da_demand_id = []
    #
    #     da_supply_q = []
    #     da_supply_p = []
    #     da_supply_id = []
    #
    #     # sort by agent type
    #     for id_agent, agent in self.traders.items():
    #         type_agent, quantity, price = agent.da_trade()
    #
    #         if type_agent is "DEMAND":
    #             da_demand_q.append(quantity)
    #             da_demand_p.append(price)
    #             da_demand_id.append(id_agent)
    #         else:
    #             da_supply_q.append(quantity)
    #             da_supply_p.append(price)
    #             da_supply_id.append(id_agent)
    #
    #     # build cumulative supply and demand
    #     demand_arg = np.argsort(da_demand_p)
    #     da_demand_q = np.cumsum(np.array(da_demand_q)[demand_arg])
    #     da_demand_p = np.array(da_demand_p)[demand_arg]
    #     da_demand_id = np.array(da_demand_id)[demand_arg]
    #     max_demand = da_demand_q[-1]
    #
    #     supply_arg = np.argsort(da_supply_p)
    #     da_supply_q = np.cumsum(np.array(da_supply_q)[supply_arg])
    #     da_supply_p = np.array(da_supply_p)[supply_arg]
    #     da_supply_id = np.array(da_supply_id)[supply_arg]
    #
    #     # find equilibrium:
    #     if da_supply_p[0] > da_demand_p[-1]:
    #         # no clearing
    #         equilibrium_q = 0
    #         equilibrium_p = None
    #     else:
    #         # clearing
    #         def linear_regression(q, da_q, da_p):
    #             id_q = np.searchsorted(da_q, q)
    #
    #             if q <= da_q[0]:
    #                 return da_p[0]
    #             elif q == da_q[-1]:
    #                 return da_p[-1]
    #             elif q > da_q[-1]:
    #                 return None
    #             else:
    #                 return (da_p[id_q - 1] - da_p[id_q]) * (q - da_q[id_q]) / \
    #                        (da_q[id_q - 1] - da_q[id_q]) + da_p[id_q]
    #
    #         def demand_function(q):
    #             return linear_regression(max_demand - q, da_demand_q, da_demand_p)
    #
    #         def supply_function(q):
    #             return linear_regression(q, da_supply_q, da_supply_p)
    #
    #         q_max = min(da_supply_q[-1], da_demand_q[-1])
    #
    #         if demand_function(q_max) <= supply_function(q_max):
    #             # explicit crossing
    #
    #             def objective(q):
    #                 return supply_function(q) - demand_function(q)
    #
    #             q_a = 0
    #             q_b = q_max
    #             equilibrium_q = bisect(objective, q_a, q_b)
    #             equilibrium_p = supply_function(equilibrium_q)
    #
    #         else:
    #             # no explicit crossing but clearing at max capacity
    #             equilibrium_q = q_max
    #             equilibrium_p = supply_function(equilibrium_q) + \
    #                             (demand_function(equilibrium_q) - supply_function(equilibrium_q)) / 2
    #
    #     logging.info('DA quantity cleared : ', equilibrium_q)
    #     logging.info('DA price : ', equilibrium_p)
    #
    #     # from the clearing price and equilibrium_q adapt the agents' available quantity
    #     for ag_q, ag_p, ag_id in zip((da_demand_q, da_supply_q),
    #                                  (da_demand_p, da_supply_p),
    #                                  (da_demand_id, da_supply_id)):
    #         quantity_supplied = 0
    #         for agent_q, agent_p, agent_id in zip(ag_q, ag_p, ag_id):
    #             residual_q = equilibrium_q - quantity_supplied
    #             if residual_q <= 0:
    #                 # agent is not cleared
    #                 self.traders[agent_id].da_position = 0
    #             else:
    #                 # clear agent
    #                 if agent_q <= residual_q:
    #                     # completely cleared
    #                     self.traders[agent_id].da_position = agent_q
    #                     quantity_supplied += agent_q
    #                 else:
    #                     # partially cleared
    #                     self.traders[agent_id].da_position = residual_q
    #                     quantity_supplied += residual_q

    def store_result(self, before_imbalance_flag):
        columns = ["Trader", "Position", "Revenues"]
        data = [(id, trader._position, trader._cash_flow) for id, trader in self.traders.items()]
        pd.DataFrame(data, columns=columns).to_csv("%s/%s" % (self.path,
                                                              "results_before_imbalance_settlement_%s.csv" % self.product_number if
                                                              before_imbalance_flag else
                                                              "results_after_imbalance_settlement_%s.csv" % self.product_number))

    def init_regulation_info(self):
        for id,trader in self.traders.items():
            trader.true_up_regulation = self.up_regulation_price
            trader.true_down_regulation = self.dn_regulation_price

class allmarket:
    """
    Initialize the the Market Operator with multiple products

    :param config_traders: dict with the market participants configuration params
    :param config_market_operator: dict with the market operator configuration params
    :param path: Path for saving results
    :param no_of_products: Number of products
    """

    def __init__(self, config_traders, config_market_operator, path, no_of_products):
        self.config_traders = config_traders
        self.config_market_operator = config_market_operator
        self.path = path

        self.no_of_products = no_of_products
        # self.start_trade = pd.to_datetime(start_trade)
        # self.end_trade = pd.to_datetime(end_trade)
        # self.time_step = pd.to_timedelta(time_step)
        # self.product_length = pd.to_timedelta(product_length)
        #        self.trading_horizon_all = pd.date_range(self.start_trade, self.end_trade + (self.no_of_products-1) * self.product_length,freq=self.time_step)
        #        self.all_positions = defaultdict(dict)
        #        self.all_forecasts = defaultdict(dict)

        self.products = dict()
        self.plotters = dict()
        self.build_products()
        self.trading_horizon_all = pd.date_range(self.products[0].start_trade,
                                                 self.products[self.no_of_products - 1].end_trade,
                                                 freq=self.products[0].time_step)
        self.all_positions = {
            time: {id: [None for p_no, product in self.products.items()] for id in self.products[0].traders} for time in
            self.trading_horizon_all}
        self.all_forecasts = {
            time: {id: [None for p_no, product in self.products.items()] for id in self.products[0].traders} for time in
            self.trading_horizon_all}

        self.runproduct()
        self.plots()
        print("breakpoint")

    @staticmethod
    def name():
        return "allmarket"

    def build_products(self):
        """
        Instantiates multiple products (multiple instances of MarketOperator)
        :return: N/A
        """

        if self.no_of_products>1:
            for p_no in range(self.no_of_products):
                config_mo_single = dict()
                # config_traders_single = defaultdict(dict)
                config_traders_single = defaultdict(lambda: defaultdict(dict))
                for value in self.config_market_operator:
                    config_mo_single[value] = self.config_market_operator[value][p_no]
                for type in self.config_traders.keys():
                    for id in self.config_traders[type].keys():
                        for value in self.config_traders[type][id].keys():
                            config_traders_single[type][id][value] = self.config_traders[type][id][value][p_no]
                self.products[p_no] = MarketOperator(config_traders=config_traders_single,
                                                     config_market_operator=config_mo_single,
                                                     path=self.path, product_number=p_no)
        else:
            self.products[0] = MarketOperator(config_traders=self.config_traders,
                                                 config_market_operator=self.config_market_operator,
                                                 path=self.path, product_number=0)


    def update_multiple_product_info(self, current_time):
        """
        Makes new private information of a trader available across all the products
        :param current_time: The time index at which the new information is added
        :return: N/A
        """
        all_product_stored = []
        all_product_eqlbm = []
        for p_no, product in self.products.items():
            for id, trader in product.traders.items():
                self.all_positions[current_time][id][p_no] = trader._position
                #               self.products[p_no].traders[id]._position
                if hasattr(trader, 'forecaster'):
                    self.all_forecasts[current_time][id][p_no] = trader.forecaster.curr
                if isinstance(trader, Storage):
                    all_product_eqlbm.append(trader.trader_behavior.eqlbm)
                    if current_time == self.trading_horizon_all[0]:
                        all_product_stored.append(trader.stored_energy)

        for p_no, product in self.products.items():
            for id, trader in product.traders.items():
                trader.product_positions = self.all_positions[current_time][id]
                if hasattr(trader, 'forecaster'):
                    trader.forecaster.product_forecasts = self.all_forecasts[current_time][id]
                if isinstance(trader, Storage):
                    trader.all_eqlbm = all_product_eqlbm
                    if current_time == self.trading_horizon_all[0]:
                        trader.all_stored = all_product_stored

    def runproduct(self):
        """
        Simulates the multiple products operation.
        :return: N/A
        """
        for current_time in self.trading_horizon_all:
            # print(current_time)
            self.update_multiple_product_info(current_time)
            logging.info("%s ############################################" % current_time)
            for p_no, product in self.products.items():
                if current_time in product.trading_horizon:
                    # if p_no == 0: print(current_time)
                    # print("    Product %s" %p_no)
                    product.runMarket(current_time, self.all_positions)
            self.store_imbalances()
        self.store_result_all(before_imbalance_flag=True)
        for p_no, product in self.products.items():
            product.imbalance_settlement()
            print(product.counter)
        self.store_result_all(before_imbalance_flag=False)

    def plots(self):
        """
        Plots the results of all products
        :return: N/A
        """
        for p_no, product in self.products.items():
            self.plotters[p_no] = Plotter(product, product.path, show_plots=False)
            self.plotters[p_no].generate_plots()

    def store_result_all(self, before_imbalance_flag):
        """
        Stores the result of position and revenue of all products in a csv file
        :return: N/A
        """
        columns = ["Product", "Trader", "Position", "Revenues"]

        data = [(p_no, id, trader._position, trader._cash_flow) for p_no, product in self.products.items() for
                id, trader in product.traders.items()]
        pd.DataFrame(data, columns=columns).to_csv("%s/%s" % (self.path,
                                                              "results_before_imbalance_settlement.csv" if
                                                              before_imbalance_flag else
                                                              "results_after_imbalance_settlement.csv"))
        stored_data = pd.DataFrame(data, columns=columns)

    def store_imbalances(self):
        """
        Debugging method which stores imbalances. It was used to understand the functioning of the simulator
        :return: N/A
        """
        for p_no, product in self.products.items():
            for id, trader in product.traders.items():
                trader.all_imbalances.append(trader.imbalance)
                trader.all_ramp_up_margin.append(trader.ramp_up_margin)
                trader.all_ramp_down_margin.append(trader.ramp_down_margin)

import logging
import random
from copy import deepcopy

import numpy as np
from mmabm.shared import Side, OType
from forecast.outage import Outage
from strategies import *

strategies = {"naive": Naive, "AA": Trader_AA}


class Agent:

    def __init__(self, ID, trading_horizon, da_position, capacity, range_imb, limit_price_sell, limit_price_buy,
                 w_update_limits,
                 eqbm_price, n_orders, strategy="naive", out_probability=0., out_percentage=0., out_time=0., aon_trader=False, switch_strategies = 0):

        """
        Class of generic trading agent. It inherits from an agent Market maker just to maintain a few nice properties
        but some of them are not used through
        :param ID: Name of the trader
        :param trading_horizon: timesteps of the trading session
        :param da_position: position from the day ahead market
        :param capacity: total available capacity of the assets
        :param limit_price_sell: minimum price willing to sell
        :param limit_price_buy: maximum price willing to buy
        :param eqbm_price: agent belief of the fair equilibrium price
        :param n_orders: number of orders that the agent is placing at each time step
        """
        self.trader_id = ID
        # Inintialize the adaptive aggressiveness strategy for each trader
        self.da_position = da_position
        self.trading_horizon = trading_horizon
        self.capacity = capacity
        self.imbalance = 0.
        self.pos_imb_price = None
        self.neg_imb_price = None
        self.range_imb = range_imb
        self.ramp_up_margin = None  # some real values need to be given?
        self.ramp_down_margin = None
        self.init_bid_price = 10
        self.init_ask_price = 60
        self.cancel_collector = []
        self._cum_position = []  # progressively put in the list the positions of each agent
        self.limit_price_buy = limit_price_buy
        self.limit_price_sell = limit_price_sell
        self.limit_price_buy_init = self.limit_price_buy
        self.limit_price_sell_init = self.limit_price_sell
        self.w_update_limits = w_update_limits
        self.trader_behavior = strategies[strategy](eqbm_price, limit_price_buy, limit_price_sell)
        self.outage = Outage(out_probability, out_percentage)
        window_outages = self.trading_horizon[int(out_time * len(self.trading_horizon)):]
        self.out_time_flag = lambda t: t in window_outages
        self.n_orders = n_orders
        self.limits_buy = []
        self.limits_sell = []
        self.aggressivenesses_buy = []
        self.aggressivenesses_sell = []
        self.eqlbm_prices = []
        self.targets_buy = []
        self.targets_sell = []
        self.submitted_orders = []
        self.quote_collector = []
        self._quote_sequence = 0
        self.local_book = {}
        self.cancel_collector = []
        self._position = 0
        self._cash_flow = 0
        self.cash_flow_collector = []
        self.position_initialize()

        self.product_positions = []
        self.own_bid_transactions = []
        self.own_ask_transactions = []
        self.all_imbalances = []
        self.all_ramp_up_margin = []
        self.all_ramp_down_margin = []
        self.aon_trader = aon_trader
        self.true_up_regulation = 0
        self.true_down_regulation = 0
        self.switch_strategies_time = self.trading_horizon[
                                    int(switch_strategies * len(self.trading_horizon))]

        if strategy == "AA":
            self.secondary_trader_behaviour = strategies["naive"](eqbm_price, limit_price_buy, limit_price_sell)
        else:
            self.secondary_trader_behaviour = strategies["AA"](eqbm_price, limit_price_buy, limit_price_sell)

    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2})'.format(class_name, self.trader_id, self.trader_behavior.name())

    def __str__(self):
        return str(tuple([self.trader_id, self.quantity]))

    @staticmethod
    def name():
        return "GenericAgent"

    @staticmethod
    def make_q():
        """
        Iniital function to define the traded quantity
        :return:  random volume
        """
        default_arr = np.array([1, 2, 3, 4, 5])
        return random.choice(default_arr)

    def update_limit(self):
        """
        Function that updates the limit proces
        :return: N/A
        """
        pass

    def process_signal(self, time, top_of_book):
        """
        Receive the market information and decide which orders to place in the market (orders are accumulated
        in the quote collector and then submitted to the market operator)
        :param time: timestep
        :param top_of_book: top of the order book
        :return:
        """
        self.quote_collector.clear()
        self._process_cancels(time)
        self._store_cum_position()

        if random.random() < 0.5:  # this can be based on Alex's thesis? three conditions for choosing the side
            side = Side.BID
        else:
            side = Side.ASK
        volume = self.make_q()
        self._create_orders(time, side, False, top_of_book, volume)
        self.switch_strategies(time, top_of_book)

    def analyse_state(self):
        pass

    def store_aggressiveness(self):
        """
        Record the aggressiveness params at each step
        :return: N/A
        """
        if self.trader_behavior.name() == "Trader_AA":
            self.aggressivenesses_buy.append(self.trader_behavior.aggressiveness_buy)
            self.aggressivenesses_sell.append(self.trader_behavior.aggressiveness_sell)

    def store_targets(self):
        """
        Record the target params at each step
        :return: N/A
        """
        if self.trader_behavior.name() == "Trader_AA":
            self.targets_buy.append(self.trader_behavior.target_buy)
            self.targets_sell.append(self.trader_behavior.target_sell)

    def store_eqlbm(self):
        """
        Record the computed equilibrium prices at each step
        :return: N/A
        """
        self.eqlbm_prices.append(self.trader_behavior.eqlbm)

    def _store_limits(self):
        """
        Record the limit rices at each step
        :return: N/A
        """
        self.limits_buy.append(self.limit_price_buy)
        self.limits_sell.append(self.limit_price_sell)

    def position_initialize(self):
        """
        Initialize the trading position to the day ahead position
        :return: N/A
        """
        self._position = self.da_position  # how is the position getting updated here?

    def _process_cancels(self, time):
        """
        Delete from the local order book the orders that were canceled from the market
        :param time: timestep
        :return: N/A
        """
        self.cancel_collector.clear()  # why do you clear cancel_collector? emptying list
        if self.local_book:  # local orders
            for q in self.local_book.values():
                if q["quantity"] > 0:
                    c = self._make_cancel_quote(q,
                                                time)  # why are you canceling order? canceling orders that are outstanding
                    self.cancel_collector.append(deepcopy(c))
        for c in self.cancel_collector:
            del self.local_book[c['order_id']]  # ?

    def _make_cancel_quote(self, q, time):
        """
        Define the cancel order
        :param q: volume
        :param time: timestep
        :return: dict of cancelling order
        """
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': self.trader_id,
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}

    def store_cum_position(self):
        """
        Record the cumulative position of the trader
        :return: N/A
        """
        self._cum_position.append(self._position)

    def _make_add_quote(self, time, side, price, quantity):
        """
        Define a new order based on the params
        :param time: time
        :param side: side (buy or sell)
        :param price: price
        :param quantity: volume
        :return: dict with the added order
        """

        self._quote_sequence += 1
        return {'order_id': self._quote_sequence, 'trader_id': self.trader_id, 'timestamp': time,
                'type': OType.ADD, 'quantity': quantity, 'side': side, 'price': price}

    def _create_orders(self, time, side, init_step_flag, top_of_book, volume):
        """
        Generate the order to be submitted to the market operator
        :param time: timestep
        :param side: side (buy or sell)
        :param init_step_flag: bool whether we are at the inital time step
        :param top_of_book: top of the order book
        :param volume: volume
        :return: N/A
        """

        volumes, prices = self.trader_behavior.getorder(self.limit_price_sell, self.limit_price_buy, side,
                                                        init_step_flag, top_of_book, volume,
                                                        n_orders=self.n_orders)
        for v, p in zip(volumes, prices):
            q = self._make_add_quote(time, side, p, v)
            self.local_book[q['order_id']] = deepcopy(q)  # add in local book
            self.quote_collector.append(deepcopy(q))  # add in quote collector
            self.submitted_orders.append(deepcopy(q))


    def confirm_trade_local(self, confirmed_trade):
        '''Modify _cash_flow and _position; update the local_book'''

        if confirmed_trade['side'] == Side.BID:
            revenue = -confirmed_trade['price'] * confirmed_trade['quantity']
            self._cash_flow += revenue
            self._position -= confirmed_trade['quantity']
        else:
            revenue = confirmed_trade['price'] * confirmed_trade['quantity']
            self._cash_flow += revenue
            self._position += confirmed_trade['quantity']
        if confirmed_trade['order_id'] not in self.local_book.keys():
            logging.info(confirmed_trade)
        to_modify = self.local_book[confirmed_trade['order_id']]
        if confirmed_trade['quantity'] == to_modify['quantity']:
            del self.local_book[to_modify['order_id']]
        else:
            self.local_book[confirmed_trade['order_id']]['quantity'] -= confirmed_trade['quantity']
        self._cumulate_cashflow(confirmed_trade['timestamp'])

        return revenue

    def _cumulate_cashflow(self, timestamp):
        self.cash_flow_collector.append({'mmid': self.trader_id, 'timestamp': timestamp, 'cash_flow': self._cash_flow,
                                         'position': self._position})

    def update_capacity(self, time):
        if self.out_time_flag(time):
            self.capacity = self.outage.update_capacity(self.capacity)

    def estimate_imbalance_price(self):

        self.pos_imb_price = self.range_imb * np.random.random_sample() + (2 * self.true_down_regulation - self.range_imb) / 2
        self.neg_imb_price = self.range_imb * np.random.random_sample() + (2 * self.true_up_regulation - self.range_imb) / 2

    def switch_strategies(self,time, top_of_book):
        if time == self.switch_strategies_time and not self.switch_strategies_time == self.trading_horizon[0]:
            temp = self.trader_behavior
            self.trader_behavior = self.secondary_trader_behaviour
            self.secondary_trader_behaviour = temp
            self.trader_behavior.initialize(top_of_book)
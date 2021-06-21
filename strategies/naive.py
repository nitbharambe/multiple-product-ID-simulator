import numpy as np

from mmabm.shared import Side
from strategies.generic import Strategy


class Naive(Strategy):

    def __init__(self, eqlbm, limit_buy, limit_sell, quote_range=5.):
        super().__init__(eqlbm, limit_buy, limit_sell)
        self._quote_range = quote_range


    def getorder(self, limit_price_sell, limit_price_buy, side, init_step_flag, tob, volume, n_orders):
        self.limit_sell = limit_price_sell
        self.limit_buy = limit_price_buy
        self.job = side
        volume_share = volume / n_orders
        if self.job == Side.BID:
            # currently a buyer (working a bid order)
            prices = np.random.choice(
                np.arange(min(tob["best_bid"] - self._quote_range, self.limit_buy - self._quote_range),
                          min(tob["best_ask"] + self._quote_range, self.limit_buy), 0.5),
                size=n_orders)
        else:
            # currently a seller (working a sell order)
            prices = np.random.choice(
                np.arange(max(tob["best_bid"], self.limit_sell),
                          max(tob["best_ask"] + self._quote_range, self.limit_sell + self._quote_range), 0.5),
                size=n_orders)
        return [volume_share] * n_orders, sorted(prices)

    def respond(self, time, tob, transactions, verbose=True):
        self.prev_best_bid_p = tob["best_bid"]
        self.prev_best_bid_q = tob["bid_size"]
        self.prev_best_ask_p = tob["best_ask"]
        self.prev_best_ask_q = tob["ask_size"]

    def initialize(self, tob):

        self.prev_best_bid_p = tob["best_bid"]
        self.prev_best_bid_q = tob["bid_size"]
        self.prev_best_ask_p = tob["best_ask"]
        self.prev_best_ask_q = tob["ask_size"]

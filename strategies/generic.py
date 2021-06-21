class Strategy:
    def __init__(self, eqlbm, limit_buy, limit_sell):
        self.eqlbm = eqlbm
        self.limit_buy = limit_buy
        self.limit_sell = limit_sell
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

    def name(self):
        return type(self).__name__

    def getorder(self, limit_price_sell, limit_price_buy, side, init_step_flag, tob, volume, n_orders):
        raise NotImplementedError

    def respond(self, time, tob, transactions, verbose=True):
        raise NotImplementedError

    def initialize(self, tob):
        raise NotImplementedError

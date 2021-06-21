import logging
from copy import deepcopy
from agents.generic_agent import Agent
from mmabm.shared import Side
import statistics


class Conventional(Agent):

    def __init__(self,  ID, trading_horizon, da_position, capacity, range_imb, limit_price_sell, limit_price_buy,
                 w_update_limits,  eqbm_price,min_stable_load=50, strategy="naive", n_orders=10, out_probability=0.,
                 out_percentage=0., out_time=0.,
                 product_number=0, ramp_up=4000, ramp_down=4000, ramp_active_time=1, aon_trader=False, switch_strategies=0):

        """
        Object for representing the conventional units
        :param ID: name of the trader
        :param trading_horizon: timesteps of the trading session
        :param da_position: position from the day ahead market
        :param capacity: total available capacity of the assets
        :param limit_price_sell: minimum price willing to sell
        :param limit_price_buy: maximum price willing to buy
        :param eqbm_price: agent belief of the fair equilibrium price
        :param n_orders: number of orders that the agent is placing at each time step
        """
        super().__init__(ID=ID, trading_horizon=trading_horizon, da_position=da_position, capacity=capacity,
                         range_imb=range_imb,
                         limit_price_sell=limit_price_sell, limit_price_buy=limit_price_buy,
                         w_update_limits=w_update_limits, eqbm_price=eqbm_price,
                         strategy=strategy, n_orders=n_orders, out_probability=out_probability,
                         out_percentage=out_percentage, out_time=out_time, aon_trader=aon_trader, switch_strategies=switch_strategies)
        self.final_imbalance = 0.
        self.min_stable_load = min_stable_load

        self.product_number = product_number
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.ramp_active_timeline = self.trading_horizon[
                                    int(ramp_active_time * len(self.trading_horizon)): len(self.trading_horizon)]
        self.update_limit()
        self.imblresult =0

    @staticmethod
    def name():
        return "Conventional"

    def compute_margins(self):
        """
        Function that computes the available upwards or downwards ramping capacity of the unit.
        The upwards(donwards) capacity is the max volume that can be sold(bought) in the market.
        :return: N/A
        """
        #
        if self.imbalance < 0.:
            self.ramp_up_margin = 0.
            self.ramp_down_margin = self._position - self.capacity #self.ramp_down

        else:
            self.ramp_up_margin = self.capacity - self._position
            self.ramp_down_margin = max(self._position - min(self.min_stable_load, self.capacity), 0)

    def process_signal(self, time, top_of_book):
        """
        Process the state of the order book, prepare the previously existing orders for canceling and prepare the new
        orders that will be submitted
        :param time: timestep
        :param top_of_book: top of the order book
        :return: N/A
        """

        self.update_capacity(time)
        self.compute_imbalance_new(time)
        #self.compute_imbalance()
        self.estimate_imbalance_price()
        self.update_limit_new()
        #self.update_limit()
        self.quote_collector.clear()
        init_step_flag = False  # time == self.trading_horizon[0]
        self._process_cancels(time)
        logging.info("Agent: %s" % self.trader_id)
        logging.info("Limit price buy: %s" % self.limit_price_buy)
        logging.info("Limit price sell: %s" % self.limit_price_sell)
        self.compute_margins_new(time)
        #self.compute_margins()

        if self.ramp_down_margin > 0:  # If there is availbable downwards capacity produce buy orders
            side = Side.BID
            self._create_orders(time, side, init_step_flag, top_of_book, self.ramp_down_margin)

        if self.ramp_up_margin > 0:  # If there is available upwards capacity produce sell orders
            side = Side.ASK
            self._create_orders(time, side, init_step_flag, top_of_book, self.ramp_up_margin)

        self._store_limits()
        self.switch_strategies(time, top_of_book)

    def compute_final_imbalance(self):
        """
        Compute the discrepancy between the final position and the energy that the unit can deliver.
        :return: N/A
        """
        if self._position > self.capacity:
            self.final_imbalance = self.capacity - self._position
        elif self._position < 0:
            self.final_imbalance = self._position
        else:
            self.final_imbalance = self.imbalance
            #pass

    def da_trade(self):
        return "SUPPLY", self.capacity, self.limit_price_sell

    def compute_imbalance(self):
        """
        Compute the difference between current position and recent forecast
        :return: N/A
        """
        self.imbalance = min(self.capacity - self._position, 0)

    def update_limit(self):
        """
        Function intended to change the limits to buy and sell based on the market conditions. For now it does not change anything
        :return: N/A
        """
        if self.imbalance < 0.:
            self.limit_price_buy = (1 - self.w_update_limits) * self.limit_price_buy + self.w_update_limits * max(
                self.neg_imb_price, self.limit_price_buy_init)
        else:
            self.limit_price_buy = self.limit_price_buy_init

    def compute_imbalance_was(self, time):
        """
        Compute the difference between current position and recent forecast
        :return: N/A
        """
        if time in self.ramp_active_timeline:
            try:
                cond1 = (self._position-self.product_positions[self.product_number + 1] > self.ramp_down)
            except:
                cond1 = False
            try:
                cond2 = (self.product_positions[self.product_number - 1] - self._position > self.ramp_down)
            except:
                cond2 = False
            if cond1 or cond2:
                min_t = -(self._position - min(self.product_positions[self.product_number + 1],
                                                        self.product_positions[self.product_number - 1]) - self.ramp_down)
                self.imbalance = min(min_t, self.capacity - self._position, 0)
#            if cond1:
#                self.imbalance = -(self._position - self.product_positions[self.product_number + 1] - self.ramp_down)
#            elif cond2:
#                self.imbalance = -(self._position - self.product_positions[self.product_number - 1] - self.ramp_down)
            else:
                self.imbalance = min(self.capacity - self._position, 0)
        else:
            self.imbalance = min(self.capacity - self._position, 0)

    def compute_imbalance_new(self, time):
        """
        Compute the difference between current position and recent forecast
        :return: N/A
        """
        if time in self.ramp_active_timeline:
            m_products = statistics.mean(self.product_positions)
            imbl = 0
            if self.product_number < len(self.product_positions) - 1:
                prod_c = self.product_positions[self.product_number + 1]
                if abs(self._position - m_products) > abs(prod_c - m_products):
                    if ( prod_c > self._position + self.ramp_up):
                        imbl = (prod_c - self._position - self.ramp_up)
                    if ( prod_c < self._position - self.ramp_down):
                        imbl = (prod_c - self._position + self.ramp_down)
            if self.product_number > 0:
                prod_c = self.product_positions[self.product_number - 1]
                if abs(self._position - m_products) > abs(prod_c - m_products):
                    if ( prod_c > self._position + self.ramp_down):
                        if imbl > 0:
                            imbl = max(imbl, (prod_c - self._position - self.ramp_down))
                        else:
                            imbl = (prod_c - self._position - self.ramp_down)
                    if ( prod_c < self._position - self.ramp_up):
                        if imbl < 0:
                            imbl = min(imbl, (prod_c - self._position + self.ramp_up))
                        else:
                            imbl = (prod_c - self._position + self.ramp_up)
            if imbl == 0:
                self.imbalance = 0
            else:
                self.imbalance = imbl
        else:
            self.imbalance = min(self.capacity - self._position, 0)

    def compute_margins_was(self,time):
        """
        Function that computes the available upwards or downwards ramping capacity of the unit.
        The upwards(donwards) capacity is the max volume that can be sold(bought) in the market.
        :return: N/A
        """
        #
        if self.imbalance < 0.:
            if time in self.ramp_active_timeline:
                self.ramp_down_margin = -self.imbalance
            else:
                self.ramp_down_margin = self._position - self.capacity
            self.ramp_up_margin = 0.

        else:
            self.ramp_up_margin = self.capacity - self._position
            self.ramp_down_margin = max(self._position - min(self.min_stable_load, self.capacity), 0)

    def compute_margins_new(self,time):
        """
        Function that computes the available upwards or downwards ramping capacity of the unit.
        The upwards(donwards) capacity is the max volume that can be sold(bought) in the market.
        :return: N/A
        """
        #
        if time in self.ramp_active_timeline:
            if self.imbalance < 0.:
                self.ramp_down_margin = max(self.ramp_down,abs(self.imbalance))
                self.ramp_up_margin = 0
            elif self.imbalance > 0:
                self.ramp_down_margin = 0.
                self.ramp_up_margin = max(self.ramp_up,abs(self.imbalance))
            else:
                self.ramp_down_margin = 0
                self.ramp_up_margin = 0
                self.ramp_up_margin = min(self.capacity - self._position, 0.5 * self.ramp_up)
                self.ramp_down_margin = min(max(self._position - min(self.min_stable_load, self.capacity), 0),
                                            0.5 * self.ramp_down)

        else:
#            self.ramp_up_margin = 2*self.ramp_up
#            self.ramp_down_margin = 2*min(self._position - min(self.min_stable_load, self.capacity), self.ramp_down)

            self.ramp_up_margin = min(self.capacity - self._position, 1.5*self.ramp_up)
            self.ramp_down_margin = min(max(self._position - min(self.min_stable_load, self.capacity), 0), 1.5*self.ramp_down)
#            self.ramp_up_margin = self.capacity - self._position
#            self.ramp_down_margin = max(self._position - min(self.min_stable_load, self.capacity), 0)

    def update_limit_new(self):
        """
        Function intended to change the limits to buy and sell based on the market conditions. For now it does not change anything
        :return: N/A
        """

        if self.imbalance < 0.:
            self.limit_price_buy = min((1 - self.w_update_limits) * self.limit_price_buy + self.w_update_limits * max(
                self.neg_imb_price,
                self.limit_price_buy_init), self.limit_price_sell)
        elif self.imbalance > 0.:
            self.limit_price_sell = max ((1 - self.w_update_limits) * self.limit_price_sell + self.w_update_limits * min(
                self.pos_imb_price,
                self.limit_price_sell_init), self.limit_price_buy)
        else:
            self.limit_price_buy = self.limit_price_buy_init
            self.limit_price_sell = self.limit_price_sell_init
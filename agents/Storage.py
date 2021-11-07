import logging
from agents.generic_agent import Agent
from mmabm.shared import Side
import statistics

class Storage(Agent):


    def __init__(self,  ID, trading_horizon, da_position, capacity, range_imb, limit_price_sell, limit_price_buy,
                 w_update_limits,  eqbm_price,
                 stored_energy, min_stable_load,
                 strategy="naive", n_orders=10, out_probability=0.,
                 out_percentage=0., out_time=0.,product_number=0, ramp_up=4000, ramp_down=4000, ramp_active_time=1, aon_trader = False, switch_strategies = 0):

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
                         out_percentage=out_percentage, out_time=out_time, aon_trader=aon_trader, switch_strategies = switch_strategies)
        self.final_imbalance = 0.
        self.stored_energy = stored_energy
        self.all_stored = []
        self.all_eqlbm = []
        self.min_stable_load = min_stable_load
        self.up_arbitrage_flag = False
        self.down_arbitrage_flag = False

        self.product_number = product_number
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.ramp_active_timeline = self.trading_horizon[
                                    int(ramp_active_time * len(self.trading_horizon)): len(self.trading_horizon)]
        self.estimate_imbalance_price()
        self.update_limit()
        self.count = 0

    @staticmethod
    def name():
        return "Storage"

    def da_trade(self):
        # TODO
        return None

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
        # self.compute_imbalance()
        self.compute_imbalance_new(time)
        self.estimate_imbalance_price()
        self.update_limit()
        self.update_limit_new()
        self.quote_collector.clear()
        init_step_flag = False  # time == self.trading_horizon[0]
        self._process_cancels(time)
        logging.info("Agent: %s" % self.trader_id)
        logging.info("Limit price buy: %s" % self.limit_price_buy)
        logging.info("Limit price sell: %s" % self.limit_price_sell)
        #self.compute_margins()
        self.compute_margins_new(time)

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

    def da_trade(self):
        return "SUPPLY", self.capacity, self.limit_price_sell

    def compute_imbalance_old(self):
        """
        Compute the difference between current position and recent forecast
        :return: N/A
        """
        self.imbalance = self.  stored_energy - sum(self.product_positions)
        #self.imbalance = min(self.capacity - self._position, 0)

    # def update_limit(self):
    #     """
    #     Function intended to change the limits to buy and sell based on the market conditions. For now it does not change anything
    #     :return: N/A
    #     """
    #     if self.imbalance < 0.:
    #         self.limit_price_buy = (1 - self.w_update_limits) * self.limit_price_buy + self.w_update_limits * max(
    #             self.neg_imb_price, self.limit_price_buy_init)
    #         if self.own_bid_transactions:
    #             self.limit_price_buy = min(self.limit_price_buy, self.own_bid_transactions[-1])
    #     else:
    #         self.limit_price_sell = (1 - self.w_update_limits) * self.limit_price_sell + self.w_update_limits * min(
    #             self.pos_imb_price,
    #             self.limit_price_sell_init)
    #         if self.own_ask_transactions:
    #             self.limit_price_sell = max(self.limit_price_sell, self.own_ask_transactions[-1])

    # def compute_imbalance_old(self, time):
    #     """
    #     Compute the difference between current position and recent forecast. Added physical ramp up/down constraints
    #     First attempt at storage imbalances
    #     :return: N/A
    #     """
    #     if time in self.ramp_active_timeline:
    #         try:
    #             cond1 = (self._position - self.product_positions[self.product_number + 1] > self.ramp_down)
    #         except:
    #             cond1 = False
    #         try:
    #             cond2 = (self._position - self.product_positions[self.product_number - 1] > self.ramp_down)
    #         except:
    #             cond2 = False
    #         if cond1 or cond2:
    #             self.imbalance = -(self._position - min(self.product_positions[self.product_number + 1],
    #                                                     self.product_positions[self.product_number - 1]) - self.ramp_down)
    #         #elif cond1:
    #         #    self.imbalance = -(self._position - self.product_positions[self.product_number + 1] - self.ramp_down)
    #         #elif cond2:
    #         #    self.imbalance = -(self._position - self.product_positions[self.product_number - 1] - self.ramp_down)
    #         else:
    #             self.imbalance = min(self.ramp_up, self.stored_energy - sum(self.product_positions))
    #     else:
    #         self.imbalance = self.stored_energy - sum(self.product_positions)

    def compute_imbalance_new(self, time):
        """
        Compute the difference between current position and recent forecast
        :return: N/A
        """
        imbl = 0
        imbl_e = 0
        if self.product_number < len(self.product_positions)-1:
            vwp_ahead = sum(self.product_positions[i] * self.all_eqlbm[i] for i in
                            range(self.product_number + 1, len(self.all_eqlbm))) / sum(
                self.product_positions[self.product_number + 1:])
            if self.all_eqlbm[self.product_number] < vwp_ahead:
                add_energy = sum(self.all_stored[self.product_number+1:]) - sum(
                    self.product_positions[self.product_number+1:])
                if add_energy > self.capacity:
                    imbl_e = - (add_energy - self.capacity)

                elif add_energy < self.min_stable_load:
                    imbl_e = - (self.min_stable_load - add_energy)
                else:
                    imbl_e = 0
        ''''
        #First attempt at improved storage algorithm
        
        if self.product_number > 0:
            vwp_behind = sum(self.product_positions[i] * self.all_eqlbm[i] for i in
                             range(self.product_number)) / sum(self.product_positions[:self.product_number])
            if self.all_eqlbm[self.product_number] < vwp_behind:
                add_energy2 = self._position - self.stored_energy
                prev_eqlbm_greater = self.all_eqlbm[self.product_number] < self.all_eqlbm[:self.product_number]
                #if prev_eqlbm_greater and add_energy<self.min_stable_load and prev_eqlbm_greater:
                #    imbl_e = add_energy2
                add_energy = sum(self.all_stored[:self.product_number]) - sum(self.product_positions[:self.product_number])
                if add_energy > self.capacity:
                    imbl_e = - (add_energy - self.capacity)
                elif add_energy < self.min_stable_load:
                    imbl_e =  min (- (self.min_stable_load - add_energy) , add_energy2)
                    if imbl_e < (self._position - self.min_stable_load) and (self._position - self.min_stable_load) <0:
                        imbl_e = max(imbl_e,self._position - self.min_stable_load)
                else:
                    imbl_e = 0
                    '''
        m_products = statistics.mean(self.product_positions)
        if self.product_number < len(self.product_positions) - 1:
            prod_c = self.product_positions[self.product_number + 1]
            if abs(self._position - m_products) > abs(prod_c - m_products):
                if ( prod_c > self._position + self.ramp_up):
                    imbl = (prod_c - self._position - self.ramp_up)
                if ( prod_c < self._position - self.ramp_down):
                    imbl = (prod_c - self._position + self.ramp_down)
            self.up_arbitrage_flag = abs(imbl) > min(self.ramp_up, self.ramp_down) and imbl > 0
            self.down_arbitrage_flag = abs(imbl) > min(self.ramp_up, self.ramp_down) and imbl < 0

        if self.product_number > 0:
            prod_c = self.product_positions[self.product_number - 1]
            if abs(self._position - m_products) > abs(prod_c - m_products):
                if ( prod_c > self._position + self.ramp_down):
                    #imbl = (prod_c - self._position - self.ramp_down)
                    if 1:
                        if imbl > 0:
                            imbl = max(imbl, (prod_c - self._position - self.ramp_down))
                        else:
                            imbl = (prod_c - self._position - self.ramp_down)
                if ( prod_c < self._position - self.ramp_up):
                    #imbl = (prod_c - self._position + self.ramp_up)
                    if 1:
                        if imbl < 0:
                            imbl = min(imbl, (prod_c - self._position + self.ramp_up))
                        else:
                            imbl = (prod_c - self._position + self.ramp_up)
            self.up_arbitrage_flag = abs(imbl) > min(self.ramp_up, self.ramp_down) and imbl > 0
            self.down_arbitrage_flag = abs(imbl) > min(self.ramp_up, self.ramp_down) and imbl < 0

        if time in self.ramp_active_timeline:
            if self.ramp_active_timeline.indexer_at_time(time) > 0.5 * len(self.ramp_active_timeline):
                self.imbalance = imbl
            elif imbl_e:
                self.imbalance = imbl_e
            else:
                self.imbalance = 0
        else:
            self.imbalance = min(self.capacity - self._position, 0)

    def compute_margins_new(self,time):
        """
        Function that computes the available upwards or downwards ramping capacity of the unit.
        The upwards(downwards) capacity is the max volume that can be sold(bought) in the market.
        :return: N/A
        """
        #
        if self.ramp_active_timeline.indexer_at_time(time) > 0.5 * len(self.ramp_active_timeline):
            if self.imbalance < 0.:
                self.ramp_down_margin = max(self.ramp_down,abs(self.imbalance))
                self.ramp_up_margin = 0
            elif self.imbalance > 0:
                self.ramp_down_margin = 0.
                self.ramp_up_margin = max(self.ramp_up,abs(self.imbalance))
            else:
                self.ramp_up_margin = min(self.capacity - self._position, 0.1 * self.ramp_up)
                self.ramp_down_margin = min(max(self._position - min(self.min_stable_load, self.capacity), 0),
                                            0.1 * self.ramp_down)
        else:
            if self.up_arbitrage_flag:
                self.ramp_up_margin = 0
            else:
                self.ramp_up_margin = min(self.capacity - self._position, 0.5 * self.ramp_up)
            if self.down_arbitrage_flag:
                self.ramp_down_margin = 0
            else:
                self.ramp_down_margin = min(max(self._position - min(self.min_stable_load, self.capacity), 0),
                                        0.5 * self.ramp_down)

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
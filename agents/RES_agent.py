import logging

from agents.generic_agent import Agent
from forecast import *
from forecast.GMPForecast import GMPForecast
from mmabm.shared import Side
import numpy as np

forecasters = {"Constant": ConstantOffsetForecast, "Cos": CosForecast, "Sin": SinForecast, "GMP": GMPForecast}


class RES(Agent):
    def __init__(self, ID, trading_horizon, da_position, capacity, init_forecast, realization, range_imb,
                 limit_price_sell,
                 limit_price_buy, w_update_limits, eqbm_price, forecast_type="Constant", update_forecast_every="15min",
                 strategy="naive",
                 n_orders=10, error_constant=None, out_probability=0., out_percentage=0., out_time=0., product_number=0, aon_trader=False, switch_strategies=0):
        """
        Object for representing the renewable energy units
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

        self.forecaster = forecasters[forecast_type](trading_horizon, capacity, init_forecast, realization,
                                                     update_forecast_every,
                                                     error_constant)
        self.recent_forecast = self.forecaster.reset()

        self.prev_diff_sell = 0.
        self.prev_diff_buy = 0.

        self.estimate_imbalance_price()
        self.update_limit()

        self.realization = None
        self.final_imbalance = None



    @staticmethod
    def name():
        return "RES"

    def process_signal(self, time, top_of_book):
        """
        Process the state of the order book, prepare the previously existing orders for canceling and prepare the new
        orders that will be submitted
        :param time: time
        :param top_of_book: top of the book
        :return:
        """
        self.quote_collector.clear()
        init_step_flag = False  # time == self.trading_horizon[0]

        self.update_capacity(time)
        self.get_updated_forecast(time, init_step_flag)  # Get the most fresh forecast
        self.compute_imbalance()  # Compute the difference between current position and recent forecast
        self.estimate_imbalance_price()
        self.update_limit()  # Intended to change the limits (for now it does nothing)

        self._process_cancels(time)  # cancel the outstanding orders
        self._store_limits()

        logging.info("Agent: %s"%self.trader_id)
        logging.info("Limit price buy: %s"%self.limit_price_buy)
        logging.info("Limit price sell: %s"%self.limit_price_sell)
        logging.info("Eqlbm: %s"%self.trader_behavior.eqlbm)

        logging.info("Position: %s"%self._position)
        logging.info("Recent forecast: %s"%self.recent_forecast)
        logging.info("Imbalance: %s"%self.imbalance)

        if self.imbalance < 0:  # if the imbalance is negative buy
            side = Side.BID
            self.ramp_down_margin = abs(self.imbalance)
            self.ramp_up_margin = 0.
        else:  # else sell
            side = Side.ASK
            self.ramp_up_margin = self.imbalance
            self.ramp_down_margin = 0.

        if abs(self.imbalance) > 0:  # Chech before placing orders with zero volume and then create the new orders
            self._create_orders(time, side, init_step_flag, top_of_book, abs(self.imbalance))
        self.switch_strategies(time, top_of_book)

    def get_updated_forecast(self, time, init_step_flag=False):
        """
        Get an update of the forecast if time is in the froecast update timeline
        :param time: time
        :param init_step_flag: book initial time step
        :return: N/A
        """

        recent_forecast = self.forecaster.forecast(time, init_step_flag)

        self.recent_forecast = self._verify_capacity_updates(recent_forecast)

    def compute_imbalance(self):
        """
        Compute the difference between current position and recent forecast
        :return: N/A
        """
        self.imbalance = self.recent_forecast - self._position

    @property
    def forecasts(self):
        return self.forecaster.history()

    def compute_final_imbalance(self):
        """
        Computs the final imbalance after the gate closure
        :return: N/A
        """
        realization = self.forecaster.realization()
        self.realization = self._verify_capacity_updates(realization)
        self.final_imbalance = self.realization - self._position

    def update_limit(self):
        """
        Function intended to change the limits to buy and sell based on the market conditions. For now it does not change anything
        :return: N/A
        """

        if self.imbalance < 0.:
            self.limit_price_buy = (1 - self.w_update_limits) * self.limit_price_buy + self.w_update_limits * max(
                self.neg_imb_price,
                self.limit_price_buy_init)

        else:
            self.limit_price_sell = (1 - self.w_update_limits) * self.limit_price_sell + self.w_update_limits * min(
                self.pos_imb_price,
                self.limit_price_sell_init)

    def da_trade(self):
        return "SUPPLY", self.recent_forecast, self.limit_price_sell

    def _verify_capacity_updates(self, value):
        return min(self.capacity, value) if value >= 0. else max(-self.capacity, value)

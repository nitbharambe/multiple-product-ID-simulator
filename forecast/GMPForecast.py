from copy import deepcopy
from itertools import cycle

import numpy as np
import pandas as pd

from forecast.generic_forecast import Forecast
from stocha_processes.gaussian_markov_process import create_process


class GMPForecast(Forecast):
    def __init__(self, trading_horizon, product_delivery_time, forecast_sign=1):
        """
        GMP auto-regressive based forecaster
        trading_horizon : array of timestamps during which we can refine our position
        product_delivery_time : timestamp of the delivery of the product
        """
        super().__init__()

        # steps during which we can refine our position
        self.trading_horizon = trading_horizon

        # time at which the delivery starts
        self.product_delivery_time = product_delivery_time

        # sign of the forecast
        self.forecast_sign = forecast_sign

        # info products
        self.products_delivery = None

        # Stochastic process variables
        self.mu_0 = None
        self.sigma_0 = None
        self.mu_q = None
        self.sigma_q = None
        self.a_q = None
        self.b_q = None

        # realisation of the process
        self.trajectory = None

    def fit(self, df, imbalance_col, steps_col, products_delivery):
        """
        fit an auto-regressive GMP
        df : data frame
        imbalance_col : str name of the column in df where the output of the process is stored
        steps_col : str name of the column in df providing the product id of the data
        products_delivery : array of timestamps, mapping from the id tot he delivery time of the product (hh:mm:ss)
        """
        if len(products_delivery) != len(pd.unique(df[steps_col])):
            raise Exception('The number of deliveries and the number steps in data frame shall be equal')

        # parameters of the process (by index)
        self.mu_0, self.sigma_0, self.mu_q, self.sigma_q, self.a_q, self.b_q = create_process(df,
                                                                                              imbalance_col,
                                                                                              steps_col)

        # mapping between index and delivery
        self.products_delivery = {time - time.normalize(): step_id for step_id, time in enumerate(products_delivery)}

        return self

    def reset(self, delivery_time=None):
        """
        generate a new realization and reset the forecasts
        delivery_time : timestamp in the delivery process for which we want the realisation to be returned
                        (default delivery time of the product we trade)
        """
        # remove previous forecasts
        self.history_forecasts = []

        # we start to generate the trajectory at most one day in advance
        trading_day = self.trading_horizon[0].normalize() - pd.Timedelta(days=1)

        # initialize variables
        self.trajectory = dict()
        previous_sa = None
        while True:
            for time, step_id in self.products_delivery.items():
                delivery = trading_day + time

                # find the latest delivery before the first trading time
                if delivery < self.trading_horizon[0]:
                    continue

                # we are in a quarter past the first trading time
                if previous_sa is None:
                    previous_sa = np.random.normal(self.mu_q[step_id], self.sigma_q[step_id])
                else:
                    previous_sa = np.random.normal(self.a_q[step_id] * previous_sa + self.b_q[step_id],
                                                   self.sigma_q[step_id])

                self.trajectory[delivery] = previous_sa

                if delivery >= self.product_delivery_time:
                    # when we reach the product we stop the trajectory
                    return self.realization(delivery_time)

            # next day
            trading_day += pd.Timedelta(days=1)

    def realization(self, delivery_time=None):
        """
        realization of a product
        delivery_time : timestamp in the delivery process for which we want the realisation to be returned
                        (default delivery time of the product we trade)
        """
        # default delivery
        if delivery_time is None:
            delivery_time = self.product_delivery_time

        return self.forecast_sign * self.trajectory[delivery_time]

    def forecast(self, time, init_step_flag, delivery_time=None):
        """
        forecast for the quantity delivered at delivery_time
        delivery_time : timestamp in the delivery process for which we want the forecast to be returned
                        (default delivery time of the product we trade)
        """
        # default delivery
        if delivery_time is None:
            delivery_time = self.product_delivery_time

        realization_passed = False
        previous_sa = None
        for real_time, real in self.trajectory.items():
            if not realization_passed and real_time <= time:
                # previous realization
                previous_sa = real
                continue
            elif not realization_passed:
                # the realization at time real_time is unobserved
                realization_passed = True

            # we propagate the observation through time in the process
            norm_time = real_time - real_time.normalize()
            step_id = self.products_delivery[norm_time]
            if previous_sa is None:
                previous_sa = np.random.normal(self.mu_q[step_id], self.sigma_q[step_id])
            else:
                previous_sa = np.random.normal(self.a_q[step_id] * previous_sa + self.b_q[step_id],
                                               self.sigma_q[step_id])

        self.history_forecasts.append(previous_sa)

        return self.forecast_sign * previous_sa

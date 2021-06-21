from forecast.generic_forecast import Forecast
import numpy as np
import pandas as pd
import statistics

class AdditiveErrorForecast(Forecast):

    def __init__(self, trading_horizon, init_forecast, real, update_forecast_every="15min"):
        super().__init__()

        self.trading_horizon = trading_horizon
        self.init_forecast = init_forecast
        self.real = real
        self.update_forecast_every = update_forecast_every

        # current forecast
        self.curr = init_forecast

        # list of the timestamp at which the forecast is updated
        self.forecast_update_timeline = pd.date_range(self.trading_horizon[0], self.trading_horizon[-1],
                                                      freq=self.update_forecast_every)
        #self.product_number = product_number
        self.product_forecasts = []

    def realization(self, delivery_time=None):
        return self.real

    def forecast(self, time, init_step_flag, delivery_time=None):
        if init_step_flag:
            self.curr = self.init_forecast
        else:
            if time in self.forecast_update_timeline:
                self.curr = self.realization() + self.error_range(time)
            else:
                return self.curr
        self.history_forecasts.append(self.curr)
        return self.curr

    def reset(self, delivery_time=None):
        self.history_forecasts = []
        return self.init_forecast

    def error_range(self, t):
        # variation from the 'real' value depending on the time
        raise Exception('Unimplemented method')

    def forecast_new(self, time, init_step_flag, delivery_time=None):
        if init_step_flag:
            self.curr = self.init_forecast
        else:
            if time in self.forecast_update_timeline:
                new_forecast = self.realization() + self.error_range(time)
                if len(self.product_forecasts) > 1:
                    mean_forecast = statistics.mean(self.product_forecasts)
                    sd_forecast= statistics.stdev(self.product_forecasts)
                    if new_forecast > mean_forecast+2*sd_forecast:
                        new_forecast = mean_forecast+2*sd_forecast
                    elif new_forecast < mean_forecast-2*sd_forecast:
                        new_forecast = mean_forecast-2*sd_forecast
                self.curr = new_forecast
            else:
                return self.curr
        self.history_forecasts.append(self.curr)
        return self.curr

class CosForecast(AdditiveErrorForecast):

    def __init__(self, trading_horizon, capacity, init_forecast, real, update_forecast_every="15min",
                 error_constant=None, capacity_pcg=0.9):
        super().__init__(trading_horizon, init_forecast, real, update_forecast_every)

        self.capacity = capacity
        self.capacity_pcg = capacity_pcg

        # constant in the error function
        self.error_constant = error_constant if error_constant is not None else len(self.trading_horizon)

    def error_range(self, t):
        return self.capacity_pcg * self.capacity * np.cos(
            self.trading_horizon.get_loc(t) / (len(self.trading_horizon) / (2 * self.error_constant)))


class SinForecast(AdditiveErrorForecast):

    def __init__(self, trading_horizon, capacity, init_forecast, real, update_forecast_every="15min",
                 error_constant=None, capacity_pcg=0.9):
        super().__init__(trading_horizon, init_forecast, real, update_forecast_every)

        self.capacity = capacity

        self.capacity_pcg = capacity_pcg

        # constant in the error function
        self.error_constant = error_constant if error_constant is not None else len(self.trading_horizon)

    def error_range(self, t):
        return self.capacity_pcg * self.capacity * np.sin(
            self.trading_horizon.get_loc(t) / (len(self.trading_horizon) / (2 * self.error_constant)))


class ConstantOffsetForecast(AdditiveErrorForecast):

    def __init__(self, trading_horizon, capacity, init_forecast, real, update_forecast_every="15min",
                 error_constant=None, capacity_pcg=0.2):
        super().__init__(trading_horizon, init_forecast, real, update_forecast_every)

        self.capacity = capacity
        self.capacity_pcg = capacity_pcg

        # constant in the error function
        self.error_constant = error_constant if error_constant is not None else len(self.trading_horizon)

    def error_range(self, t):
        return self.capacity_pcg * self.capacity
# time t?


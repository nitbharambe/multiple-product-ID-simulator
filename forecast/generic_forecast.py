from copy import deepcopy


class Forecast:

    def __init__(self):
        self.history_forecasts = []

    def realization(self, delivery_time=None):
        raise Exception('Unimplemented class')

    def forecast(self, time, init_step_flag, delivery_time=None):
        raise Exception('Unimplemented class')

    def reset(self, delivery_time=None):
        raise Exception('Unimplemented class')

    def history(self):
        return deepcopy(self.history_forecasts)

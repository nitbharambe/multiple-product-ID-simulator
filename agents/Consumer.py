from agents.RES_agent import RES


class Consumer(RES):

    def __init__(self, ID, trading_horizon, da_position, capacity, range_imb, limit_price_sell, limit_price_buy,
                 w_update_limits,
                 init_forecast, realization, eqbm_price, forecast_type="Constant", update_forecast_every="10min",
                 strategy="naive", n_orders=10, error_constant=None, out_probability=0., out_percentage=0.,
                 out_time=0.,product_number=0, aon_trader=False, switch_strategies = 0):
        super().__init__(ID=ID, trading_horizon=trading_horizon, da_position=da_position, capacity=capacity,
                         range_imb=range_imb,
                         limit_price_sell=limit_price_sell, limit_price_buy=limit_price_buy,
                         w_update_limits=w_update_limits, init_forecast=init_forecast, forecast_type=forecast_type,
                         realization=realization, error_constant=error_constant, eqbm_price=eqbm_price,
                         update_forecast_every=update_forecast_every, strategy=strategy, n_orders=n_orders,
                         out_probability=out_probability, out_percentage=out_percentage, out_time=out_time, aon_trader=aon_trader,
                         switch_strategies=switch_strategies)

        # read the historical data
        # df = pd.read_csv('historical_data/processed_data/load_imbalance_quarter.csv')
        # times at which a quarter product is deliver in the day
        # prod_delivery = pd.date_range(pd.Timestamp("00:00"), pd.Timestamp("23:45"), freq="15min")
        # time of the product we trade
        # product_del_traded = prod_delivery[-1] + pd.Timedelta("30min")
        # forecast object for imbalance DA IA
        # self.forecaster = (GMPForecast(trading_horizon, product_del_traded, forecast_sign=-1)
        #                    .fit(df, "Actual Total Load [MW] - BZN|DE-LU", "Quarter", prod_delivery))
        # self.recent_forecast = self.forecaster.reset()

    @staticmethod
    def name():
        return "Consumer"

    def da_trade(self):
        quantity_forecast = self.recent_forecast
        print(quantity_forecast)
        if quantity_forecast <= 0:
            return "DEMAND", abs(quantity_forecast), self.limit_price_buy
        else:
            return "SUPPLY", abs(quantity_forecast), self.limit_price_sell

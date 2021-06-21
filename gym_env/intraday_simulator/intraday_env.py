import json

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from agents.Market_operator import MarketOperator


def gather_action_dimension(simulator):
    return np.ones(2), np.ones(2)


def gather_space_dimension(simulator):
    return np.ones(2), np.ones(2)


class IntradayEnv(gym.Env):

    def __init__(self, trader_config_path="data/trader_config.json", mo_config_path="data/market_operator_config.json"):
        with open(trader_config_path) as json_file:
            config_traders = json.load(json_file)

        with open(mo_config_path) as json_file:
            config_market_operator = json.load(json_file)

        self.simulator = MarketOperator(config_traders, config_market_operator)

        self.action_space = make_action_space(simulator=self.simulator)
        self.observation_space = make_observation_space(simulator=self.simulator)
        self.env_step = None
        self.state = None
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.env_step = 0.
        self.state = self.simulator.reset()
        return self.state  # self._observation(self.state)

    def step(self, action, current_time=""):
        self.state, reward = self.simulator.step(action, current_time)
        self.env_step += 1
        done = current_time == self.simulator.trading_horizon[-1]
        info = None
        return self.state, reward, done, info  # self._observation(self.state), reward, done, info

    @staticmethod
    def _observation(state):
        return np.array(state, np.float32)


def make_action_space(simulator):
    lower, upper = gather_action_dimension(simulator)
    action_space = spaces.Box(lower, upper, dtype=np.float32)
    return action_space


def make_observation_space(simulator):
    lower, upper = gather_space_dimension(simulator)
    observation_space = spaces.Box(lower, upper, dtype=np.float32)
    return observation_space


if __name__ == '__main__':
    env = IntradayEnv()
    s = env.reset()

    for current_time in env.simulator.trading_horizon:  # Loop through each timestep in the trading horizon
        print(current_time, "############################################")

        ids = list(env.simulator.traders.keys())
        # random.shuffle(ids)  # Randomize the order in which the traders submit their orders
        for id in ids:  # Loo through the agents
            print(s)
            tr = env.simulator.traders[id]
            env.simulator.update_agents_beliefs(current_time, s, None)  # First update their beliefs
            tr.process_signal(current_time,
                              s)  # Based on the tob and their belief: 1) cancel previous orders if any 2) submit new orders in the market

            n_s, r, d, i = env.step((tr.cancel_collector, tr.quote_collector), current_time=current_time)
            tr.store_cum_position()
            # Save the final  position of the agent after processing all the orders
            s = n_s

        [print(id, trader._position, trader._cash_flow) for id, trader in env.simulator.traders.items()]

    env.simulator.imbalance_settlement()

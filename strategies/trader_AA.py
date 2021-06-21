'''
Created on 1 Dec 2012
@author: Ash Booth
AA order execution strategy as described in: "Perukrishnen, Cliff and Jennings (2008)
'Strategic Bidding in Continuous Double Auctions'. Artificial Intelligence Journal,
172, (14), 1700-1729".
    With notable...
    Amendments:
    - slightly modified equilibrium price updating
    - spin up period instead of rounds
    Additions:
    - Includes functions for using Newton-Rhapson method for finding
      complementary theta values.
'''
import logging
import math
import random
import numpy as np

from mmabm.shared import Side
from strategies.generic import Strategy


class Trader_AA(Strategy, object):

    def __init__(self, eqlbm, limit_buy, limit_sell):
        super().__init__(eqlbm, limit_buy, limit_sell)
        # External parameters (you must choose [optimise] values yourselves)
        self.spin_up_time = 20
        self.eta = 2.0
        self.theta_max = 2.0
        self.theta_min = -8.0
        self.lambda_a = 0.1
        self.lambda_r = 0.2
        self.beta_1 = 0.4
        self.beta_2 = 0.4
        self.gamma = 2.0
        self.nLastTrades = 2  # N in AIJ08
        self.ema_param = 2 / float(self.nLastTrades + 1)
        self.maxNewtonItter = 10
        self.maxNewtonError = 0.0001

        # The order we're trying to trade
        self.orders = []
        self.active = False
        self.job = None

        # Parameters describing what the market looks like and it's contstraints
        self.marketMax = 200
        self.marketMin = -200
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

        # Internal parameters (spin up time need to get values for some of these)
        self.theta = -1.0 * (5.0 * random.random())
        self.smithsAlpha = None
        self.lastTrades = []  # TODO Make it a deque of fixed size
        self.smithsAlphaMin = 0.02
        self.smithsAlphaMax = 0.15
        self.phi = 50

        self.aggressiveness_buy = -1.0 * (0.3 * random.random())
        self.aggressiveness_sell = -1.0 * (0.3 * random.random())
        self.target_buy = None
        self.target_sell = None

    def updateEq(self):
        # Updates the equilibrium price estimate using EMA

        # self.eqlbm = self.ema_param * price + (1 - self.ema_param) * self.eqlbm
        if self.lastTrades:
            self.eqlbm = np.average(self.lastTrades, weights=range(1, len(self.lastTrades) + 1))

    def newton4Buying(self):
        # runs Newton-Raphson to find theta_est (the value of theta that makes the 1st
        # derivative of eqn(3) continuous)
        theta_est = self.theta
        if abs(self.limit_buy - self.eqlbm) > 1:
            rightHside = ((self.theta * (self.limit_buy - self.eqlbm)) / float(math.exp(self.theta) - 1))
            i = 0
            while i <= self.maxNewtonItter:

                eX = math.exp(theta_est)
                eXminOne = eX - 1
                fofX = (((theta_est * self.eqlbm) / float(eXminOne)) - rightHside)
                if abs(fofX) <= self.maxNewtonError:
                    break
                dfofX = ((self.eqlbm / eXminOne) - ((eX * self.eqlbm * theta_est) / float(eXminOne * eXminOne)))
                theta_est = (theta_est - (fofX / float(dfofX)))
                i += 1
            if theta_est == 0.0: theta_est += 0.000001
        return theta_est

    def newton4Selling(self):
        # runs Newton-Raphson to find theta_est (the value of theta that makes the 1st
        # derivative of eqn(4) continuous)
        theta_est = self.theta
        rightHside = ((self.theta * (self.eqlbm - self.limit_sell)) / float(math.exp(self.theta) - 1))
        i = 0
        while i <= self.maxNewtonItter:
            eX = math.exp(theta_est)
            eXminOne = eX - 1
            fofX = (((theta_est * (self.marketMax - self.eqlbm)) / float(eXminOne)) - rightHside)
            if abs(fofX) <= self.maxNewtonError:
                break
            dfofX = (((self.marketMax - self.eqlbm) / eXminOne) - (
                    (eX * (self.marketMax - self.eqlbm) * theta_est) / float(eXminOne * eXminOne)))
            theta_est = (theta_est - (fofX / float(dfofX)))
            i += 1
        if theta_est == 0.0: theta_est += 0.000001

        return theta_est

    def updateTarget(self):
        # relates to eqns (3),(4),(5) and (6)
        # For buying
        if self.limit_buy < self.eqlbm:
            # Extra-marginal buyer
            if self.aggressiveness_buy >= 0:
                target = self.limit_buy
            else:
                target = self.limit_buy * (
                        1 - (math.exp(-self.aggressiveness_buy * self.theta) - 1) / float(math.exp(self.theta) - 1))
            self.target_buy = target

        else:
            # Intra-marginal buyer
            if self.aggressiveness_buy >= 0:
                target = (self.eqlbm + (self.limit_buy - self.eqlbm) * (
                        (math.exp(self.aggressiveness_buy * self.theta) - 1) / float(math.exp(self.theta) - 1)))
            else:
                theta_est = self.newton4Buying()
                target = self.eqlbm * (
                        1 - (math.exp(-self.aggressiveness_buy * theta_est) - 1) / float(math.exp(theta_est) - 1))
            self.target_buy = target
        # For selling
        if self.limit_sell > self.eqlbm:
            # Extra-marginal seller
            if self.aggressiveness_sell >= 0:
                target = self.limit_sell
            else:
                target = self.limit_sell + (self.marketMax - self.limit_sell) * (
                        (math.exp(-self.aggressiveness_sell * self.theta) - 1) / float(math.exp(self.theta) - 1))
            self.target_sell = target
        else:
            # Intra-marginal seller
            if self.aggressiveness_sell >= 0:
                target = self.limit_sell + (self.eqlbm - self.limit_sell) * (
                        1 - (math.exp(self.aggressiveness_sell * self.theta) - 1) / float(math.exp(self.theta) - 1))
            else:
                theta_est = self.newton4Selling()
                target = self.eqlbm + (self.marketMax - self.eqlbm) * (
                        (math.exp(-self.aggressiveness_sell * theta_est) - 1) / (math.exp(theta_est) - 1))
            self.target_sell = target

    def calcRshout(self, target, buying):
        if buying:
            # Are we extramarginal?
            if self.eqlbm >= self.limit_buy:
                r_shout = 0.0
            else:  # Intra-marginal
                if target > self.eqlbm:
                    if target > self.limit_buy:
                        target = self.limit_buy

                    r_shout = math.log((((target - self.eqlbm) * (math.exp(self.theta) - 1)) / (
                            self.limit_buy - self.eqlbm)) + 1) / self.theta
                else:  # other formula for intra buyer
                    r_shout = math.log(
                        (1 - (target / self.eqlbm)) * (
                                math.exp(self.newton4Buying()) - 1) + 1) / -self.newton4Buying()
        else:  # Selling
            # Are we extra-marginal?
            if self.limit_sell >= self.eqlbm:
                r_shout = 0.0
            else:  # Intra-marginal
                if target > self.eqlbm:
                    r_shout = math.log(((target - self.eqlbm) * (math.exp(self.newton4Selling()) - 1)) / (self.marketMax - self.eqlbm) + 1) / -self.newton4Selling()

                else:  # other intra seller formula
                    if target < self.limit_sell: target = self.limit_sell
                    r_shout = math.log((1 - (target - self.limit_sell) / (self.eqlbm - self.limit_sell)) * (
                            math.exp(self.theta) - 1) + 1) / self.theta

        return r_shout

    def updateAgg(self, up, buying, target):
        if buying:
            old_agg = self.aggressiveness_buy
        else:
            old_agg = self.aggressiveness_sell
        if up:
            delta = (1 + self.lambda_r) * self.calcRshout(target, buying) + self.lambda_a
        else:
            delta = (1 - self.lambda_r) * self.calcRshout(target, buying) - self.lambda_a
        new_agg = old_agg + self.beta_1 * (delta - old_agg)
        if new_agg > 1.0:
            new_agg = 1.0
        elif new_agg < -1.0:
            new_agg = -1.0

        return new_agg

    def updateSmithsAlpha(self):

        if not (len(self.lastTrades) <= self.nLastTrades): self.lastTrades.pop(0)
        self.smithsAlpha = math.sqrt(
            sum(((p - self.eqlbm) ** 2) for p in self.lastTrades) * (1 / float(len(self.lastTrades)))) / self.eqlbm
        if self.smithsAlphaMin == None:
            self.smithsAlphaMin = self.smithsAlpha
            self.smithsAlphaMax = self.smithsAlphaMax
        else:
            if self.smithsAlpha < self.smithsAlphaMin: self.smithsAlphaMin = self.smithsAlpha
            if self.smithsAlpha > self.smithsAlphaMax: self.smithsAlphaMax = self.smithsAlpha

    def updateTheta(self):
        alphaBar = (self.smithsAlpha - self.smithsAlphaMin) / (self.smithsAlphaMax - self.smithsAlphaMin)
        desiredTheta = (self.theta_max - self.theta_min) * (
                1 - (alphaBar * math.exp(self.gamma * (alphaBar - 1)))) + self.theta_min
        theta = self.theta + self.beta_2 * (desiredTheta - self.theta)
        if theta == 0:    theta += 0.0000001
        self.theta = theta

    def getorder(self, limit_price_sell, limit_price_buy, side, init_step_flag, tob, volume, n_orders):

        self.active = True
        self.limit_sell = limit_price_sell
        self.limit_buy = limit_price_buy
        self.job = side
        self.updateTarget()
        if self.job == Side.BID:
            # currently a buyer (working a bid order)
            if init_step_flag:
                ask_plus = (1 + self.lambda_r) * self.prev_best_ask_p + self.lambda_a
                quoteprice = self.prev_best_bid_p + (min(self.limit_buy, ask_plus) - self.prev_best_bid_p) / self.eta
            else:
                if (self.target_buy - self.prev_best_bid_p) > 0:
                    quoteprice = self.prev_best_bid_p + (self.target_buy - self.prev_best_bid_p) / self.eta
                else:
                    quoteprice = self.prev_best_bid_p + (self.target_buy - self.prev_best_bid_p)

            step = volume / n_orders
            price_distribution = [self.get_price_distribution_buyer(V, quoteprice,
                                                                    self.marketMin if quoteprice < tob["vwap_buy"] else
                                                                    tob[
                                                                        "vwap_buy"]) for V
                                  in np.arange(0, 1, step / volume)]
        else:
            # currently a seller (working a sell order)
            if init_step_flag:
                bid_minus = (1 - self.lambda_r) * self.prev_best_bid_p - self.lambda_a
                quoteprice = self.prev_best_ask_p - (self.prev_best_ask_p - max(self.limit_sell, bid_minus)) / self.eta
            else:
                if (self.prev_best_ask_p - self.target_sell) > 0:
                    quoteprice = (self.prev_best_ask_p - (self.prev_best_ask_p - self.target_sell) / self.eta)
                else:
                    quoteprice = self.target_sell

            step = volume / n_orders
            price_distribution = [self.get_price_distribution_seller(V, quoteprice,
                                                                     self.marketMax if quoteprice > tob["vwap_ask"] else
                                                                     tob[
                                                                         "vwap_ask"]) for V
                                  in np.arange(0, 1, step / volume)]
        return [step] * n_orders, price_distribution

    def get_price_distribution_seller(self, V, quoteprice, max_price):
        return (quoteprice + (max_price - quoteprice) * (
                (math.exp(V * (self.aggressiveness_sell * self.phi)) - 1) / float(math.exp(self.aggressiveness_sell
                                                                                           * self.phi) - 1)))

    def get_price_distribution_buyer(self, V, quoteprice, min_price):
        return (quoteprice + (min_price - quoteprice) * (
                (math.exp(V * (self.aggressiveness_buy * self.phi)) - 1) / float(math.exp(self.aggressiveness_buy
                                                                                          * self.phi) - 1)))

    def respond(self, time, tob, transactions, verbose=True):
        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False
        lob_best_bid_p = tob["best_bid"]
        lob_best_bid_q = None
        if lob_best_bid_p != None:
            # non-empty bid LOB
            lob_best_bid_q = tob["bid_size"]
            if self.prev_best_bid_p < lob_best_bid_p:
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif transactions != None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p != None:
            # the bid LOB has been emptied by a hit
            bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        lob_best_ask_p = tob["best_ask"]
        lob_best_ask_q = None
        if lob_best_ask_p != None:
            # non-empty ask LOB
            lob_best_ask_q = tob["ask_size"]
            if self.prev_best_ask_p > lob_best_ask_p:
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif transactions != None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced -- assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p != None:
            # the bid LOB is empty now but was not previously, so must have been hit
            ask_lifted = True

        if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
            logging.info('B_improved %s, B_hit %s, A_improved %s, A_lifted %s' % (
            bid_improved, bid_hit, ask_improved, ask_lifted))

        deal = bid_hit or ask_lifted
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_q = lob_best_ask_q
        if self.spin_up_time > 0: self.spin_up_time -= 1
        if deal:
            self.lastTrades = [transaction['price'] for transaction in transactions]
            price = self.lastTrades[-1]
            self.updateEq()
            self.updateSmithsAlpha()
            self.updateTheta()

        if deal:
            if self.target_buy >= price:
                self.aggressiveness_buy = self.updateAgg(False, True, price)
            else:
                self.aggressiveness_buy = self.updateAgg(True, True, price)
        elif bid_improved and (self.target_buy <= lob_best_bid_p):
            self.aggressiveness_buy = self.updateAgg(True, True, self.prev_best_bid_p)
        # For selling
        if deal:
            if self.target_sell <= price:
                self.aggressiveness_sell = self.updateAgg(False, False, price)
            else:
                self.aggressiveness_sell = self.updateAgg(True, False, price)
        elif ask_improved and (self.target_sell >= lob_best_ask_p):
            self.aggressiveness_sell = self.updateAgg(True, False, self.prev_best_ask_p)

        self.updateTarget()

    def initialize(self, tob):

        self.prev_best_bid_p = tob["best_bid"]
        self.prev_best_bid_q = tob["bid_size"]
        self.prev_best_ask_p = tob["best_ask"]
        self.prev_best_ask_q = tob["ask_size"]

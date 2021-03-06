import random

import numpy as np

from math import floor, log
from mmabm.shared import Side, OType, TType


class ZITrader:
    '''
    ZITrader generates quotes (dicts) based on mechanical probabilities.
    
    A general base class for specific trader types.
    Public attributes: quote_collector
    Public methods: none
    '''
    trader_type = TType.ZITrader

    def __init__(self, name, maxq):
        '''
        Initialize ZITrader with some base class attributes and a method
        
        quote_collector is a public container for carrying quotes to the exchange
        '''
        self.trader_id = name  # trader id
        self.quantity = self._make_q(maxq)
        self.quote_collector = []
        self._quote_sequence = 0

    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2})'.format(class_name, self.trader_id, self.quantity)

    def __str__(self):
        return str(tuple([self.trader_id, self.quantity]))

    def _make_q(self, maxq):
        '''Determine order size'''
        default_arr = np.array([1, 5, 10, 25, 50])
        return random.choice(default_arr[default_arr <= maxq])

    def _make_add_quote(self, time, side, price):
        '''Make one add quote (dict)'''
        self._quote_sequence += 1
        return {'order_id': self._quote_sequence, 'trader_id': self.trader_id, 'timestamp': time,
                'type': OType.ADD, 'quantity': self.quantity, 'side': side, 'price': price}


class Provider(ZITrader):
    '''
    Provider generates quotes (dicts) based on make probability.
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector, local_book
    Public methods: confirm_cancel_local, confirm_trade_local, process_signal, bulk_cancel
    '''
    trader_type = TType.Provider

    def __init__(self, name, maxq, delta, pAlpha):
        '''Provider has own delta; a local_book to track outstanding orders and a 
        cancel_collector to convey cancel messages to the exchange.
        '''
        super().__init__(name, maxq)
        self._delta = delta
        self.delta_t = self._make_delta(pAlpha)
        self.local_book = {}
        self.cancel_collector = []

    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2}, {3})'.format(class_name, self.trader_id, self.quantity, self._delta)

    def __str__(self):
        return str(tuple([self.trader_id, self.quantity, self._delta]))

    def _make_delta(self, pAlpha):
        return int(floor(random.expovariate(pAlpha) + 1) * self.quantity)

    def _make_cancel_quote(self, q, time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': q['trader_id'],
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}

    def confirm_trade_local(self, confirm):
        to_modify = self.local_book[confirm['order_id']]
        if confirm['quantity'] == to_modify['quantity']:
            del self.local_book[to_modify['order_id']]
        else:
            self.local_book[confirm['order_id']]['quantity'] -= confirm['quantity']

    def bulk_cancel(self, time):
        '''bulk_cancel cancels _delta percent of outstanding orders'''
        self.cancel_collector.clear()
        for x in self.local_book.keys():
            if random.random() < self._delta:
                self.cancel_collector.append(self._make_cancel_quote(self.local_book[x], time))
        for c in self.cancel_collector:
            del self.local_book[c['order_id']]

    def process_signal(self, time, qsignal, q_provider, lambda_t):
        '''Provider buys or sells with probability related to q_provide'''
        if random.random() < q_provider:
            side = Side.BID
            price = self._choose_price_from_exp(side, qsignal['best_ask'], lambda_t)
        else:
            side = Side.ASK
            price = self._choose_price_from_exp(side, qsignal['best_bid'], lambda_t)
        q = self._make_add_quote(time, side, price)
        self.local_book[q['order_id']] = q
        return q

    def _choose_price_from_exp(self, side, inside_price, lambda_t):
        '''Prices chosen from an exponential distribution'''
        # make pricing explicit for now. Logic scales for other mpi.
        plug = int(lambda_t * log(random.random()))
        if side == Side.BID:
            return inside_price - 1 - plug
        else:
            return inside_price + 1 + plug


class MarketMaker(Provider):
    '''
    MarketMaker generates a series of quotes near the inside (dicts) based on make probability.
    
    Subclass of Provider
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector (from Provider),
    cash_flow_collector
    Public methods: confirm_cancel_local (from Provider), confirm_trade_local, process_signal 
    '''
    trader_type = TType.MarketMaker

    def __init__(self, name, maxq, pAlpha, delta, num_quotes, quote_range):
        '''_num_quotes and _quote_range determine the depth of MM quoting;
        _position and _cashflow are stored MM metrics
        '''
        super().__init__(name, maxq, delta, pAlpha)
        self._num_quotes = num_quotes
        self._quote_range = quote_range
        self._position = 0
        self._cash_flow = 0
        self.cash_flow_collector = []

    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2}, {3}, {4}, {5})'.format(class_name, self.trader_id, self.quantity, self._delta,
                                                     self._num_quotes, self._quote_range)

    def __str__(self):
        return str(tuple([self.trader_id, self.quantity, self._delta, self._num_quotes, self._quote_range]))

    def confirm_trade_local(self, confirm):
        '''Modify _cash_flow and _position; update the local_book'''

        if confirm['side'] == Side.BID:
            revenue = -confirm['price'] * confirm['quantity']
            self._cash_flow += revenue
            self._position -= confirm['quantity']
        else:
            revenue = confirm['price'] * confirm['quantity']
            self._cash_flow += revenue
            self._position += confirm['quantity']
        if confirm['order_id'] not in self.local_book.keys():
            print(confirm)
        to_modify = self.local_book[confirm['order_id']]
        if confirm['quantity'] == to_modify['quantity']:
            del self.local_book[to_modify['order_id']]
        else:
            self.local_book[confirm['order_id']]['quantity'] -= confirm['quantity']
        self._cumulate_cashflow(confirm['timestamp'])

        return revenue

    def _cumulate_cashflow(self, timestamp):
        self.cash_flow_collector.append({'mmid': self.trader_id, 'timestamp': timestamp, 'cash_flow': self._cash_flow,
                                         'position': self._position})

    def process_signal(self, time, qsignal, q_provider):
        '''
        MM chooses prices from a grid determined by the best prevailing prices.
        MM never joins the best price if it has size=1.
        '''
        # make pricing explicit for now. Logic scales for other mpi and quote ranges.
        self.quote_collector.clear()
        if random.random() < q_provider:
            max_bid_price = qsignal['best_bid'] if qsignal['bid_size'] > 1 else qsignal['best_bid'] - 1
            prices = np.random.choice(range(max_bid_price - self._quote_range + 1, max_bid_price + 1),
                                      size=self._num_quotes)
            side = Side.BID
        else:
            min_ask_price = qsignal['best_ask'] if qsignal['ask_size'] > 1 else qsignal['best_ask'] + 1
            prices = np.random.choice(range(min_ask_price, min_ask_price + self._quote_range), size=self._num_quotes)
            side = Side.ASK
        for price in prices:
            q = self._make_add_quote(time, side, price)
            self.local_book[q['order_id']] = q
            self.quote_collector.append(q)


class PennyJumper(ZITrader):
    '''
    PennyJumper jumps in front of best quotes when possible
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector
    Public methods: confirm_trade_local (from ZITrader)
    '''
    trader_type = TType.PennyJumper

    def __init__(self, name, maxq, mpi):
        '''
        Initialize PennyJumper
        
        cancel_collector is a public container for carrying cancel messages to the exchange
        PennyJumper tracks private _ask_quote and _bid_quote to determine whether it is alone
        at the inside or not.
        '''
        super().__init__(name, maxq)
        self._mpi = mpi
        self.cancel_collector = []
        self._ask_quote = None
        self._bid_quote = None

    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2}, {3})'.format(class_name, self.trader_id, self.quantity, self._mpi)

    def __str__(self):
        return str(tuple([self.trader_id, self.quantity, self._mpi]))

    def _make_cancel_quote(self, q, time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': q['trader_id'],
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}

    def confirm_trade_local(self, confirm):
        '''PJ has at most one bid and one ask outstanding - if it executes, set price None'''
        if confirm['side'] == Side.BID:
            self._bid_quote = None
        else:
            self._ask_quote = None

    def process_signal(self, time, qsignal, q_taker):
        '''PJ determines if it is alone at the inside, cancels if not and replaces if there is an available price 
        point inside the current quotes.
        '''
        self.quote_collector.clear()
        self.cancel_collector.clear()
        if qsignal['best_ask'] - qsignal['best_bid'] > self._mpi:
            # q_taker > 0.5 implies greater probability of a buy order; PJ jumps the bid
            if random.random() < q_taker:
                if self._bid_quote:  # check if not alone at the bid
                    if self._bid_quote['price'] < qsignal['best_bid'] or self._bid_quote['quantity'] < qsignal[
                        'bid_size']:
                        self.cancel_collector.append(self._make_cancel_quote(self._bid_quote, time))
                        self._bid_quote = None
                if not self._bid_quote:
                    self._bid_quote = self._make_add_quote(time, Side.BID, qsignal['best_bid'] + self._mpi)
                    self.quote_collector.append(self._bid_quote)
            else:
                if self._ask_quote:  # check if not alone at the ask
                    if self._ask_quote['price'] > qsignal['best_ask'] or self._ask_quote['quantity'] < qsignal[
                        'ask_size']:
                        self.cancel_collector.append(self._make_cancel_quote(self._ask_quote, time))
                        self._ask_quote = None
                if not self._ask_quote:
                    self._ask_quote = self._make_add_quote(time, Side.ASK, qsignal['best_ask'] - self._mpi)
                    self.quote_collector.append(self._ask_quote)
        else:  # spread = mpi
            if self._bid_quote:  # check if not alone at the bid
                if self._bid_quote['price'] < qsignal['best_bid'] or self._bid_quote['quantity'] < qsignal['bid_size']:
                    self.cancel_collector.append(self._make_cancel_quote(self._bid_quote, time))
                    self._bid_quote = None
            if self._ask_quote:  # check if not alone at the ask
                if self._ask_quote['price'] > qsignal['best_ask'] or self._ask_quote['quantity'] < qsignal['ask_size']:
                    self.cancel_collector.append(self._make_cancel_quote(self._ask_quote, time))
                    self._ask_quote = None


class Taker(ZITrader):
    '''
    Taker generates quotes (dicts) based on take probability.
        
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader)
    Public methods: process_signal 
    '''
    trader_type = TType.Taker

    def __init__(self, name, maxq, tMu):
        super().__init__(name, maxq)
        self.delta_t = self._make_delta(tMu)

    def _make_delta(self, tMu):
        return int(floor(random.expovariate(tMu) + 1) * self.quantity)

    def process_signal(self, time, q_taker):
        '''Taker buys or sells with 50% probability.'''
        if random.random() < q_taker:  # q_taker > 0.5 implies greater probability of a buy order
            return self._make_add_quote(time, Side.BID, 2000000)
        else:
            return self._make_add_quote(time, Side.ASK, 0)


class InformedTrader(ZITrader):
    '''
    InformedTrader generates quotes (dicts) based upon a fixed direction
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader)
    Public methods: process_signal
    '''
    trader_type = TType.Informed

    def __init__(self, name, maxq, informedTrades, informedRunLength, start, stop):
        ZITrader.__init__(self, name, maxq)
        self._side = random.choice([Side.BID, Side.ASK])
        self._price = 0 if self._side == Side.ASK else 2000000
        self.delta_t = self._make_delta(informedTrades, informedRunLength, start, stop)

    def _make_delta(self, informedTrades, informedRunLength, start, stop):
        numChoices = int(informedTrades / (informedRunLength * self.quantity)) + 1
        choiceRange = range(start, stop - informedRunLength + 1)
        delta_t = set()
        for _ in range(1, numChoices):
            runL = 0
            step = random.choice(choiceRange)
            while runL < informedRunLength:
                while step in delta_t:
                    step += 1
                delta_t.add(step)
                step += 1
                runL += 1
        return delta_t

    def process_signal(self, time):
        '''InformedTrader buys or sells pre-specified attribute.'''
        return self._make_add_quote(time, self._side, self._price)

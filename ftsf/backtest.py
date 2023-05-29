'''This submodule contains instrument to test trading strategy based on model on historical data.'''

import numpy as np

from .model import Model
from .scaler import Scaler

class BackTesting:
    '''
    Models trading strategy and backtests it.

    Attributes:

        initial_money: Initial amount of money to work with.

        money: Current amount of money.

        stocks: Current amount of stocks.

        model: Instance of model used for forecasting.
    '''
    __initial_money = 0
    __money = 0
    __stocks = 0
    __model = None

    def __init__(self, model : Model, initial_money : float):
        '''
        Initializes money and model.
        '''
        self.__model = model
        self.__initial_money = initial_money

    def __sell(self, price : float):
        '''
        Internal method to model stocks selling.
        '''
        volume = self.__stocks
        self.__money += volume*price
        self.__stocks -= volume

    def __buy(self, price : float):
        '''
        Internal method to model stocks buying.
        '''
        volume = int(self.__money / price)
        self.__money -= volume*price
        self.__stocks += volume

    def __predict_steps(self, current_state : list, depth = 15, steps_forward = 1) -> float:
        '''
        Internal method to forecast for a few steps forward.
        '''
        current_state = current_state.copy()
        cs = Scaler().fit(current_state)
        for i in range(steps_forward):
            current_state += self.__model.predict(cs.scale(np.array(current_state[-depth:]).reshape((1,depth,1))), cs).reshape(-1).tolist()
        return current_state[-1]

    def test(self, states, depth = 15, steps_forward = 5):
        '''
        Implement strategy based on model forecasts, then backtests it.

        Args:
            states (list): Array of historic values.
            depth (int): Number of values to forecast on.
            steps_forward (int): Number of steps to forecast.

        Example:
        >>> str = Backtesting(model, 10e4)
        >>> str.test(states, 15, 1)
        Successfully tested: 12532
        +2532 or 25.32% in 50 days.
        2 trades, avg profit 1266 $ per trade.
        '''
        self.__money = self.__initial_money
        trades_no = 0
        for i in range(len(states)-depth):
            current_state = states[i:i+depth]
            if (self.__predict_steps(current_state, depth, steps_forward) > current_state[-1]):
                if self.__money > current_state[-1]:
                    self.__buy(current_state[-1])
                    print(f'Bought {self.__stocks} stocks.')
                    bought = current_state[-1]
                    trades_no +=1
            else:
                if self.__stocks > 0:
                    print(f'Sold {self.__stocks} stocks.')
                    print(f'Delta = {current_state[-1] - bought}/stock')
                    self.__sell(current_state[-1])
        self.__sell(current_state[-1])
        print(f'Successfully tested: \n{self.__money }$')
        print(f'{"+" if self.__money - self.__initial_money > 0 else ""}{round(self.__money - self.__initial_money)}$ or \
        {"+" if self.__money - self.__initial_money>0 else ""}{round((self.__money - self.__initial_money)/self.__initial_money*100)}% in {len(states)} days.')
        print(f'{trades_no}, avg profit {(self.__money - self.__initial_money)/trades_no}$ per trade.')

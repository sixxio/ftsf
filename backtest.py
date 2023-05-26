from .scaler import Scaler
from .model import Model
import numpy as np

class BackTesting:
    money = 0
    stocks = 0
    model = None

    def __init__(self, model : Model, initial_money : float):
        self.model = model
        self.money = initial_money

    def sell(self, price : float):
        volume = self.stocks
        self.money += volume*price
        self.stocks -= volume

    def buy(self, price : float):
        volume = int(self.money / price)
        self.money -= volume*price
        self.stocks += volume

    def predict(self, current_state : list, horizon : int) -> float:
        current_state = current_state.copy()
        cs = Scaler().fit(current_state)
        for i in range(horizon):
            current_state += cs.unscale(self.model.predict(cs.scale(np.array(current_state[-20:]).reshape((1,20,1)))).reshape(-1)).tolist()
        return current_state[-1]  

    def test(self, states, horizon = 5):
        self.initial_money = self.money
        trades_no = 0
        for i in range(len(states)-horizon):
            current_state = states[i:i+horizon]
            if (self.predict(current_state, 1) > current_state[-1]):
                if self.money > current_state[-1]:
                    self.buy(self, current_state[-1])
                    print(f'Bought {self.stocks} stocks.')
                    bought = current_state[-1]
                    trades_no +=1
            else:
                if self.stocks > 0:
                    print(f'Sold {self.stocks} stocks.')
                    print(f'Delta = {current_state[-1] - bought}/stock')
                    self.sell(self, current_state[-1])
            print(f'{self.money = }, {self.stocks = }')
        self.sell(self, current_state[-1])
        print(f'Successfully tested: \n{self.money = }\n{self.stocks = }')
        print(f'{"+" if self.money - self.initial_money > 0 else ""}{round(self.money - self.initial_money)} or \
                {"+" if self.money - self.initial_money>0 else ""}{round((self.money - self.initial_money)/self.initial_money*100)}% in {len(states)} days.')
        print(f'{trades_no}, avg profit: {(self.money - self.initial_money)/trades_no} per trade.')
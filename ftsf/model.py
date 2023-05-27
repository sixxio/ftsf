# ML methods
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor, XGBRFRegressor

# Autoregressive methods
from statsmodels.tsa.arima.model import ARIMA

# NN methods
from tensorflow.keras import Sequential

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

from .utils import get_topologies

class Model:
    '''
    Generic ML/AR/NN based model with unified interface.


    Models:

        ML: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, SVR, CatBoostRegressor, XGBoostRegressor, XGBoostRandomForestRegressor.

        AR: ARMA, ARIMA.

        NN: CNN + SimpleRNN/LSTM/GRU, LSTM x1-3, GRU x1-3, SimpleRNN x1-3, CNN, MLP(1-3).

    Attributes:

        type: Type of model.

        backend: Type of backend used to work with model.

        name: Model name.

        model: Model object.
    '''

    __type = ''
    __backend = ''
    __name = ''
    __model = None

    def __init__(self, name, lag = 15, optimizer = 'nadam', loss = 'mse'):
        '''
        Initializes the instance of model based on defined type.
        
        Args:

            name: Defines exact model.

            lag: An integer indicating number of values to base prediction on.

            data: Data to use on autoregressive model fit.

            optimizer: Optimizer to use on compiling neural network based models.

            loss: Loss function to use on compiling neural network based models.
        
        Example:
        >>> Model('LSTM x2', 15)
        '''
        models = {'ml': ['LR', 'DTR', 'RFR', 'GBR', 'SVR', 'CBR', 'XGBR', 'XGBRFR'],
                  'ar': ['ARMA(2,1)', 'ARIMA(2,1,1)'], 
                  'nn': ['CNN + LSTM', 'LSTM x3', 'LSTM x2', 'LSTM x1', 'CNN + GRU', 'GRU x3', 'GRU x2', 'GRU x1', \
                         'CNN + SimpleRNN', 'SimpleRNN x3', 'SimpleRNN x2', 'SimpleRNN x1', 'CNN', 'MLP(3)', 'MLP(2)', 'MLP(1)']}
        
        self.__type = [key for key, value in models.items() if name in value][0]
        self.__name = name

        if self.__type == 'ml':
            if name == 'CBR':
                self.__backend = 'Catboost'
            elif name[:2] == 'XG':
                self.__backend = 'XGBoost'
            else:
                self.__backend = 'Scikit-learn'
    
            self.__model = {  'LR' :  LinearRegression(),
                            'DTR' : DecisionTreeRegressor(min_samples_leaf=5),
                            'RFR' : RandomForestRegressor(),
                            'GBR' : GradientBoostingRegressor(),
                            'SVR' : SVR(kernel='linear', epsilon=1e-3),
                            'CBR' : CatBoostRegressor(loss_function='MAPE'),
                            'XGBR': XGBRegressor(objective='reg:squarederror'),
                            'XGBRFR': XGBRFRegressor(objective = 'reg:squarederror')}[name]
        
        elif self.__type == 'ar':
            self.__backend = 'Statsmodels'
        
        elif self.__type == 'nn':
            self.__backend = 'TensorFlow.Keras'
            self.__model = Sequential(get_topologies(lag)[name])
            self.__model.compile(optimizer = optimizer, loss = loss,  metrics = ['mse', 'mae', 'mape'])

    def fit(self, x_train, y_train, epochs = 25, batch_size = 32):
        '''
        Fits model using x_train and y_train.

        Note that autoregressive models differ from other and can be fitted only on one sample of data.
        
        Args:

            x_train: Array with previous price values.

            y_train: Array with target price values.

            epochs: Number of epochs in case of using neural network.

            batch_size: Number of objects in one fit batch in case of using neural network.

        Example:
        >>> model.fit(x_train, y_train)
        '''

        if self.__type == 'nn':
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            self.__model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 0)
        elif self.__name == 'CBR':
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
            self.__model.fit(x_train, y_train, silent = True)
        elif self.__backend != 'Statsmodels':
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
            self.__model.fit(x_train, y_train)
        elif self.__backend == 'Statsmodels':
            if self.__name.find('I') != -1:
                p = float(self.__name[self.__name.find('(') + 1 : self.__name.find(',')])
                d = float(self.__name[self.__name.find(',') + 1 : self.__name.find(',', self.__name.find(',') + 1)])
                q = float(self.__name[self.__name.find(',', self.__name.find(',') + 1) + 1 : self.__name.find(')')])
            else:
                p = float(self.__name[self.__name.find('(') + 1 : self.__name.find(',')])
                d = 0
                q = float(self.__name[self.__name.find(',') + 1 : self.__name.find(')')])
            self.__model = ARIMA(x_train, order = (p, d, q), enforce_stationarity = False)
        
        return self

    def predict(self, x_test, scaler):
        '''
        Predicts and scales values using x_test and scaler.
        
        Args:

            x_test - previous price values;

            scaler - scaler will be used to scale values back.

        Returns:

            Array of predicted and scaled price values.

        Example:
        >>> model.predict(x_test, scaler)
        [123, 124, 123, 128]
        '''

        if self.__type == 'ar':
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
            return scaler.unscale(self.__model.fit().forecast(steps = 1)[0])
        elif self.__type == 'nn':
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
            return scaler.unscale(self.__model.predict(x_test, verbose = 0).reshape(-1))
        else:
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
            return scaler.unscale(self.__model.predict(x_test))

    def evaluate(self, x_test, y_test, scaler):
        '''
        Predicts and scales values using x_test and scaler, then measures MSE, MAE, MAPE and R2.
        
        Args:

            x_test: Array with previous price values.

            y_test: Array with target price values.

            scaler: Scaler to be used to scale values back.

        Returns:

            Dict with errors values.

        Example:
        >>> model.evaluate(x_test, y_test, scaler)
        {'mse': 123.123, 'mae': 123.123, 'mape': 123.123, 'r2': 123.123}
        '''

        true = scaler.unscale(y_test)
        prediction = self.predict(x_test, scaler)
        return {'mse': mean_squared_error(np.array(true), np.array(prediction)),
                'mae': mean_absolute_error(np.array(true), np.array(prediction)),
                'mape': mean_absolute_percentage_error(np.array(true), np.array(prediction)),
                'r2':  r2_score(np.array(true), np.array(prediction))}

    def summary(self):
        '''
        Shows short summary about model, its type and backend.

        Returns:
            String with short model description.
        
        Example:
        >>> model.summary()
        ML model XGBRegressor, backend is based on XGBoost.
        '''
        
        print(f'{self.__type.upper()} model {self.__name}, backend is based on {self.__backend}.')
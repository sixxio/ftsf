from .utils import get_topologies

# ML methods
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import catboost as cb, xgboost as xg

# Autoregressive methods
from statsmodels.tsa.arima.model import ARIMA

# NN methods
from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import LSTM, Dense, GRU, SimpleRNN, Conv1D, Flatten

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

class Model:
    '''
    Represents generic model with unified interface.

    Backends: 

        ML: CatBoost, XGBoost, Sklearn;

        AR: Statsmodels;

        NN:TensorFlow.Keras.

    Models:

        ML: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, SVR, CatBoostRegressor, XGBoostRegressor, XGBoostRandomForestRegressor;

        AR: ARMA, ARIMA;

        NN: CNN + SimpleRNN/LSTM/GRU, LSTM x1-3, GRU x1-3, SimpleRNN x1-3, CNN, MLP(1-3).


    Methods:

        fit(x_train, y_train) - fits model using x_train and y_train data;

        predict(x_train, y_train, x_test, x_test) - forecasts values using x_test;

        evaluate(x_test, x_test) - forecasts values using x_test, then measures MSE, MAE, MAPE and R2;

        summary() - shows short summary about model, its type and backend.

    '''

    __type = ''
    __backend = ''
    __model_name = ''
    __model = None

    def __init__(self, model_name, lag = 15, data = None, optimizer = 'nadam', loss = 'mse'):
        models = {'ml': ['LR', 'DTR', 'RFR', 'GBR', 'SVR', 'CBR', 'XGBR', 'XGBRFR'],
                  'ar': ['ARMA(2,1)', 'ARIMA(2,1,1)'], 
                  'nn': ['CNN + LSTM', 'LSTM x3', 'LSTM x2', 'LSTM x1', 'CNN + GRU', 'GRU x3', 'GRU x2', 'GRU x1', \
                         'CNN + SimpleRNN', 'SimpleRNN x3', 'SimpleRNN x2', 'SimpleRNN x1', 'CNN', 'MLP(3)', 'MLP(2)', 'MLP(1)']}
        
        self.__type = [key for key,value in models.items() if model_name in value][0]
        self.__model_name = model_name

        if self.__type == 'ml':
            if model_name == 'CBR':
                self.__backend = 'Catboost'
            elif model_name[:2] == 'XG':
                self.__backend = 'XGBoost'
            else:
                self.__backend = 'Scikit-learn'
    
            self.__model = {  'LR' :  LinearRegression(),
                            'DTR' : DecisionTreeRegressor(min_samples_leaf=5),
                            'RFR' : RandomForestRegressor(),
                            'GBR' : GradientBoostingRegressor(),
                            'SVR' : SVR(kernel='linear', epsilon=1e-3),
                            'CBR' : cb.CatBoostRegressor(loss_function='MAPE'),
                            'XGBR': xg.XGBRegressor(objective='reg:squarederror'),
                            'XGBRFR':xg.XGBRFRegressor(objective = 'reg:squarederror')}[model_name]
        
        elif self.__type == 'ar':
            self.__backend = 'Statsmodels'
            if self.__model_name.find('I') != -1:
                p = float(self.__model_name[self.__model_name.find('(')+1:self.__model_name.find(',')])
                d = float(self.__model_name[self.__model_name.find(',')+1:self.__model_name.find(',', self.__model_name.find(',')+1)])
                q = float(self.__model_name[self.__model_name.find(',', self.__model_name.find(',')+1)+1:self.__model_name.find(')')])
            else:
                p = float(self.__model_name[self.__model_name.find('(')+1:self.__model_name.find(',')])
                d = 0
                q = float(self.__model_name[self.__model_name.find(',')+1:self.__model_name.find(')')])
            self.__model = ARIMA(data, order = (p, d, q), enforce_stationarity=False)
        
        elif self.__type == 'nn':
            self.__backend = 'TensorFlow.Keras'
            self.__model = Sequential(get_topologies(lag)[model_name])
            self.__model.compile(optimizer = optimizer, loss = loss,  metrics = ['mse', 'mae', 'mape'])

    def fit(self, x_train, y_train, epochs = 25, batch_size = 32):
        '''
        Fits model using x_train and y_train.
        
        Parameters:

        x_train - previous price values;

        y_train - target price;

        epochs - number of epochs in case of using neural network;

        batch_size - number of objects in one fit batch in case of using neural network.
        '''

        if self.__type == 'nn':
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            self.__model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 0)
        elif self.__model_name == 'CBR':
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
            self.__model.fit(x_train, y_train, silent = True)
        elif self.__backend != 'statsmodels':
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
            self.__model.fit(x_train, y_train)
        
        return self

    def predict(self, x_test, scaler):
        '''
        Predicts and scales values using x_test and scaler.
        
        Args:

        x_test - previous price values;

        scaler - scaler will be used to scale values back.

        Result:

        predicted and scaled price values.
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
        
        Parameters:

        x_test - previous price values;

        scaler - scaler will be used to scale values back.

        Result:

        dict with errors values.
        '''
                
        prediction = self.predict(x_test, scaler)
        true = scaler.unscale(y_test)
        return {'mse': mean_squared_error(np.array(true), np.array(prediction)),
                'mae': mean_absolute_error(np.array(true), np.array(prediction)),
                'mape': mean_absolute_percentage_error(np.array(true), np.array(prediction)),
                'r2':  r2_score(np.array(true), np.array(prediction))}

    def summary(self):
        '''
        Shows short summary about model, its type and backend.
        '''
        
        print(f'{self.__type.upper()} model {self.__model_name}, backend is based on {self.__backend}.')
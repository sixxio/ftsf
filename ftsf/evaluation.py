# Data processing
import numpy as np, pandas as pd, requests as rq
import time, json, re

# ML methods
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import catboost as cb
import xgboost as xg

# Autoregressive methods
from statsmodels.tsa.arima.model import ARIMA

# NN methods
from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import LSTM, Dense, GRU, SimpleRNN, Conv1D, Flatten

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,  mean_absolute_percentage_error

# Muting warnings
import warnings
warnings.filterwarnings("ignore")

from .preprocessing import get_data
from .utils import get_topologies

# def evaluate_ml_models(x_train, x_test, y_train, y_test, scaler, ticker):
#     '''
#     Evaluates LR, DTR, RFR, GBR and SVR models on data.
#     '''
#     statistics = []
#     models = {  'LR' :  LinearRegression(),
#                 'DTR' : DecisionTreeRegressor(min_samples_leaf=5),
#                 'RFR' : RandomForestRegressor(),
#                 'GBR' : GradientBoostingRegressor(),
#                 'SVR' : SVR(kernel='linear', epsilon=1e-3)}
    
#     for model_description, model in models.items():
#         start_time = time.time()
#         model.fit(x_train, y_train)
#         utilized = time.time() - start_time
#         preds = model.predict(x_test)
#         statistics.append({ 'time': utilized, 
#                             'mse': mean_squared_error(scaler.unscale(y_test), scaler.unscale(preds)),
#                             'mae': mean_absolute_error(scaler.unscale(y_test), scaler.unscale(preds)),
#                             'mape': mean_absolute_percentage_error(scaler.unscale(y_test), scaler.unscale(preds)),
#                             'r2':  r2_score(scaler.unscale(y_test), scaler.unscale(preds)),
#                             'model': model_description,
#                             'ticker': ticker})
#     cbr = cb.CatBoostRegressor(loss_function='MAPE')
#     cbr.fit(x_train,y_train,silent=True)
#     xgbr = xg.XGBRegressor(objective='reg:squarederror')
#     xgbr.fit(x_train,y_train)
#     xgbrfr = xg.XGBRFRegressor(objective = 'reg:squarederror')
#     xgbrfr.fit(x_train,y_train)
#     for j,i in enumerate([cbr,xgbr,xgbrfr]):
#         preds = i.predict(x_test)
#         statistics.append({ 'time': utilized, 
#                     'mse': mean_squared_error(scaler.unscale(y_test), scaler.unscale(preds)),
#                     'mae': mean_absolute_error(scaler.unscale(y_test), scaler.unscale(preds)),
#                     'mape': mean_absolute_percentage_error(scaler.unscale(y_test), scaler.unscale(preds)),
#                     'r2':  r2_score(scaler.unscale(y_test), scaler.unscale(preds)),
#                     'model': ['CBR', 'XGBR', 'XGBRFR'][j],
#                     'ticker': ticker})
#     return statistics   



from .model import Model
def evaluate_ml_models(x_train, x_test, y_train, y_test, scaler):#, ticker):
    '''
    Evaluates LR, DTR, RFR, GBR and SVR models on data.
    '''
    statistics = []
    for i in ['LR', 'DTR', 'RFR', 'GBR', 'SVR', 'CBR', 'XGBR', 'XGBRFR']:
        current_model = Model(i)
        start_time = time.time()
        current_model.fit(x_train, y_train)
        utilized = time.time() - start_time
        pred = current_model.predict(x_test, scaler)
        true = scaler.unscale(y_test)
        statistics.append({ 'time': utilized, 
                    'mse': mean_squared_error(true, pred),
                    'mae': mean_absolute_error(true, pred),
                    'mape': mean_absolute_percentage_error(true, pred),
                    'r2':  r2_score(true, pred),
                    'model': i})#,
                    # 'ticker': ticker})
    return statistics

def evaluate_ar_models(x_train, x_test, y_train, y_test, scaler):#, ticker):
    '''
    Evaluates LR, DTR, RFR, GBR and SVR models on data.
    '''
    statistics = []
    for i in ['ARMA(2,1)', 'ARIMA(2,1,1)']:
        pred = []
        for j in range(len(y_test)):
            start_time = time.time()
            current_model = Model(i, data=x_test[j])            
            utilized = time.time() - start_time
            pred.append(current_model.predict(x_test[j], scaler))
        true = scaler.unscale(y_test)
        statistics.append({ 'time': utilized, 
                    'mse': mean_squared_error(true, pred),
                    'mae': mean_absolute_error(true, pred),
                    'mape': mean_absolute_percentage_error(true, pred),
                    'r2':  r2_score(true, pred),
                    'model': i})
                        # 'ticker': ticker})
    return statistics   

def evaluate_neural_networks(x_train, x_test, y_train, y_test, scaler, optimizer = 'nadam', loss = 'mse', epochs = 10, batch_size = 256):
    '''
    Evaluates LR, DTR, RFR, GBR and SVR models on data.
    '''
    statistics = []
    for i in ['CNN + LSTM', 'LSTM x3', 'LSTM x2', 'LSTM x1', 'CNN + GRU', 'GRU x3', 'GRU x2', 'GRU x1', \
              'CNN + SimpleRNN', 'SimpleRNN x3', 'SimpleRNN x2', 'SimpleRNN x1', 'CNN', 'MLP(3)', 'MLP(2)','MLP(1)']:
        current_model = Model(i, lag = x_train.shape[1]+1, optimizer=optimizer, loss=loss)
        start_time = time.time()
        current_model.fit(x_train, y_train, epochs, batch_size)
        utilized = time.time() - start_time
        pred = current_model.predict(x_test, scaler)
        true = scaler.unscale(y_test)
        statistics.append({ 'time': utilized, 
                    'mse': mean_squared_error(true, pred),
                    'mae': mean_absolute_error(true, pred),
                    'mape': mean_absolute_percentage_error(true, pred),
                    'r2':  r2_score(true, pred),
                    'model': i})
                    # 'ticker': ticker})
    return statistics   





# def evaluate_neural_networks(topologies_dict: dict, x_train, x_test, y_train, y_test, scaler, ticker, optimizer = 'nadam', loss = 'mse', epochs = 10, batch_size = 256) -> list:
#     '''
#     Evaluates neural networks with defined topologies with defined hyperparameters on provided data.
#     '''
#     statistics = []
#     for topology_description, topology in topologies_dict.items():
#         current_topology_model = Sequential(topology)
#         current_topology_model.compile(optimizer = optimizer, loss = loss, metrics = ['mae', 'mse'])
#         start_time = time.time()
#         current_topology_model.fit(x = x_train, y = y_train, epochs = epochs, batch_size = batch_size, verbose = 0)
#         utilized_time = time.time() - start_time
#         preds = current_topology_model.predict(x_test, verbose=0).reshape(-1)
#         statistics.append({'time':utilized_time, 
#                     'mse': mean_squared_error(scaler.unscale(y_test), scaler.unscale(preds)),
#                     'mae': mean_absolute_error(scaler.unscale(y_test), scaler.unscale(preds)),
#                     'mape': mean_absolute_percentage_error(scaler.unscale(y_test), scaler.unscale(preds)),
#                     'r2':  r2_score(scaler.unscale(y_test), scaler.unscale(preds)),
#                     'model': topology_description,
#                     'ticker': ticker})
#     return statistics

# def evaluate_autoregressive_models(x_train, x_test, y_train, y_test, scaler, ticker) -> list:
#     '''
#     Evaluates ARMA and ARIMA models on provided data.
#     '''
#     statistics = []
#     p, q = 2, 1
#     for d in range(2):
#         forecasts, true = [], []
#         for i in range(int(len(y_test)/10)):
#             start_time = time.time()
#             model = ARIMA(scaler.unscale(x_test[i]), order = (p, d, q), enforce_stationarity=False)
#             forecasts.append(model.fit().forecast(steps = 1)[0])
#             true.append(scaler.unscale(y_test[i]))
#             utilized = time.time() - start_time
#         statistics.append({'time':utilized, 
#                     'mse': mean_squared_error(np.array(true), np.array(forecasts)),
#                     'mae': mean_absolute_error(np.array(true), np.array(forecasts)),
#                     'mape': mean_absolute_percentage_error(np.array(true), np.array(forecasts)),
#                     'r2':  r2_score(np.array(true), np.array(forecasts)),
#                     'model': f'ARIMA({p},{d},{q})' if d > 0 else f'ARMA({p},{q})',
#                     'ticker' : ticker})
#     return statistics

def evaluate_all_models(tickers = [], length = 15, out_type = 'df'):
    '''
    Evaluates ml, ar and nn models on provided data.
    '''
    statistics = []
    for i in tickers:

        flat_x_train, flat_x_test, flat_y_train, flat_y_test, flat_scaler = get_data(ticker = i, length = length, flatten = True)
        statistics += evaluate_ml_models(flat_x_train, flat_x_test, flat_y_train, flat_y_test, flat_scaler, i)
        statistics +=  evaluate_autoregressive_models(flat_x_train, flat_x_test, flat_y_train, flat_y_test, flat_scaler, i)

        x_train, x_test, y_train, y_test, scaler = get_data(ticker = i, length = length)
        statistics += evaluate_neural_networks(get_topologies(length), x_train, x_test, y_train, y_test, scaler, i)

    if out_type == 'df':
        return pd.DataFrame(statistics).sort_values(by='mape')
    elif out_type == 'list':
        return statistics
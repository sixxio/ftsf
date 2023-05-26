# Data processing
import numpy as np, pandas as pd, time

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
from .scaler import CustomScaler


from pathlib import Path
PACKAGEDIR = Path(__file__).parent.absolute()

"""
The mod module
"""


def get_topologies(lag = 30):
    neurons_per_layer = lag-1
    input_shape = (neurons_per_layer, 1)
    return {  'CNN + LSTM': [Conv1D(filters = 128, kernel_size = 3, activation = 'relu', input_shape = input_shape),
                                LSTM(units = neurons_per_layer),
                                Dense(units = 1)],
                    'LSTM x3': [LSTM(units = neurons_per_layer, 
                                    return_sequences = True, 
                                    input_shape = input_shape),
                                LSTM(units = neurons_per_layer, 
                                    return_sequences = True),
                                LSTM(units = neurons_per_layer),
                                Dense(units = 1)],
                    'LSTM x2': [LSTM(units = neurons_per_layer, 
                                    return_sequences = True, 
                                    input_shape = input_shape),
                                LSTM(units = neurons_per_layer),
                                Dense(units = 1)],
                    'LSTM x1': [LSTM(units = neurons_per_layer, 
                                    input_shape = input_shape),
                                Dense(units = 1)],
                    'CNN + GRU': [Conv1D(filters = 128, kernel_size = 3, activation = 'relu', input_shape = input_shape),
                                GRU(units = neurons_per_layer),
                                Dense(units = 1)],
                    'GRU x3' : [GRU(units = neurons_per_layer, 
                                    return_sequences = True, 
                                    input_shape = input_shape),
                                GRU(units = neurons_per_layer, 
                                    return_sequences = True),
                                GRU(units = neurons_per_layer),
                                Dense(units = 1)],
                    'GRU x2' : [GRU(units = neurons_per_layer, 
                                    return_sequences = True, 
                                    input_shape = input_shape),
                                GRU(units = neurons_per_layer),
                                Dense(units = 1)],
                    'GRU x1' : [GRU(units = neurons_per_layer, 
                                    input_shape = input_shape),
                                Dense(units = 1)],
                    'CNN + SimpleRNN': [Conv1D(filters = 128, kernel_size = 3, activation = 'relu', input_shape = input_shape),
                                        SimpleRNN(units = neurons_per_layer),
                                        Dense(units = 1)],
                    'SimpleRNN x3':[SimpleRNN(units = neurons_per_layer, 
                                            return_sequences = True, 
                                            input_shape = input_shape),
                                    SimpleRNN(units = neurons_per_layer, 
                                            return_sequences = True),
                                    SimpleRNN(units = neurons_per_layer),
                                    Dense(units = 1)],
                    'SimpleRNN x2':[SimpleRNN(units = neurons_per_layer, 
                                            return_sequences = True, 
                                            input_shape = input_shape),
                                    SimpleRNN(units = neurons_per_layer),
                                    Dense(units = 1)],
                    'SimpleRNN x1':[SimpleRNN(units = neurons_per_layer, 
                                            input_shape = input_shape),
                                    Dense(units = 1)],
                    'CNN': [Conv1D(filters = 32, kernel_size = 5, input_shape = input_shape, activation = 'relu'),
                            Flatten(),
                            Dense(units = 1)],
                    'MLP(3)': [Dense(units = neurons_per_layer, input_shape = (neurons_per_layer,)),
                            Dense(units = neurons_per_layer*2),
                            Dense(units = neurons_per_layer),
                            Dense(units = 1)],        
                    'MLP(2)': [Dense(units = neurons_per_layer, input_shape = (neurons_per_layer,)),
                            Dense(units = neurons_per_layer),
                            Dense(units = 1)],
                    'MLP(1)': [Dense(units = neurons_per_layer, input_shape = (neurons_per_layer,)),
                            Dense(units = 1)],
                    }

def evaluate_ml_models(x_train, x_test, y_train, y_test, scaler):
    '''
    Evaluates LR, DTR, RFR, GBR and SVR models on data.
    '''
    statistics = []
    models = {  'LR' :  LinearRegression(),
                'DTR' : DecisionTreeRegressor(min_samples_leaf=5),
                'RFR' : RandomForestRegressor(),
                'GBR' : GradientBoostingRegressor(),
                'SVR' : SVR(kernel='linear', epsilon=1e-3)}
    
    for model_description, model in models.items():
        start_time = time.time()
        model.fit(x_train, y_train)
        utilized = time.time() - start_time
        preds = model.predict(x_test)
        statistics.append({ 'time': utilized, 
                            'mse': mean_squared_error(scaler.unscale(y_test), scaler.unscale(preds)),
                            'mae': mean_absolute_error(scaler.unscale(y_test), scaler.unscale(preds)),
                            'mape': mean_absolute_percentage_error(scaler.unscale(y_test), scaler.unscale(preds)),
                            'r2':  r2_score(scaler.unscale(y_test), scaler.unscale(preds)),
                            'model': model_description})
    cbr = cb.CatBoostRegressor(loss_function='MAPE')
    cbr.fit(x_train,y_train,silent=True)
    xgbr = xg.XGBRegressor(objective='reg:squarederror')
    xgbr.fit(x_train,y_train)
    xgbrfr = xg.XGBRFRegressor(objective = 'reg:squarederror')
    xgbrfr.fit(x_train,y_train)
    for j,i in enumerate([cbr,xgbr,xgbrfr]):
        preds = i.predict(x_test)
        statistics.append({ 'time': utilized, 
                    'mse': mean_squared_error(scaler.unscale(y_test), scaler.unscale(preds)),
                    'mae': mean_absolute_error(scaler.unscale(y_test), scaler.unscale(preds)),
                    'mape': mean_absolute_percentage_error(scaler.unscale(y_test), scaler.unscale(preds)),
                    'r2':  r2_score(scaler.unscale(y_test), scaler.unscale(preds)),
                    'model': ['CBR', 'XGBR', 'XGBRFR'][j]})
    return statistics   


def evaluate_neural_networks(topologies_dict: dict, x_train, x_test, y_train, y_test, scaler, optimizer = 'nadam', loss = 'mse', epochs = 10, batch_size = 256) -> list:
    '''
    Evaluates neural networks with defined topologies with defined hyperparameters on provided data.
    '''
    statistics = []
    for topology_description, topology in topologies_dict.items():
        current_topology_model = Sequential(topology)
        current_topology_model.compile(optimizer = optimizer, loss = loss, metrics = ['mae', 'mse'])
        start_time = time.time()
        current_topology_model.fit(x = x_train, y = y_train, epochs = epochs, batch_size = batch_size, verbose = 0)
        utilized_time = time.time() - start_time
        preds = current_topology_model.predict(x_test, verbose=0).reshape(-1)
        statistics.append({'time':utilized_time, 
                    'mse': mean_squared_error(scaler.unscale(y_test), scaler.unscale(preds)),
                    'mae': mean_absolute_error(scaler.unscale(y_test), scaler.unscale(preds)),
                    'mape': mean_absolute_percentage_error(scaler.unscale(y_test), scaler.unscale(preds)),
                    'r2':  r2_score(scaler.unscale(y_test), scaler.unscale(preds)),
                    'model': topology_description})
    return statistics

def evaluate_autoregressive_models(x_train, x_test, y_train, y_test, scaler) -> list:
    '''
    Evaluates ARMA and ARIMA models on provided data.
    '''
    statistics = []
    p, q = 2, 1
    for d in range(2):
        forecasts, true = [], []
        for i in range(int(len(y_test)/10)):
            start_time = time.time()
            model = ARIMA(scaler.unscale(x_test[i]), order = (p, d, q), enforce_stationarity=False)
            forecasts.append(model.fit().forecast(steps = 1)[0])
            true.append(scaler.unscale(y_test[i]))
            utilized = time.time() - start_time
        statistics.append({'time':utilized, 
                    'mse': mean_squared_error(np.array(true), np.array(forecasts)),
                    'mae': mean_absolute_error(np.array(true), np.array(forecasts)),
                    'mape': mean_absolute_percentage_error(np.array(true), np.array(forecasts)),
                    'r2':  r2_score(np.array(true), np.array(forecasts)),
                    'model': f'ARIMA({p},{d},{q})' if d > 0 else f'ARMA({p},{q})'})
    return statistics

def evaluate_all_models(tickers = [], length = 15):
    '''
    Evaluates ml, ar and nn
    '''
    for i in tickers:

        flat_x_train, flat_x_test, flat_y_train, flat_y_test, flat_scaler = get_data(ticker = i, length = length, flatten = True)
        statistics = evaluate_ml_models(flat_x_train, flat_x_test, flat_y_train, flat_y_test, flat_scaler)
        statistics +=  evaluate_autoregressive_models(flat_x_train, flat_x_test, flat_y_train, flat_y_test, flat_scaler)

        x_train, x_test, y_train, y_test, scaler = get_data(ticker = i, length = length)
        statistics += evaluate_neural_networks(get_topologies(length), x_train, x_test, y_train, y_test, scaler)

        current_ticker_statistics = pd.DataFrame(statistics)
        current_ticker_statistics['ticker'] = i
        total_statistics = pd.concat([total_statistics, current_ticker_statistics])
    return pd.DataFrame(statistics).sort_values(by='mape')
import pandas as pd, time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,  mean_absolute_percentage_error
import plotly.express as px
from .model import Model

import warnings
warnings.filterwarnings("ignore")


def evaluate_ml_models(x_train, x_test, y_train, y_test, scaler, out_type = 'list'):
    '''
    Evaluates machine learning models on data.

    Args:
        x_train (np.array): Input values to fit model.
        x_test (np.array): Input values to evaluate model.
        y_train (np.array): Target values to fit model.
        y_test (np.array): Target values to fit model.
        scaler (Scaler): Scaler to scale/unscale values during evaluation.
        out_type (str): Format to return data. 
    
    Returns:
        list: List of dictionaries with time, MSE, MAE, MAPE, R2 values and model name for each model.
        or
        pd.DataFrame : dataframe with the same data.

    Example:
    >>> evaluate_ml_models(x_train, x_test, y_train, y_test, scaler)
    [{'time' : 0.12, 'mse' : 0.04, 'mae' : 0.2, 'mape' : 0.01, 'model' : 'LR'}, ...]
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
                    'model': i})
        
    if out_type == 'df':
        return pd.DataFrame(statistics).sort_values(by='mape')
    elif out_type == 'list':
        return statistics

def evaluate_ar_models(x_train, x_test, y_train, y_test, scaler, out_type = 'list'):
    '''
    Evaluates autoregressive models on data.

    Args:
        x_train (np.array): Input values to fit model.
        x_test (np.array): Input values to evaluate model.
        y_train (np.array): Target values to fit model.
        y_test (np.array): Target values to fit model.
        scaler (Scaler): Scaler to scale/unscale values during evaluation.
        out_type (str): Format to return data. 
    
    Returns:
        list: List of dictionaries with time, MSE, MAE, MAPE, R2 values and model name for each model.
        or
        pd.DataFrame : dataframe with the same data.

    Example:
    >>> evaluate_ar_models(x_train, x_test, y_train, y_test, scaler)
    [{'time' : 0.12, 'mse' : 0.04, 'mae' : 0.2, 'mape' : 0.01, 'model' : 'ARMA(2,1)'}, ...]
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
        
    if out_type == 'df':
        return pd.DataFrame(statistics).sort_values(by='mape')
    elif out_type == 'list':
        return statistics  

def evaluate_nn_models(x_train, x_test, y_train, y_test, scaler, optimizer = 'nadam', loss = 'mse', epochs = 10, batch_size = 256, out_type = 'list'):
    '''
    Evaluates neural network based models on data.

    Args:
        x_train (np.array): Input values to fit model.
        x_test (np.array): Input values to evaluate model.
        y_train (np.array): Target values to fit model.
        y_test (np.array): Target values to fit model.
        scaler (Scaler): Scaler to scale/unscale values during evaluation.
        optimizer (str): tf.Keras optimizer name (optional: only for neural networks).
        loss (str): tf.Keras loss name (optional: only for neural networks).
        epochs (int): Number of epochs during fitting (optional: only for neural networks).
        batch_size (int): Size of batch during fitting (optional: only for neural networks).
        out_type (str): Format to return data. 
    
    Returns:
        list: List of dictionaries with time, MSE, MAE, MAPE, R2 values and model name for each model.
        or
        pd.DataFrame : dataframe with the same data.

    Example:
    >>> evaluate_nn_models(x_train, x_test, y_train, y_test, scaler)
    [{'time' : 0.12, 'mse' : 0.04, 'mae' : 0.2, 'mape' : 0.01, 'model' : 'LSTM x1'}, ...]
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
    
    if out_type == 'df':
        return pd.DataFrame(statistics).sort_values(by='mape')
    elif out_type == 'list':
        return statistics


def evaluate_all_models(x_train, x_test, y_train, y_test, scaler, optimizer = 'nadam', loss = 'mse', epochs = 10, batch_size = 256, out_type = 'df'):
    '''
    Evaluates all models on data.

    Args:
        x_train (np.array): Input values to fit model.
        x_test (np.array): Input values to evaluate model.
        y_train (np.array): Target values to fit model.
        y_test (np.array): Target values to fit model.
        scaler (Scaler): Scaler to scale/unscale values during evaluation.
        optimizer (str): tf.Keras optimizer name (optional: only for neural networks).
        loss (str): tf.Keras loss name (optional: only for neural networks).
        epochs (int): Number of epochs during fitting (optional: only for neural networks).
        batch_size (int): Size of batch during fitting (optional: only for neural networks).
        out_type (str): Format to return data. 
    
    Returns:
        list: List of dictionaries with time, MSE, MAE, MAPE, R2 values and model name for each model.
        or
        pd.DataFrame : dataframe with the same data.

    Example:
    >>> evaluate_all_models(x_train, x_test, y_train, y_test, scaler)
    [{'time' : 0.12, 'mse' : 0.04, 'mae' : 0.2, 'mape' : 0.01, 'model' : 'LR'}, ...]
    '''

    statistics = evaluate_ml_models(x_train, x_test, y_train, y_test, scaler)
    statistics += evaluate_ar_models(x_train, x_test, y_train, y_test, scaler)
    statistics += evaluate_nn_models(x_train, x_test, y_train, y_test, scaler, optimizer, loss, epochs, batch_size)

    if out_type == 'df':
        return pd.DataFrame(statistics).sort_values(by='mape')
    elif out_type == 'list':
        return statistics
    
def plot_histogran(results):
    '''
    Plots histogram for resulting error values.
    '''
    px.histogram(results, x='model', y='mape')
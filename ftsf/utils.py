'''This submodule contains utils to get nn topologies and models list, parse model names.'''

from tensorflow.keras.layers import (GRU, LSTM, Conv1D, Dense, Flatten,
                                     SimpleRNN)


def get_topologies(lag = 30):
    '''
    Generating neural network topologies adapted to defined lag.

    Args:

        lag: Number of previous values will be used to forecast on.

    Returns:

        Dict with neural network topologies and its names.

    Example:
    >>> get_topologies(15):
    {'CNN + LSTM' : [..], ..}
    '''

    neurons_per_layer = lag-1
    input_shape = (neurons_per_layer, 1)

    return {'CNN + LSTM': [Conv1D(filters = 128, kernel_size = 3, activation = 'relu', input_shape = input_shape),
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
                    Dense(units = 1)]}

def get_models(type):
    '''
    Getting a list of the available models.

    Args:

        type: Type of models to include to list.

    Returns:

        List of models of defined type or dict with all models divided by its types.

    Example:
    >>> get_models('ml')
    ['LR', 'DTR', 'RFR', 'GBR', 'SVR', 'CBR', 'XGBR', 'XGBRFR']
    '''

    models = {'ml': ['LR', 'DTR', 'RFR', 'GBR', 'SVR', 'CBR', 'XGBR', 'XGBRFR'],
            'ar': ['ARMA(2,1)', 'ARIMA(2,1,1)'],
            'nn': ['CNN + LSTM', 'LSTM x3', 'LSTM x2', 'LSTM x1', 'CNN + GRU', 'GRU x3', 'GRU x2', 'GRU x1', \
                'CNN + SimpleRNN', 'SimpleRNN x3', 'SimpleRNN x2', 'SimpleRNN x1', 'CNN', 'MLP(3)', 'MLP(2)', 'MLP(1)']}

    return models[type.lower()] if type != 'all' else models

def parse_name(name):
    '''
    Parsing p, d and q parameters from autoregressive model name.

    Args:

        name: Name of autoregressive model.

    Returns:

        List of parameters p, d and q.

    Example:
    >>> parse_name('ARIMA(2,1,1)')
    (2, 1, 1)
    '''
    if name.find('I') != -1:
        coef_p = float(name[name.find('(') + 1 : name.find(',')])
        coef_d = float(name[name.find(',') + 1 : name.find(',', name.find(',') + 1)])
        coef_q = float(name[name.find(',', name.find(',') + 1) + 1 : name.find(')')])
    else:
        coef_p = float(name[name.find('(') + 1 : name.find(',')])
        coef_d = 0
        coef_q = float(name[name.find(',') + 1 : name.find(')')])
    return (coef_p, coef_d, coef_q)

from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import LSTM, Dense, GRU, SimpleRNN, Conv1D, Flatten

def get_topologies(lag = 30):
    '''
    Returns dict of neural network topologies adapted to defined lag. 

    Args:
        lag (int): 

    
    '''
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
                            Dense(units = 1)]
                    }
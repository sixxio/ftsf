# ftsf
This package provides high-level interface to work with financial time series forecasting models.
### Installation
To install this package run command:
> pip install ftsf
### Usage
The usage of package is simple:
> from ftsf.model import Model  
> from ftsf.preprocessing import get_data  
> x_train, x_test, y_train, y_test, scaler = get_data('AAPL', length=15)  
> model = Model('LSTM x2', lag = 15)  
> model.fit(x_train, y_train)
> model.evaluate(x_test, y_test, scaler)  
{'mse' : 0.679, 'mae' : 0.8291, 'mape' : 0.00783, 'r2' : 0.98778}
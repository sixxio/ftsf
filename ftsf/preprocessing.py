import pandas as pd, re, json, requests as rq, numpy as np, datetime

from .scaler import Scaler

from pathlib import Path
PACKAGEDIR = Path(__file__).parent.absolute()


def get_data(ticker = 'AAPL', column = 'close', length = 15, step = 1, start_date = '01-01-2013', split_date = '01-01-2022', end_date = '01-01-2023', flatten = False, validate = False):
    '''
    Prepares data for training and evaluating.

    Args:
    ticker (str): Ticker to get data for.
    column (str): Column from HLOCV to work with.
    length (int): Length of arrays, length-1 values are for fitting, 1 value is for evaluating.
    step (int): Step to cut data using sliding window.
    start_date (str): Date to start cutting data from.
    split_date (str): Date to split into train/test.
    end_date (str): Date to end cutting data.
    flatten (bool): Shape format:  (cuts, length-1) or (cuts, length-1, 1).
    validate (bool): To make validation split or not.
    
    Returns:
        x_train, x_test, y_train, y_test, scaler: train and test data with scaler to work with.

    Example:
    >>> get_data('AAPL', 'close', 10)
    [[0.122, 0.12, 0.125, ..], ..]
    [0.12, ..]
    [[0.132, 0.13, 0.135, ..], ..]
    [0.13, ..]
    Scaler()
    '''
    data = pd.read_parquet(f'{PACKAGEDIR}/data.parquet.gz')
    data = data[(data.dateTime >= start_date) & (data.ticker == ticker)]

    train_data = data[(data.dateTime < split_date)][column].values.reshape(-1, 1)
    test_data =  data[(data.dateTime >= split_date) & (data.dateTime < end_date)][column].values.reshape(-1, 1)

    train_set = [train_data[j:j+length,0] for j in range(0, len(train_data)-length, step)]
    test_set =  [test_data[j:j+length,0] for j in range(0, len(test_data)-length, step)]

    scaler = Scaler().fit(train_set)
    scaled_train_set = scaler.scale(train_set)
    scaled_test_set = scaler.scale(test_set)

    scaled_x_train, scaled_x_test = scaled_train_set[:,:length-1], scaled_test_set[:,:length-1]
    scaled_y_train, scaled_y_test = scaled_train_set[:,length-1],  scaled_test_set[:,length-1]

    if not flatten:
        scaled_x_train = np.reshape(scaled_x_train, (scaled_x_train.shape[0], length - 1,1))
        scaled_x_test = np.reshape(scaled_x_test, (scaled_x_test.shape[0], length - 1,1))

    if validate:
        val_data = data[(data.dateTime >= end_date) & (data.ticker == ticker)]['price'].values.reshape(-1, 1)
        val_set = [val_data[j:j+length,0] for j in range(0, len(val_data)-length, step)]
        scaled_val_set = scaler.scale(val_set)
        scaled_x_val, scaled_y_val = scaled_val_set[:,:length-1], scaled_val_set[:,length-1]
        return scaled_x_train, scaled_x_test, scaled_x_val, scaled_y_train, scaled_y_test, scaled_y_val, scaler
    else:
        return scaled_x_train, scaled_x_test, scaled_y_train, scaled_y_test, scaler

def update_data(tickers = []):
    '''
    Updates stored data for defined tickers.

    Args:
    tickers (list): Tickers list to update data for.

    Example:
    >>> get_data(['AMZN', 'TSLA', 'NVDA']])
    Info about 3 tickers successfully uploaded.
    '''
    all_tickers_df = pd.read_parquet(f'{PACKAGEDIR}/data.parquet.gz')
    for i in tickers:
        headers = {"Accept":"text/html", "Accept-Language":"en-US", "Referer":"https://www.nasdaq.com/", "User-Agent":"Chrome/64.0.3282.119"} 
        resp = rq.get(f'https://api.nasdaq.com/api/quote/{i}/chart?assetclass=stocks&fromdate=2023-04-29&todate={datetime.datetime.now().strftime("%Y-%m-%d")}', headers=headers, verify=True)
        if resp.status_code == 200:
            try:
                smth = json.loads(re.search('\[.*\]', resp.text).group())
                cur_tick_data = pd.DataFrame([smth[k]['z'] for k in range(len(smth))])
                cur_tick_data['ticker'] = i
                for col_name in ['high','low','open','close','volume','value']:
                    cur_tick_data[col_name] = pd.to_numeric(cur_tick_data[col_name].str.replace(',',''))
                cur_tick_data['dateTime'] = pd.to_datetime(cur_tick_data['dateTime'])
                all_tickers_df = pd.concat([all_tickers_df, cur_tick_data])
            except KeyError:
                pass
        else:
            print(f'ERROR: smth is wrong with {i}')
    all_tickers_df.to_parquet(f'{PACKAGEDIR}/data.parquet.gz', compression='gzip')
    print(f'Info about {len(tickers)} tickers successfully uploaded.')
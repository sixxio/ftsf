import sys

sys.path.insert(0, '../ftsf')

from ftsf import evaluation as ev
from ftsf import preprocessing as pr


def test_eval_ml_len():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=5)
    assert len(ev.evaluate_ml_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler)) == 8

def test_eval_ar_len():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=5)
    assert len(ev.evaluate_ar_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler)) == 2

def test_eval_nn_len():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=7)
    assert len(ev.evaluate_nn_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, epochs = 2, batch_size = 512)) == 16

def test_eval_all_len():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=7)
    assert len(ev.evaluate_all_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, epochs = 2, batch_size = 512, out_type='list')) == 26


def test_eval_ml_keys():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=5)
    assert list(ev.evaluate_ml_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler)[0].keys()) == ['time', 'mse', 'mae', 'mape', 'r2', 'model']

def test_eval_ar_keys():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=5)
    assert list(ev.evaluate_ar_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler)[0].keys()) == ['time', 'mse', 'mae', 'mape', 'r2', 'model']

def test_eval_nn_keys():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=7)
    assert list(ev.evaluate_nn_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, epochs = 2, batch_size = 512)[0].keys()) == ['time', 'mse', 'mae', 'mape', 'r2', 'model']

def test_eval_all_keys():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=7)
    assert list(ev.evaluate_all_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, epochs = 2, batch_size = 512, out_type='list')[0].keys()) == ['time', 'mse', 'mae', 'mape', 'r2', 'model']


def test_eval_ml_df():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=5)
    assert ev.evaluate_ml_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, out_type='df').shape == (8,6)

def test_eval_ar_df():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=5)
    assert ev.evaluate_ar_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, out_type='df').shape == (2,6)

def test_eval_nn_df():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=7)
    assert ev.evaluate_nn_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, epochs = 2, batch_size = 512, out_type='df').shape == (16,6)

def test_eval_all_df():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=7)
    assert ev.evaluate_all_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, epochs = 2, batch_size = 512, out_type='df').shape == (26,6)


def test_eval_ml_df_cols():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=5)
    assert ev.evaluate_ml_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, out_type='df').columns.to_list() == ['time', 'mse', 'mae', 'mape', 'r2', 'model']

def test_eval_ar_df_cols():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=5)
    assert ev.evaluate_ar_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, out_type='df').columns.to_list() == ['time', 'mse', 'mae', 'mape', 'r2', 'model']

def test_eval_nn_df_cols():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=7)
    assert ev.evaluate_nn_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, epochs = 2, batch_size = 512, out_type='df').columns.to_list() == ['time', 'mse', 'mae', 'mape', 'r2', 'model']

def test_eval_all_df_cols():
    test_xtr, test_xt, test_ytr, test_yt, test_scaler = pr.get_data(start_date='11-11-2021', end_date='03-03-2022', length=7)
    assert ev.evaluate_all_models(test_xtr, test_xt, test_ytr, test_yt, test_scaler, epochs = 2, batch_size = 512, out_type='df').columns.to_list() == ['time', 'mse', 'mae', 'mape', 'r2', 'model']
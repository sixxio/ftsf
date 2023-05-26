import sys
sys.path.insert(0, '../ftsf')

import ftsf, pytest

def test_scaling_on_1d_array():
    assert ([[0,0.5,1,1.5]] == ftsf.CustomScaler().fit([[0,1,2,3]]).scale([[0,1,2,3]])).all()
def test_check_work1():
    assert 1 == 1
def test_check_work2():
    assert 1 == 1
def test_check_work3():
    assert 1 == 1


# eval 
# ключи и их порядок
# количество моделей в ответе
# ошибки больше нуля, р2 меньше единицы
# итого 9

# scaler
# нормализация
# параметры

# basktest
# покупка продажа

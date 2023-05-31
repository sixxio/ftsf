import sys

sys.path.insert(0, '../ftsf')

from ftsf.scaler import Scaler


def test_scaling():
    assert ([[0,0.5,1,1.5]] == Scaler().fit([[0,1,2]]).scale([[0,1,2,3]])).all()

def test_shape():
    assert Scaler().fit([[0,1,2]]).scale([[0,1,2,3]]).shape == (1,4)

def test_setting_params():
    assert Scaler([0,1]).params() == (0,1)

def test_auto_setting_params():
    assert Scaler().fit([[0,1,2,3]]).params() == (0,3)
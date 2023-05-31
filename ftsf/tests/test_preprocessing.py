import sys

sys.path.insert(0, '../ftsf')

from ftsf import preprocessing as pr


def test_auto_length():
    assert pr.get_data()[0].shape[1] == 14

def test_length():
    assert pr.get_data(length=5)[0].shape[1] == 4

def test_shape():
    assert pr.get_data(length=5)[0].shape[2] == 1

def test_shape_misc():
    assert len(pr.get_data(length=5)[0].shape) == 3

def test_shape_flat():
    assert len(pr.get_data(length=5, flatten = True)[0].shape) == 2

def test_output_form():
    assert len(pr.get_data(length=5)) == 5

def test_output_form_validate():
    assert len(pr.get_data(length=5, validate = True)) == 7
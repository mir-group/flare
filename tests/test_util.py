from flare.util import element_to_Z, Z_to_element
import pytest
import numpy as np

from pytest import raises


def test_element_to_Z():
    for i in range(120):
        assert element_to_Z(i) == i

    assert element_to_Z('1') == 1
    assert element_to_Z(np.int(1.0)) == 1

    for pair in zip(['H', 'C', 'O', 'Og'], [1, 6, 8, 118]):
        assert element_to_Z(pair[0]) == pair[1]


def test_elt_warning():
    with pytest.warns(Warning):
        element_to_Z('Fe2')


def test_Z_to_element():
    for i in range(1,118):
        assert isinstance(Z_to_element(i),str)

    for pair in zip([1, 6, '8', '118'], ['H', 'C', 'O', 'Og']):
        assert Z_to_element(pair[0]) == pair[1]

    with raises(ValueError):
        Z_to_element('a')
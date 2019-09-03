from flare.util import element_to_Z
import pytest

def test_element_to_Z():
    for i in range(120):
        assert element_to_Z(i) == i

    for pair in zip(['H', 'C', 'O', 'Og'], [1, 6, 8, 118]):

        assert element_to_Z(pair[0]) == pair[1]

def test_elt_warning():
    with pytest.warns(Warning):
        element_to_Z('Fe2')

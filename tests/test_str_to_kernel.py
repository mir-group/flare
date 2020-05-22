import sys
from copy import deepcopy
import pytest
import numpy as np
from numpy.random import random, randint

from flare import env, struc, gp
from flare.kernels.utils import str_to_kernel_set as stks


@pytest.mark.parametrize('kernel_array', [['twobody'], ['threebody'], ['twobody', 'threebody'],
                                          ['twobody', 'threebody', 'manybody']])
@pytest.mark.parametrize('component', ['sc', 'mc'])
@pytest.mark.parametrize('nspecie', [1, 2])
def test_stk(kernel_array, component, nspecie):
    """Check whether the str_to_kernel_set can return kernel functions
    properly"""

    try:
        k, kg, ek, efk = stks(kernel_array, component, nspecie)
    except:
        raise RuntimeError(f"fail to return kernel {kernel_array} {component} {nspecie}")

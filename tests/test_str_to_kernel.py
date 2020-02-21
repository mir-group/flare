import sys
from copy import deepcopy
import pytest
import numpy as np
from numpy.random import random, randint

from flare import env, struc, gp
from flare.kernels.utils import str_to_kernel_set as stks

@pytest.mark.parametrize('kernel_name', ['2mc', '3mc', '2+3mc', '2', '3',
                                         '2+3'])
def test_stk(kernel_name):
    """Check whether the str_to_kernel_set can return kernel functions
    properly"""

    try:
        k, kg, ek, efk = stks(kernel_name)
    except:
        raise RuntimeError(f"fail to return kernel {kernel_name}")

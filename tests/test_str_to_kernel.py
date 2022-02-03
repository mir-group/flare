import sys
from copy import deepcopy
import pytest
import numpy as np
from numpy.random import random, randint

from flare.kernels.utils import str_to_kernel_set as stks


@pytest.mark.parametrize(
    "kernels",
    [
        ["twobody"],
        ["threebody"],
        ["twobody", "threebody"],
        ["twobody", "threebody", "manybody"],
    ],
)
@pytest.mark.parametrize("component", ["sc", "mc"])
@pytest.mark.parametrize("nspecie", [1, 2])
def test_stk(kernels, component, nspecie):
    """Check whether the str_to_kernel_set can return kernel functions
    properly"""

    try:
        k, kg, ek, efk, _, _, _ = stks(kernels, component, {"nspecie": nspecie})
    except:
        raise RuntimeError(f"fail to return kernel {kernels} {component} {nspecie}")

import numpy as np
from numba import njit
from flare.env import AtomicEnvironment
from typing import Callable
import flare.cutoffs as cf
from math import exp


class ManyBodyKernel:
    def __init__(self, hyperparameters: 'ndarray', cutoff: float,
                 cutoff_func: Callable = cf.quadratic_cutoff):
        self.hyperparameters = hyperparameters
        self.signal_variance = hyperparameters[0]
        self.length_scale = hyperparameters[1]
        self.cutoff = cutoff
        self.cutoff_func = cutoff_func

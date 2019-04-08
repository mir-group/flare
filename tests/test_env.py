import pytest
import numpy as np
import sys
import time
from random import random
sys.path.append('../otf_engine')
import env
import struc


def test_species_count():
    cell = np.eye(3)
    species = ['A', 'B', 'C']
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    struc_test = struc.Structure(cell, species, positions)
    env_test = env.AtomicEnvironment(struc_test, 0, np.array([1, 1]))
    assert(len(env_test.bond_array_2) == len(env_test.etypes))
    assert(isinstance(env_test.etypes[0], np.int8))

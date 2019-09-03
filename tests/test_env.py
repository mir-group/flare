import pytest
import numpy as np
import sys
import time
from random import random
from flare import env, struc


def test_species_count():
    cell = np.eye(3)
    species = [1, 2, 3]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    struc_test = struc.Structure(cell, species, positions)
    env_test = env.AtomicEnvironment(struc_test, 0, np.array([1, 1]))
    assert(len(struc_test.positions) == len(struc_test.coded_species))
    assert(len(env_test.bond_array_2) == len(env_test.etypes))
    assert(isinstance(env_test.etypes[0], np.int8))

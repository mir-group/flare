import pytest
import numpy as np
import sys
import time
from random import random
sys.path.append('../otf_engine')
import fast_env
import env
import struc


def test_env_agreement():
    # create test structure
    positions = [np.array([random(), random(), random()]),
                 np.array([random(), random(), random()]),
                 np.array([random(), random(), random()]),
                 np.array([random(), random(), random()]),
                 np.array([random(), random(), random()])]
    species = ['B', 'A', 'B', 'A', 'B']
    cell = np.eye(3)

    cutoff_2 = 1
    cutoff_3 = 1
    cutoffs = np.array([cutoff_2, cutoff_3])

    test_structure = struc.Structure(cell, species, positions, cutoff_2)

    # create environment
    atom = 0
    toy_env = env.ChemicalEnvironment(test_structure, atom)
    toy_env_2 = fast_env.AtomicEnvironment(test_structure, atom, cutoffs)

    assert(np.array_equal(toy_env.bond_array, toy_env_2.bond_array_2))

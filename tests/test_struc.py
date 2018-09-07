#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" Test the Structure class from struc.py

Steven Torrisi
"""

import pytest


import numpy as np
import sys
from test_GaussianProcess import get_random_structure

sys.path.append('../src')
from struc import Structure

# ------------------------------------------------------
#                   test  Structure functions
# ------------------------------------------------------


def test_random_structure_setup():
    struct, forces = get_random_structure(cell=np.eye(3),
                                          unique_species=["A", "B", ],
                                          cutoff=np.random.uniform(1, 10.),
                                          noa=2)

    assert np.equal(struct.lattice, np.eye(3)).all()
    assert 'A' in struct.unique_species or 'B' in struct.unique_species
    assert len(struct.positions) == 2


def test_2_body_bond_order():
    """
    Written by Simon B
    :return:
    """
    lattice = np.eye(3)
    species = ['B', 'A']
    positions = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    test_structure = Structure(lattice, species, positions, cutoff)

    # test species_to_bond
    assert (test_structure.bond_list == [['B', 'B'], ['B', 'A'], ['A', 'A']])

# TODO IO-based unit tests for pasrsing the output files of runs (even though
# some may be random

#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" Test the Structure class from struc.py

Steven Torrisi
"""

import pytest


import numpy as np
import sys
from test_GaussianProcess import get_random_structure

sys.path.append('../otf_engine')
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


"""
def test_index_finding():
    noa=30
    struc,_ = get_random_structure(cell=2*np.eye(3),unique_species=['A'],
                                 cutoff=1,noa=noa)

    target_atom = np.random.randint(0,noa)

    target_pos = struc.positions[target_atom]

    assert struc.get_index_from_position(target_pos) == target_atom

    for _ in range(10000):
        shift = np.random.randint(-10,10,3)
        target_pos_shifted = target_pos + shift[0]*struc.vec1 + shift[1]*struc.vec2 + \
                   shift[2]*struc.vec3

        assert struc.get_index_from_position(target_pos_shifted) == target_atom


"""


# TODO IO-based unit tests for pasrsing the output files of runs (even though
# some may be random

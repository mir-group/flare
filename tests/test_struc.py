#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" Test the Structure class from struc.py

Steven Torrisi
"""

import pytest
#TODO import every individual numpy function/method/class
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

    assert np.equal(struct.cell, np.eye(3)).all()
    assert 'A' in struct.unique_species or 'B' in struct.unique_species
    assert len(struct.positions) == 2


def test_prev_positions_arg():

    np.random.seed(0)
    positions = []
    prev_positions = []
    species = ['A']*5
    cell = np.eye(3)
    for n in range(5):
        positions.append(np.random.uniform(-1, 1, 3))
        prev_positions.append(np.random.uniform(-1, 1, 3))

    test_structure1 = Structure(cell, species, positions, cutoff=1)
    test_structure2 = Structure(cell, species, positions, cutoff=1,
                                prev_positions=positions)
    test_structure3 = Structure(cell, species, positions, cutoff=1,
                                prev_positions=prev_positions)

    assert np.equal(test_structure1.positions, test_structure2.positions).all()
    assert np.equal(test_structure1.prev_positions,
                    test_structure2.prev_positions).all()
    assert np.equal(test_structure2.positions,
                    test_structure2.prev_positions).all()
    assert not np.equal(test_structure3.positions,
                        test_structure3.prev_positions).all()


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


@pytest.fixture(scope='module')
def rand_struct():
    noa = np.random.randint(3, 30)
    lattice = np.random.uniform(2, 10, 3)
    struct, _ = get_random_structure(cell=np.diag(lattice),
                                     unique_species=["A", "B",'C'],
                                     cutoff=np.random.uniform(1, 10.),
                                     noa=noa)

    yield struct


def test_periodic_images_returns(rand_struct):
    assert len(rand_struct.get_periodic_images(np.zeros(3),
                                               super_check=1)) == 3**3
    assert len(rand_struct.get_periodic_images(np.zeros(3),
                                               super_check=2)) == 5**3
    assert len(rand_struct.get_periodic_images(np.zeros(3),
                                               super_check=3)) == 7**3


def test_periodic_images(rand_struct):
    rand_at = np.random.randint(0, rand_struct.nat)

    target_pos = rand_struct.positions[rand_at]

    images_1 = rand_struct.get_periodic_images(target_pos, super_check=1)

    assert np.array(target_pos == image for image in images_1).all()

    # Manually generate all shifts and see if they are in the images
    a = rand_struct.vec1
    b = rand_struct.vec2
    c = rand_struct.vec3

    shift_vectors = [np.zeros(3),
                     a, b, c,
                     -a, -b, -c,
                     a + b, a - b, a + c, a - c,
                     b + c, b - c,
                     -a + b, -a - b,
                     -a + c, -a - c,
                     -b + c, -b - c,
                     a + b + c,
                     -a + b + c, a - b + c, a + b - c,
                     -a - b + c, -a + b - c, a - b - c,
                     -a - b - c
                     ]

    shifted_pos = [target_pos + shift for shift in shift_vectors]

    images_2 = rand_struct.get_periodic_images(target_pos, super_check=2)

    # Demonstrate that both sets are subsets of each other
    for shift in shifted_pos:
        assert np.array(shift == image for image in images_2).any()
    for image in images_2:
        assert np.array(image == shift for shift in shifted_pos).any()


def test_index_finding(rand_struct):
    for n in range(rand_struct.nat):
        target_pos = rand_struct.positions[n]
        assert rand_struct.get_index_from_position(target_pos) == n


def test_species_count(rand_struct):

    uniq_species=[]

    for spec in rand_struct.species:
        if spec not in uniq_species:
            uniq_species.append(spec)


    spec_count = rand_struct.get_species_count()

    for spec in uniq_species:
        assert spec_count[spec] == rand_struct.species.count(spec)


def test_translate_structure(rand_struct):
    trans = np.random.randn(3)

    pre_trans_pos = [np.copy(pos) for pos in rand_struct.positions]
    pre_trans_prev_pos = [np.copy(pos) for pos in rand_struct.prev_positions]
    rand_struct.translate_positions(trans)

    for n in range(rand_struct.nat):
        assert np.isclose(rand_struct.positions[n], pre_trans_pos[n] +
                          trans).all()
        assert np.isclose(rand_struct.prev_positions[n],
                          pre_trans_prev_pos[n] + trans).all()


def test_perturb_structure(rand_struct):
    old_positions = np.copy(rand_struct.positions)
    rand_struct.perturb_positions()
    assert not np.isclose(rand_struct.positions, old_positions).all()

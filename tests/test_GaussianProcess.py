#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" Gaussian Process Regression test suite based on py.test

Simon Batzner
"""

import pytest

import numpy as np

from gp import GaussianProcess
from env import Structure, ChemicalEnvironment


def get_random_structure(cell, unique_species, cutoff, noa):
    """Create a random test structure """
    positions = []
    forces = []
    species = []
    for n in range(noa):
        positions.append(np.random.uniform(-1, 1, 3))
        forces.append(np.random.uniform(-1, 1, 3))
        species.append(unique_species[np.random.randint(0, 2)])

    test_structure = Structure(cell, species, positions, cutoff)

    return test_structure, forces

# ------------------------------------------------------
#          fixtures
# ------------------------------------------------------


# set the scope to module so it will only be setup once
@pytest.fixture(scope='module')
def two_body_gp(params):
    """Returns a GP instance with an rbf kernel"""

    print("\nSetting up...\n")

    # params
    cell = np.eye(3)
    unique_species = ['B', 'A']
    cutoff = 0.8
    noa = 10

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  cutoff, noa)

    # create test point
    test_structure_2, _ = get_random_structure(cell, unique_species,
                                               cutoff, noa)

    # test update_db
    gaussian = GaussianProcess(kernel='two_body')
    gaussian.update_db(test_structure, forces)

    # return gaussian
    yield gaussian

    # code after yield will be executed once all tests are run
    # this will not be run if an exception is raised in the setup
    print("\nTearing down\n")
    del gaussian


@pytest.fixture(scope='module')
def params():
    parameters = {'unique_species': ['B', 'A'], 'cutoff': 0.8, 'noa': 10,
              'db_pts': 30}
    yield parameters
    del parameters


@pytest.fixture(scope='module')
def test_point():
    """Create test point for kernel to compare against"""
    # params
    cell = np.eye(3)
    unique_species = ['B', 'A']
    cutoff = 0.8
    noa = 10

    test_structure_2, _ = get_random_structure(cell, unique_species,
                                               cutoff, noa)

    test_pt = ChemicalEnvironment(test_structure_2, 0)

    yield test_pt
    del test_pt


# ------------------------------------------------------
#                   test GP methods
# ------------------------------------------------------

def test_update_db(two_body_gp, params):
    assert(len(two_body_gp.training_data) == params['noa'])
    assert(len(two_body_gp.training_data) == len(two_body_gp.training_data))
    assert(len(two_body_gp.training_labels_np) ==
           len(two_body_gp.training_data * 3))



def test_set_kernel(two_body_gp, params):
    two_body_gp.set_kernel(sigma_f=1, length_scale=1, sigma_n=0.1)
    assert(two_body_gp.k_mat.shape == (params['db_pts'], params['db_pts']))


def test_get_kernel_vector(two_body_gp, test_point, params):
    assert(two_body_gp.get_kernel_vector(test_point, 1).shape ==
           (params['db_pts'],))


def test_set_alpha(two_body_gp, params):
    two_body_gp.set_alpha()
    assert(two_body_gp.alpha.shape == (params['db_pts'],))


# ------------------------------------------------------
#        example of how to do parametrized testing
# ------------------------------------------------------

# @pytest.mark.parametrize("sigma_f", "length_scale", "sigma_n",
#                          [(1, 2, 3), (4, 5, 6)])
# def test_dummy(sigma_f, length_scale, sigma_n):
#     pass






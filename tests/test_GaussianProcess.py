#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" Gaussian Process Regression test suite based on py.test

Simon Batzner
"""

import pytest

import numpy as np

from gp import GaussianProcess, minus_like_hyp
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
def two_body_gp():
    """Returns a GP instance with a two-body numba-based kernel"""
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
    print("\n\nTearing down\n")
    del gaussian


# set the scope to module so it will only be setup once
@pytest.fixture(scope='module')
def two_body_gp_py():
    """Returns a GP instance with an two-body python-based kernel"""

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
    gaussian = GaussianProcess(kernel='two_body_py')
    gaussian.update_db(test_structure, forces)

    # return gaussian
    yield gaussian

    # code after yield will be executed once all tests are run
    # this will not be run if an exception is raised in the setup
    print("\n\nTearing down\n")
    del gaussian


@pytest.fixture(scope='module')
def params():
    parameters = {'unique_species': ['B', 'A'],
                  'cutoff': 0.8,
                  'noa': 10,
                  'cell': np.eye(3),
                  'db_pts': 60}
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

    # params
    test_structure, forces = get_random_structure(params['cell'],
                                                  params['unique_species'],
                                                  params['cutoff'],
                                                  params['noa'])

    # add structure and forces to db
    two_body_gp.update_db(test_structure, forces)

    assert(len(two_body_gp.training_data) == params['noa'] * 2)
    assert(len(two_body_gp.training_labels_np) == params['noa'] * 2 * 3)


def test_update_db_py(two_body_gp_py, params):

    # params
    test_structure, forces = get_random_structure(params['cell'],
                                                  params['unique_species'],
                                                  params['cutoff'],
                                                  params['noa'])

    # add structure and forces to db
    two_body_gp_py.update_db(test_structure, forces)

    assert(len(two_body_gp_py.training_data) == params['noa'] * 2)
    assert(len(two_body_gp_py.training_labels_np) == params['noa'] * 2 * 3)


def test_minus_like_hyp(two_body_gp):
    neg_like = minus_like_hyp(hyp=[1, 1, 1], gp=two_body_gp)
    assert(isinstance(neg_like, float))


def test_minus_like_hyp_py(two_body_gp_py):
    neg_like = minus_like_hyp(hyp=[1, 1, 1], gp=two_body_gp_py)
    assert(isinstance(neg_like, float))


def test_like_hyp(two_body_gp):
    like = two_body_gp.like_hyp(hyp=[1, 1, 1])
    assert(isinstance(like, float))


def test_like_hyp_py(two_body_gp_py):
    like = two_body_gp_py.like_hyp(hyp=[1, 1, 1])
    assert(isinstance(like, float))


def test_set_kernel(two_body_gp, params):
    two_body_gp.set_kernel(sigma_f=1, length_scale=1, sigma_n=0.1)
    assert(two_body_gp.k_mat.shape == (params['db_pts'], params['db_pts']))


def test_set_kernel_py(two_body_gp_py, params):
    two_body_gp_py.set_kernel(sigma_f=1, length_scale=1, sigma_n=0.1)
    assert(two_body_gp_py.k_mat.shape == (params['db_pts'], params['db_pts']))


def test_set_alpha(two_body_gp, params):
    two_body_gp.set_alpha()
    assert (two_body_gp.alpha.shape == (params['db_pts'],))


def test_set_alpha_py(two_body_gp_py, params):
    two_body_gp_py.set_alpha()
    assert (two_body_gp_py.alpha.shape == (params['db_pts'],))


def test_get_likelihood(two_body_gp, params):
    like = two_body_gp.get_likelihood()
    assert (isinstance(like, float))


def test_get_likelihood_py(two_body_gp_py, params):
    like = two_body_gp_py.get_likelihood()
    assert (isinstance(like, float))


def test_get_kernel_vector(two_body_gp, test_point, params):
    assert(two_body_gp.get_kernel_vector(test_point, 1).shape ==
           (params['db_pts'],))


def test_get_kernel_vector_py(two_body_gp_py, test_point, params):
    assert(two_body_gp_py.get_kernel_vector(test_point, 1).shape ==
           (params['db_pts'],))


def test_train(two_body_gp, params):
    hyp = [two_body_gp.sigma_f,
           two_body_gp.length_scale,
           two_body_gp.sigma_n]

    # add struc and forces to db
    test_structure, forces = get_random_structure(params['cell'],
                                                  params['unique_species'],
                                                  params['cutoff'],
                                                  params['noa'])
    two_body_gp.update_db(test_structure, forces)

    # train gp
    two_body_gp.train()

    hyp_post = [two_body_gp.sigma_f,
                two_body_gp.length_scale,
                two_body_gp.sigma_n]

    # check if hyperparams have been updated
    assert(hyp != hyp_post)


def test_predict(two_body_gp, test_point):
    pred = two_body_gp.predict(x_t=test_point, d=1)
    assert(len(pred) == 2)
    assert(isinstance(pred[0], float))
    assert(isinstance(pred[1], float))


def test_predict_py(two_body_gp_py, test_point):
    pred = two_body_gp_py.predict(x_t=test_point, d=1)
    assert(len(pred) == 2)
    assert(isinstance(pred[0], float))
    assert(isinstance(pred[1], float))


# ------------------------------------------------------
#        example of how to do parametrized testing
# ------------------------------------------------------
#
# @pytest.mark.parametrize("sigma_f", "length_scale", "sigma_n",
#                          [(1, 2, 3), (4, 5, 6)])
# def test_dummy(sigma_f, length_scale, sigma_n):
#     pass

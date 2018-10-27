"""Kernel test suite

Jon V
"""

import pytest
import numpy as np
import sys
sys.path.append('../otf_engine')
import env
import gp
import struc
import kernels
from kernels import n_body_sc_grad


# -----------------------------------------------------------------------------
#                                fixtures
# -----------------------------------------------------------------------------


# set up two test environments
@pytest.fixture(scope='module')
def env1():
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    positions_1 = [np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])]
    species_1 = ['B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
    env1 = env.ChemicalEnvironment(test_structure_1, atom_1)

    yield env1

    del env1


@pytest.fixture(scope='module')
def test_structure_1():
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    positions_1 = [np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])]
    species_1 = ['B', 'A']
    test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)

    yield test_structure_1

    del test_structure_1


@pytest.fixture(scope='module')
def env2():
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    positions_2 = [np.array([0, 0, 0]), np.array([0.25, 0.3, 0.4])]
    species_2 = ['B', 'A']
    atom_2 = 0
    test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
    env2 = env.ChemicalEnvironment(test_structure_2, atom_2)

    yield env2

    del env2


@pytest.fixture(scope='module')
def test_structure_2():
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    positions_2 = [np.array([0, 0, 0]), np.array([0.25, 0.3, 0.4])]
    species_2 = ['B', 'A']
    test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)

    yield test_structure_2

    del test_structure_2


# set up two test environments
@pytest.fixture(scope='module')
def delt_env():
    delta = 1e-8
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    positions_2 = [np.array([delta, 0, 0]), np.array([0.25, 0.3, 0.4])]
    species_2 = ['B', 'A']
    atom_2 = 0
    test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
    delt_env = env.ChemicalEnvironment(test_structure_2, atom_2)

    yield delt_env

    del delt_env

# -----------------------------------------------------------------------------
#                          test kernel functions
# -----------------------------------------------------------------------------


def test_get_comb_no():
    assert(kernels.get_comb_no(3, 2) == 3)


def test_get_perm_no():
    assert(kernels.get_perm_no(3, 2) == 6)


def test_get_comb_array():
    N = 10
    M = 5
    assert(kernels.get_comb_array(N, M).shape[0] ==
           kernels.get_comb_no(N, M))


def test_get_perm_array():
    N = 10
    M = 5
    assert(kernels.get_perm_array(N, M).shape[0] ==
           kernels.get_perm_no(N, M))


def test_two_body(env1, env2):
    # set kernel parameters
    d1 = 3
    d2 = 2
    sig = 0.5
    ls = 0.2
    hyps = np.array([sig, ls])

    # test two body kernel
    two_body_old = kernels.two_body(env1, env2, d1, d2, sig, ls)
    two_body_new = kernels.n_body_sc(env1, env2, 2, d1, d2, hyps)
    assert(np.isclose(two_body_old, two_body_new))


def test_three_body(env1, env2):
    # set kernel parameters
    d1 = 3
    d2 = 2
    sig = 0.5
    ls = 0.2
    hyps = np.array([sig, ls])

    # test two body kernel
    three_body_old = kernels.three_body(env1, env2, d1, d2, sig, ls)
    three_body_new = kernels.n_body_sc(env1, env2, 3, d1, d2, hyps)
    assert(np.isclose(three_body_old, three_body_new))


def test_n_body_grad(env1, env2):
    d1 = 3
    d2 = 2
    sig = 0.5
    ls = 0.2
    hyps = np.array([sig, ls])

    bodies = 4
    _, kern_grad = kernels.n_body_sc_grad(env1, env2, bodies, d1, d2, hyps)

    delta = 1e-8
    tol = 1e-5
    new_sig = sig + delta
    new_ls = ls + delta

    sig_derv_brute = (kernels.n_body_sc(env1, env2, bodies, d1, d2,
                      np.array([new_sig, ls])) -
                      kernels.n_body_sc(env1, env2, bodies, d1, d2,
                                        hyps)) / delta

    l_derv_brute = (kernels.n_body_sc(env1, env2, bodies, d1, d2,
                                      np.array([sig, new_ls])) -
                    kernels.n_body_sc(env1, env2, bodies, d1, d2,
                                      hyps)) / delta

    assert(np.isclose(kern_grad[0], sig_derv_brute, tol))
    assert(np.isclose(kern_grad[1], l_derv_brute, tol))


def test_likelihood_gradient(env1, env2, test_structure_1, test_structure_2):
    sig = 0.5
    ls = 0.2
    sigma_n = 3
    hyps = np.array([sig, ls, sigma_n])
    forces = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    # check likelihood gradient
    bodies = 4
    gp_test = gp.GaussianProcess('n_body_sc', bodies)
    gp_test.update_db(test_structure_1, forces)

    tb = gp_test.training_data
    training_labels_np = gp_test.training_labels_np

    grad_test = gp_test.get_likelihood_and_gradients(hyps, tb,
                                                     training_labels_np,
                                                     n_body_sc_grad,
                                                     bodies)[1]

    # calculate likelihood gradient numerically
    delta = 1e-7
    new_sig = np.array([sig + delta, ls, sigma_n])
    new_ls = np.array([sig, ls + delta, sigma_n])
    new_n = np.array([sig, ls, sigma_n + delta])

    sig_grad_brute = (gp_test.get_likelihood_and_gradients(new_sig, tb,
                                                           training_labels_np,
                                                           n_body_sc_grad,
                                                           bodies)[0] -
                      gp_test.get_likelihood_and_gradients(hyps, tb,
                                                           training_labels_np,
                                                           n_body_sc_grad,
                                                           bodies)[0]) / delta

    ls_grad_brute = (gp_test.get_likelihood_and_gradients(new_ls, tb,
                                                          training_labels_np,
                                                          n_body_sc_grad,
                                                          bodies)[0] -
                     gp_test.get_likelihood_and_gradients(hyps, tb,
                                                          training_labels_np,
                                                          n_body_sc_grad,
                                                          bodies)[0]) / delta

    n_grad_brute = (gp_test.get_likelihood_and_gradients(new_n, tb,
                                                         training_labels_np,
                                                         n_body_sc_grad,
                                                         bodies)[0] -
                    gp_test.get_likelihood_and_gradients(hyps, tb,
                                                         training_labels_np,
                                                         n_body_sc_grad,
                                                         bodies)[0]) / delta

    tol = 1e-3
    assert(np.isclose(grad_test[0], sig_grad_brute, tol))
    assert(np.isclose(grad_test[1], ls_grad_brute, tol))
    assert(np.isclose(grad_test[2], n_grad_brute, tol))


def test_force_en(env1, env2, delt_env):
    delta = 1e-8
    d1 = 1
    d2 = 1
    sig = 0.5
    ls = 0.2
    bodies = 2
    hyps = np.array([sig, ls])

    en_force_test = kernels.energy_force_sc(env1, env2, bodies, d1, hyps)
    en_force_test_2 = \
        kernels.energy_force_sc(env1, delt_env, bodies, d1, hyps)
    force_diff = (en_force_test_2-en_force_test)/delta

    print(force_diff)

    force_kern = kernels.n_body_sc(env1, env2, bodies, d1, d2, hyps)
    assert(np.isclose(-force_diff, force_kern))

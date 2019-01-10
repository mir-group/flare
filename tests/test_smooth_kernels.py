import pytest
import numpy as np
import sys
sys.path.append('../otf_engine')
import env
import gp
import struc
import kernels
from kernels import n_body_sc_grad, n_body_mc_grad
import smooth_kernels
from smooth_kernels import two_body_smooth, two_body_smooth_grad

# -----------------------------------------------------------------------------
#                                fixtures
# -----------------------------------------------------------------------------


# set up two test environments
@pytest.fixture(scope='module')
def env1():
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    positions_1 = [np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])]
    species_1 = ['A', 'A']
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
    species_1 = ['A', 'A']
    test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)

    yield test_structure_1

    del test_structure_1


@pytest.fixture(scope='module')
def env2():
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    positions_2 = [np.array([0, 0, 0]), np.array([0.25, 0.3, 0.4])]
    species_2 = ['A', 'A']
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
    species_2 = ['A', 'A']
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
    species_2 = ['A', 'A']
    atom_2 = 0
    test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
    delt_env = env.ChemicalEnvironment(test_structure_2, atom_2)

    yield delt_env

    del delt_env


# -----------------------------------------------------------------------------
#                          test kernel functions
# -----------------------------------------------------------------------------


def test_two_body(env1, env2):
    d1 = 3
    d2 = 2
    sig = 0.5
    ls = 0.2
    d = 0.1
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001
    hyps = np.array([sig, ls, d])

    _, kern_grad = two_body_smooth_grad(env1, env2, d1, d2, hyps, cutoff)

    delta = 1e-8
    tol = 1e-5
    new_sig = sig + delta
    new_ls = ls + delta
    new_d = d + delta

    sig_derv_brute = (two_body_smooth(env1, env2, d1, d2,
                      np.array([new_sig, ls, d]), cutoff) -
                      two_body_smooth(env1, env2, d1, d2,
                                      hyps, cutoff)) / delta

    l_derv_brute = (two_body_smooth(env1, env2, d1, d2,
                                    np.array([sig, new_ls, d]), cutoff) -
                    two_body_smooth(env1, env2, d1, d2,
                                    hyps, cutoff)) / delta

    d_derv_brute = (two_body_smooth(env1, env2, d1, d2,
                                    np.array([sig, ls, new_d]), cutoff) -
                    two_body_smooth(env1, env2, d1, d2,
                                    hyps, cutoff)) / delta

    assert(np.isclose(kern_grad[0], sig_derv_brute, tol))
    assert(np.isclose(kern_grad[1], l_derv_brute, tol))
    assert(np.isclose(kern_grad[2], d_derv_brute, tol))

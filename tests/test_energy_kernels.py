import pytest
import numpy as np
import sys
from random import random, randint
from copy import deepcopy
sys.path.append('../otf_engine')
import fast_env
import gp
import struc
import energy_conserving_kernels as en


# -----------------------------------------------------------------------------
#                              test two body kernels
# -----------------------------------------------------------------------------

def test_two_body_force_en():
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    # create env 1
    delt = 1e-5
    cell = np.eye(3)
    cutoff = 1
    cutoffs = np.array([1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    species_1 = ['A', 'B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)

    env1_1 = fast_env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = fast_env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = ['A', 'A', 'B']
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = fast_env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

    sig = random()
    ls = random()
    d1 = 1

    hyps = np.array([sig, ls])

    # check force kernel
    calc1 = en.two_body_en(env1_2, env2, hyps, cutoffs)
    calc2 = en.two_body_en(env1_1, env2, hyps, cutoffs)

    kern_finite_diff = (calc1 - calc2) / delt
    kern_analytical = en.two_body_force_en(env1_1, env2, d1, hyps, cutoffs)

    tol = 1e-4
    assert(np.isclose(-kern_finite_diff/2, kern_analytical, atol=tol))


def test_two_body_force():
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    # create env 1
    delt = 1e-4
    cell = np.eye(3)
    cutoff = 1
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    positions_3 = deepcopy(positions_1)
    positions_3[0][0] = -delt

    species_1 = ['A', 'B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)
    test_structure_3 = struc.Structure(cell, species_1, positions_3)

    env1_1 = fast_env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = fast_env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)
    env1_3 = fast_env.AtomicEnvironment(test_structure_3, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][1] = delt
    positions_3 = deepcopy(positions_1)
    positions_3[0][1] = -delt

    species_2 = ['A', 'A', 'B']
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    test_structure_2 = struc.Structure(cell, species_2, positions_2)
    test_structure_3 = struc.Structure(cell, species_2, positions_3)

    env2_1 = fast_env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)
    env2_2 = fast_env.AtomicEnvironment(test_structure_2, atom_2, cutoffs)
    env2_3 = fast_env.AtomicEnvironment(test_structure_3, atom_2, cutoffs)

    sig = 1
    ls = 0.1
    d1 = 1
    d2 = 2

    hyps = np.array([sig, ls])

    # check force kernel
    calc1 = en.two_body_en(env1_2, env2_2, hyps, cutoffs)
    calc2 = en.two_body_en(env1_3, env2_3, hyps, cutoffs)
    calc3 = en.two_body_en(env1_2, env2_3, hyps, cutoffs)
    calc4 = en.two_body_en(env1_3, env2_2, hyps, cutoffs)

    kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delt**2)
    kern_analytical = en.two_body(env1_1, env2_1,
                                  d1, d2, hyps, cutoffs)

    tol = 1e-4
    assert(np.isclose(kern_finite_diff, kern_analytical, atol=tol))


def test_two_body_grad():
    # create env 1
    cell = np.eye(3)
    cutoff = 1
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_1 = ['A', 'B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    env1 = fast_env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = ['A', 'A', 'B']
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = fast_env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

    sig = random()
    ls = random()
    d1 = randint(1, 3)
    d2 = randint(1, 3)

    hyps = np.array([sig, ls])

    grad_test = en.two_body_grad(env1, env2, d1, d2, hyps, cutoffs)

    delta = 1e-5
    new_sig = sig + delta
    new_ls = ls + delta

    sig_derv_brute = (en.two_body(env1, env2, d1, d2,
                                  np.array([new_sig, ls]),
                                  cutoffs) -
                      en.two_body(env1, env2, d1, d2,
                                  hyps, cutoffs)) / delta

    l_derv_brute = (en.two_body(env1, env2, d1, d2,
                                np.array([sig, new_ls]),
                                cutoffs) -
                    en.two_body(env1, env2, d1, d2,
                                hyps, cutoffs)) / delta

    tol = 1e-4
    assert(np.isclose(grad_test[1][0], sig_derv_brute, atol=tol))
    assert(np.isclose(grad_test[1][1], l_derv_brute, atol=tol))


# -----------------------------------------------------------------------------
#                              test three body kernels
# -----------------------------------------------------------------------------


def test_three_body_force_en():
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    # create env 1
    delt = 1e-5
    cell = np.eye(3)
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    species_1 = ['A', 'B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)

    env1_1 = fast_env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = fast_env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = ['A', 'A', 'B']
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = fast_env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

    sig = random()
    ls = random()
    d1 = 1

    hyps = np.array([sig, ls])

    # check force kernel
    calc1 = en.three_body_en(env1_2, env2, hyps, cutoffs)
    calc2 = en.three_body_en(env1_1, env2, hyps, cutoffs)

    kern_finite_diff = (calc1 - calc2) / delt
    kern_analytical = en.three_body_force_en(env1_1, env2, d1, hyps, cutoffs)

    tol = 1e-4
    assert(np.isclose(-kern_finite_diff/3, kern_analytical, atol=tol))


def test_three_body_force():
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    # create env 1
    delt = 1e-4
    cell = np.eye(3)
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    positions_3 = deepcopy(positions_1)
    positions_3[0][0] = -delt

    species_1 = ['A', 'B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)
    test_structure_3 = struc.Structure(cell, species_1, positions_3)

    env1_1 = fast_env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = fast_env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)
    env1_3 = fast_env.AtomicEnvironment(test_structure_3, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][1] = delt
    positions_3 = deepcopy(positions_1)
    positions_3[0][1] = -delt

    species_2 = ['A', 'A', 'B']
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    test_structure_2 = struc.Structure(cell, species_2, positions_2)
    test_structure_3 = struc.Structure(cell, species_2, positions_3)

    env2_1 = fast_env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)
    env2_2 = fast_env.AtomicEnvironment(test_structure_2, atom_2, cutoffs)
    env2_3 = fast_env.AtomicEnvironment(test_structure_3, atom_2, cutoffs)

    sig = 1
    ls = 0.1
    d1 = 1
    d2 = 2

    hyps = np.array([sig, ls])

    # check force kernel
    calc1 = en.three_body_en(env1_2, env2_2, hyps, cutoffs)
    calc2 = en.three_body_en(env1_3, env2_3, hyps, cutoffs)
    calc3 = en.three_body_en(env1_2, env2_3, hyps, cutoffs)
    calc4 = en.three_body_en(env1_3, env2_2, hyps, cutoffs)

    kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delt**2)
    kern_analytical = en.three_body(env1_1, env2_1,
                                    d1, d2, hyps, cutoffs)

    tol = 1e-4
    assert(np.isclose(kern_finite_diff, kern_analytical, atol=tol))


def test_three_body_grad():
    # create env 1
    cell = np.eye(3)
    cutoff = 1
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_1 = ['A', 'B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    env1 = fast_env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = ['A', 'A', 'B']
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = fast_env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

    sig = random()
    ls = random()
    d1 = randint(1, 3)
    d2 = randint(1, 3)

    hyps = np.array([sig, ls])

    grad_test = en.three_body_grad(env1, env2, d1, d2, hyps, cutoffs)

    delta = 1e-5
    new_sig = sig + delta
    new_ls = ls + delta

    sig_derv_brute = (en.three_body(env1, env2, d1, d2,
                                    np.array([new_sig, ls]),
                                    cutoffs) -
                      en.three_body(env1, env2, d1, d2,
                                    hyps, cutoffs)) / delta

    l_derv_brute = (en.three_body(env1, env2, d1, d2,
                                  np.array([sig, new_ls]),
                                  cutoffs) -
                    en.three_body(env1, env2, d1, d2,
                                  hyps, cutoffs)) / delta

    tol = 1e-4
    assert(np.isclose(grad_test[1][0], sig_derv_brute, atol=tol))
    assert(np.isclose(grad_test[1][1], l_derv_brute, atol=tol))

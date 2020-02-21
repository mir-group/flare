import pytest
import numpy as np
import sys
from random import random, randint
from copy import deepcopy
from flare import env, gp, struc
import flare.kernels.kernels as en

from flare.kernels.utils import from_mask_to_args, from_grad_to_mask

# -----------------------------------------------------------------------------
#                        test two plus three body kernels
# -----------------------------------------------------------------------------

# TODO: fix this test to properly account for factors of 2 and 3
def test_two_plus_three_body_force_en():
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    # create env 1
    delt = 1e-8
    cell = np.eye(3)
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    species_1 = [1, 2, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)

    env1_1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = [1, 1, 2]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

    # set hyperparameters
    sig1 = random()
    ls1 = random()
    sig2 = random()
    ls2 = random()
    d1 = 1

    hyps = np.array([sig1, ls1, sig2, ls2])

    # check force kernel
    calc1 = en.two_body_en(env1_2, env2, hyps[0:2], cutoffs)
    calc2 = en.two_body_en(env1_1, env2, hyps[0:2], cutoffs)
    calc3 = en.three_body_en(env1_2, env2, hyps[2:4], cutoffs)
    calc4 = en.three_body_en(env1_1, env2, hyps[2:4], cutoffs)

    kern_finite_diff = (calc1 - calc2) / (2 * delt) + \
        (calc3 - calc4) / (3 * delt)
    kern_analytical = \
        en.two_plus_three_force_en(env1_1, env2, d1, hyps, cutoffs)

    tol = 1e-4
    assert(np.isclose(-kern_finite_diff, kern_analytical, atol=tol))


def test_two_plus_three_body_force():
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    # create env 1
    delt = 1e-5
    cell = np.eye(3)
    cutoffs = np.array([1, 0.9])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    positions_3 = deepcopy(positions_1)
    positions_3[0][0] = -delt

    species_1 = [1, 2, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)
    test_structure_3 = struc.Structure(cell, species_1, positions_3)

    env1_1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)
    env1_3 = env.AtomicEnvironment(test_structure_3, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][1] = delt
    positions_3 = deepcopy(positions_1)
    positions_3[0][1] = -delt

    species_2 = [1, 1, 2]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    test_structure_2 = struc.Structure(cell, species_2, positions_2)
    test_structure_3 = struc.Structure(cell, species_2, positions_3)

    env2_1 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)
    env2_2 = env.AtomicEnvironment(test_structure_2, atom_2, cutoffs)
    env2_3 = env.AtomicEnvironment(test_structure_3, atom_2, cutoffs)

    # set hyperparameters
    sig1 = random()
    ls1 = random()
    sig2 = random()
    ls2 = random()
    d1 = 1
    d2 = 2

    hyps = np.array([sig1, ls1, sig2, ls2])

    # check force kernel
    calc1 = en.two_plus_three_en(env1_2, env2_2, hyps, cutoffs)
    calc2 = en.two_plus_three_en(env1_3, env2_3, hyps, cutoffs)
    calc3 = en.two_plus_three_en(env1_2, env2_3, hyps, cutoffs)
    calc4 = en.two_plus_three_en(env1_3, env2_2, hyps, cutoffs)

    kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delt**2)
    kern_analytical = en.two_plus_three_body(env1_1, env2_1,
                                             d1, d2, hyps, cutoffs)

    tol = 1e-4
    assert(np.isclose(kern_finite_diff, kern_analytical, atol=tol))


def test_two_plus_three_body_grad():
    # create env 1
    cell = np.eye(3)
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_1 = [1, 2, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    env1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = [1, 1, 2]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

    # set hyperparameters
    sig1 = random()
    ls1 = random()
    sig2 = random()
    ls2 = random()

    d1 = randint(1, 3)
    d2 = randint(1, 3)

    delta = 1e-8

    hyps = np.array([sig1, ls1, sig2, ls2])
    hyps1 = np.array([sig1+delta, ls1, sig2, ls2])
    hyps2 = np.array([sig1, ls1+delta, sig2, ls2])
    hyps3 = np.array([sig1, ls1, sig2+delta, ls2])
    hyps4 = np.array([sig1, ls1, sig2, ls2+delta])

    grad_test = en.two_plus_three_body_grad(env1, env2, d1, d2, hyps, cutoffs)

    sig1_derv_brute = (en.two_plus_three_body(env1, env2, d1, d2,
                                              hyps1, cutoffs) -
                       en.two_plus_three_body(env1, env2, d1, d2,
                                              hyps, cutoffs)) / delta

    l1_derv_brute = \
        (en.two_plus_three_body(env1, env2, d1, d2, hyps2, cutoffs) -
         en.two_plus_three_body(env1, env2, d1, d2, hyps, cutoffs)) / delta

    sig2_derv_brute = \
        (en.two_plus_three_body(env1, env2, d1, d2,
                                hyps3, cutoffs) -
         en.two_plus_three_body(env1, env2, d1, d2,
                                hyps, cutoffs)) / delta

    l2_derv_brute = \
        (en.two_plus_three_body(env1, env2, d1, d2, hyps4, cutoffs) -
         en.two_plus_three_body(env1, env2, d1, d2, hyps, cutoffs)) / delta

    tol = 1e-4
    assert(np.isclose(grad_test[1][0], sig1_derv_brute, atol=tol))
    assert(np.isclose(grad_test[1][1], l1_derv_brute, atol=tol))
    assert(np.isclose(grad_test[1][2], sig2_derv_brute, atol=tol))
    assert(np.isclose(grad_test[1][3], l2_derv_brute, atol=tol))


# -----------------------------------------------------------------------------
#                              test two body kernels
# -----------------------------------------------------------------------------

def test_two_body_force_en():
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    # create env 1
    delt = 1e-8
    cell = np.eye(3)
    cutoffs = np.array([1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    species_1 = [1, 2, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)

    env1_1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = [1, 1, 2]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

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
    delt = 1e-5
    cell = np.eye(3)
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    positions_3 = deepcopy(positions_1)
    positions_3[0][0] = -delt

    species_1 = [1, 2, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)
    test_structure_3 = struc.Structure(cell, species_1, positions_3)

    env1_1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)
    env1_3 = env.AtomicEnvironment(test_structure_3, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][1] = delt
    positions_3 = deepcopy(positions_1)
    positions_3[0][1] = -delt

    species_2 = [1, 1, 2]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    test_structure_2 = struc.Structure(cell, species_2, positions_2)
    test_structure_3 = struc.Structure(cell, species_2, positions_3)

    env2_1 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)
    env2_2 = env.AtomicEnvironment(test_structure_2, atom_2, cutoffs)
    env2_3 = env.AtomicEnvironment(test_structure_3, atom_2, cutoffs)

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
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_1 = [1, 2, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    env1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = [1, 1, 2]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

    sig = random()
    ls = random()
    d1 = randint(1, 3)
    d2 = randint(1, 3)

    hyps = np.array([sig, ls])

    grad_test = en.two_body_grad(env1, env2, d1, d2, hyps, cutoffs)

    delta = 1e-8
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
    delt = 1e-8
    cell = np.eye(3)
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    species_1 = [1, 2, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)

    env1_1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = [1, 1, 2]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

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
    delt = 1e-5
    cell = np.eye(3)
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    positions_3 = deepcopy(positions_1)
    positions_3[0][0] = -delt

    species_1 = [1, 2, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)
    test_structure_3 = struc.Structure(cell, species_1, positions_3)

    env1_1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)
    env1_3 = env.AtomicEnvironment(test_structure_3, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][1] = delt
    positions_3 = deepcopy(positions_1)
    positions_3[0][1] = -delt

    species_2 = [1, 1, 2]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    test_structure_2 = struc.Structure(cell, species_2, positions_2)
    test_structure_3 = struc.Structure(cell, species_2, positions_3)

    env2_1 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)
    env2_2 = env.AtomicEnvironment(test_structure_2, atom_2, cutoffs)
    env2_3 = env.AtomicEnvironment(test_structure_3, atom_2, cutoffs)

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
    cutoffs = np.array([1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_1 = [1, 2, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    env1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = [1, 1, 2]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

    sig = random()
    ls = random()
    d1 = randint(1, 3)
    d2 = randint(1, 3)

    hyps = np.array([sig, ls])

    grad_test = en.three_body_grad(env1, env2, d1, d2, hyps, cutoffs)

    delta = 1e-8
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


def test_masked_hyperparameter_function():
    """
    Test simple input permutations for the from_mask_to_args function
    :return:
    """

    cutoffs = [3, 4]

    # Standard sig2, ls2, sig3, ls3, noise hyp array
    with pytest.raises(NameError):
        from_mask_to_args([], {}, cutoffs)
    # -----------------------
    # Test simple input cases
    # -----------------------
    hyps_mask = {'nbond': 1, 'nspec':1, 'spec_mask':np.zeros(118)}
    hyps = [1,2,5]
    c, ns, sm, n2b, bm, n3b, tm, sig2, ls2, sig3, ls3 \
            = from_mask_to_args(hyps, hyps_mask, cutoffs)
    assert (np.equal(sig2, [1]).all())
    assert (np.equal(ls2, [2]).all())

    hyps = [3,4,5]
    hyps_mask = {'ntriplet': 1, 'nspec':1, 'spec_mask':np.zeros(118)}
    c, ns, sm, n2b, bm, n3b, tm, sig2, ls2, sig3, ls3 \
            = from_mask_to_args(hyps, hyps_mask, cutoffs)
    assert (np.equal(sig3, [3]).all())
    assert (np.equal(ls3, [4]).all())

    hyps = [1, 2, 3, 4, 5]
    hyps_mask = {'nbond': 1, 'ntriplet':1,
            'nspec':1, 'spec_mask':np.zeros(118)}
    c, ns, sm, n2b, bm, n3b, tm, sig2, ls2, sig3, ls3 \
            = from_mask_to_args(hyps, hyps_mask, cutoffs)
    assert (np.equal(sig2, [1]).all())
    assert (np.equal(ls2, [2]).all())
    assert (np.equal(sig3, [3]).all())
    assert (np.equal(ls3, [4]).all())

    hyps = [1, 2, 3, 4, 5]
    hyps_mask['map']=[0, 1, 2, 3, 4]
    hyps_mask['original'] = [1, 2, 3, 4, 5, 6]
    c, ns, sm, n2b, bm, n3b, tm, sig2, ls2, sig3, ls3 \
            = from_mask_to_args(hyps, hyps_mask, cutoffs)
    assert (np.equal(sig2, [1]).all())
    assert (np.equal(ls2, [2]).all())
    assert (np.equal(sig3, [3]).all())
    assert (np.equal(ls3, [4]).all())

    # -----------------------
    # Test simple 2+3 body input case
    # -----------------------

    # Hyps : sig21, sig22, ls21, ls22, sig31, sig32, ls31, ls32, noise
    hyps = [1.1,1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 5]
    hyps_mask = {'nbond': 2, 'ntriplet': 2,
                 'nspec':1, 'spec_mask':np.zeros(118)}
    c, ns, sm, n2b, bm, n3b, tm, sig2, ls2, sig3, ls3 \
            = from_mask_to_args(hyps, hyps_mask, cutoffs)
    assert (np.equal(sig2, [1.1, 1.2]).all())
    assert (np.equal(ls2, [2.1, 2.2]).all())
    assert (np.equal(sig3, [3.1, 3.2]).all())
    assert (np.equal(ls3, [4.1, 4.2]).all())


def test_grad_mask_function():
    """
    Test simple permutations for the from_grad_to_mask function
    :return:
    """

    grad = 'A'

    assert from_grad_to_mask(grad,hyps_mask={}) == 'A'

    # Test map for a standard slate of hyperparameters
    grad = [1, 2, 3, 4, 5]
    hyps_mask = {'map': [0, 3]}
    assert (from_grad_to_mask(grad, hyps_mask) == [1, 4]).all()


    # Test map when noise variance is included
    grad = [1, 2, 3, 4, 5]
    hyps_mask = {'map': [0, 3, 4]}
    assert (from_grad_to_mask(grad, hyps_mask) == [1, 4, 5]).all()

    # Test map if the last parameter is not  sigma_noise
    grad = [1, 2, 3, 4, 5]
    hyps_mask = {'map': [0, 3, 5]}
    assert (from_grad_to_mask(grad, hyps_mask) == [1, 4]).all()


# -----------------------------------------------------------------------------
#                              test many body kernels
# -----------------------------------------------------------------------------


def test_many_body_force():
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    # create env 1
    delt = 1e-5
    cell = 10.0 * np.eye(3)
    cutoffs = np.array([1.2, 1.2, 1.2])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([0., 1., 0.]) + 0.1 * np.array([random(), random(), random()]),
                   np.array([1., 0., 0.]) + 0.1 * np.array([random(), random(), random()]),
                   np.array([1., 1., 0.]) + 0.1 * np.array([random(), random(), random()])]

    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    positions_3 = deepcopy(positions_1)
    positions_3[0][0] = -delt

    species_1 = [1, 1, 1, 1]
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)
    test_structure_3 = struc.Structure(cell, species_1, positions_3)

    env1_1_0 = env.AtomicEnvironment(test_structure_1, 0, cutoffs)

    env1_2_0 = env.AtomicEnvironment(test_structure_2, 0, cutoffs)
    env1_2_1 = env.AtomicEnvironment(test_structure_2, 1, cutoffs)
    env1_2_2 = env.AtomicEnvironment(test_structure_2, 2, cutoffs)

    env1_3_0 = env.AtomicEnvironment(test_structure_3, 0, cutoffs)
    env1_3_1 = env.AtomicEnvironment(test_structure_3, 1, cutoffs)
    env1_3_2 = env.AtomicEnvironment(test_structure_3, 2, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([0., 1., 0.]) + 0.1 * np.array([random(), random(), random()]),
                   np.array([1., 0., 0.]) + 0.1 * np.array([random(), random(), random()]),
                   np.array([1., 1., 0.]) + 0.1 * np.array([random(), random(), random()])]

    positions_2 = deepcopy(positions_1)
    positions_2[0][1] = delt
    positions_3 = deepcopy(positions_1)
    positions_3[0][1] = -delt

    species_2 = [1, 1, 1, 1]
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    test_structure_2 = struc.Structure(cell, species_2, positions_2)
    test_structure_3 = struc.Structure(cell, species_2, positions_3)

    env2_1_0 = env.AtomicEnvironment(test_structure_1, 0, cutoffs)

    env2_2_0 = env.AtomicEnvironment(test_structure_2, 0, cutoffs)
    env2_2_1 = env.AtomicEnvironment(test_structure_2, 1, cutoffs)
    env2_2_2 = env.AtomicEnvironment(test_structure_2, 2, cutoffs)

    env2_3_0 = env.AtomicEnvironment(test_structure_3, 0, cutoffs)
    env2_3_1 = env.AtomicEnvironment(test_structure_3, 1, cutoffs)
    env2_3_2 = env.AtomicEnvironment(test_structure_3, 2, cutoffs)

    sig = random()
    ls = random()
    d1 = 1
    d2 = 2

    hyps = np.array([sig, ls])

    # check force kernel
    calc1 = en.many_body_en(env1_2_0, env2_2_0, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_0, env2_3_0, hyps, cutoffs)
    calc3 = en.many_body_en(env1_2_0, env2_3_0, hyps, cutoffs)
    calc4 = en.many_body_en(env1_3_0, env2_2_0, hyps, cutoffs)

    kern_finite_diff_00 = (calc1 + calc2 - calc3 - calc4) / (4 * delt ** 2)

    # check force kernel
    calc1 = en.many_body_en(env1_2_0, env2_2_1, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_0, env2_3_1, hyps, cutoffs)
    calc3 = en.many_body_en(env1_2_0, env2_3_1, hyps, cutoffs)
    calc4 = en.many_body_en(env1_3_0, env2_2_1, hyps, cutoffs)

    kern_finite_diff_01 = (calc1 + calc2 - calc3 - calc4) / (4 * delt ** 2)

    # check force kernel
    calc1 = en.many_body_en(env1_2_0, env2_2_2, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_0, env2_3_2, hyps, cutoffs)
    calc3 = en.many_body_en(env1_2_0, env2_3_2, hyps, cutoffs)
    calc4 = en.many_body_en(env1_3_0, env2_2_2, hyps, cutoffs)

    kern_finite_diff_02 = (calc1 + calc2 - calc3 - calc4) / (4 * delt ** 2)

    # check force kernel
    calc1 = en.many_body_en(env1_2_1, env2_2_0, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_1, env2_3_0, hyps, cutoffs)
    calc3 = en.many_body_en(env1_2_1, env2_3_0, hyps, cutoffs)
    calc4 = en.many_body_en(env1_3_1, env2_2_0, hyps, cutoffs)

    kern_finite_diff_10 = (calc1 + calc2 - calc3 - calc4) / (4 * delt ** 2)

    # check force kernel
    calc1 = en.many_body_en(env1_2_1, env2_2_1, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_1, env2_3_1, hyps, cutoffs)
    calc3 = en.many_body_en(env1_2_1, env2_3_1, hyps, cutoffs)
    calc4 = en.many_body_en(env1_3_1, env2_2_1, hyps, cutoffs)

    kern_finite_diff_11 = (calc1 + calc2 - calc3 - calc4) / (4 * delt ** 2)

    # check force kernel
    calc1 = en.many_body_en(env1_2_1, env2_2_2, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_1, env2_3_2, hyps, cutoffs)
    calc3 = en.many_body_en(env1_2_1, env2_3_2, hyps, cutoffs)
    calc4 = en.many_body_en(env1_3_1, env2_2_2, hyps, cutoffs)

    kern_finite_diff_12 = (calc1 + calc2 - calc3 - calc4) / (4 * delt ** 2)

    # check force kernel
    calc1 = en.many_body_en(env1_2_2, env2_2_0, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_2, env2_3_0, hyps, cutoffs)
    calc3 = en.many_body_en(env1_2_2, env2_3_0, hyps, cutoffs)
    calc4 = en.many_body_en(env1_3_2, env2_2_0, hyps, cutoffs)

    kern_finite_diff_20 = (calc1 + calc2 - calc3 - calc4) / (4 * delt ** 2)

    # check force kernel
    calc1 = en.many_body_en(env1_2_2, env2_2_1, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_2, env2_3_1, hyps, cutoffs)
    calc3 = en.many_body_en(env1_2_2, env2_3_1, hyps, cutoffs)
    calc4 = en.many_body_en(env1_3_2, env2_2_1, hyps, cutoffs)

    kern_finite_diff_21 = (calc1 + calc2 - calc3 - calc4) / (4 * delt ** 2)

    # check force kernel
    calc1 = en.many_body_en(env1_2_2, env2_2_2, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_2, env2_3_2, hyps, cutoffs)
    calc3 = en.many_body_en(env1_2_2, env2_3_2, hyps, cutoffs)
    calc4 = en.many_body_en(env1_3_2, env2_2_2, hyps, cutoffs)

    kern_finite_diff_22 = (calc1 + calc2 - calc3 - calc4) / (4 * delt ** 2)

    kern_finite_diff = (kern_finite_diff_00 + kern_finite_diff_01 + kern_finite_diff_02 +
                        kern_finite_diff_10 + kern_finite_diff_11 + kern_finite_diff_12 +
                        kern_finite_diff_20 + kern_finite_diff_21 + kern_finite_diff_22)

    kern_analytical = en.many_body(env1_1_0, env2_1_0,
                                   d1, d2, hyps, cutoffs)

    tol = 1e-4

    assert (np.isclose(kern_finite_diff, kern_analytical, atol=tol))


def test_many_body_force_en():
    """Check that the analytical force-energy kernel matches finite difference of
    energy kernel."""

    # TODO: why env1_1_1, env1_1_2 and env1_1_3 (and other variables) are never used?
    # create env 1
    delt = 1e-5
    cell = 10.0 * np.eye(3)
    cutoffs = np.array([1.2, 1.2, 1.2])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([0., 1., 0.]) + 0.1 * np.array([random(), random(), random()]),
                   np.array([1., 0., 0.]) + 0.1 * np.array([random(), random(), random()]),
                   np.array([1., 1., 0.]) + 0.1 * np.array([random(), random(), random()])]

    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    positions_3 = deepcopy(positions_1)
    positions_3[0][0] = -delt

    species_1 = [1, 1, 1, 1]

    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)
    test_structure_3 = struc.Structure(cell, species_1, positions_3)

    env1_1_0 = env.AtomicEnvironment(test_structure_1, 0, cutoffs)

    env1_2_0 = env.AtomicEnvironment(test_structure_2, 0, cutoffs)
    env1_2_1 = env.AtomicEnvironment(test_structure_2, 1, cutoffs)
    env1_2_2 = env.AtomicEnvironment(test_structure_2, 2, cutoffs)

    env1_3_0 = env.AtomicEnvironment(test_structure_3, 0, cutoffs)
    env1_3_1 = env.AtomicEnvironment(test_structure_3, 1, cutoffs)
    env1_3_2 = env.AtomicEnvironment(test_structure_3, 2, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([0., 1., 0.]) + 0.1 * np.array([random(), random(), random()]),
                   np.array([1., 0., 0.]) + 0.1 * np.array([random(), random(), random()]),
                   np.array([1., 1., 0.]) + 0.1 * np.array([random(), random(), random()])]

    species_2 = [1, 1, 1, 1]

    test_structure_1 = struc.Structure(cell, species_2, positions_1)

    env2_1_0 = env.AtomicEnvironment(test_structure_1, 0, cutoffs)

    sig = random()
    ls = random()
    d1 = 1

    hyps = np.array([sig, ls])

    # check force kernel
    calc1 = en.many_body_en(env1_2_0, env2_1_0, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_0, env2_1_0, hyps, cutoffs)
    kern_finite_diff_00 = (calc1 - calc2) / (2 * delt)

    calc1 = en.many_body_en(env1_2_1, env2_1_0, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_1, env2_1_0, hyps, cutoffs)
    kern_finite_diff_10 = (calc1 - calc2) / (2 * delt)

    calc1 = en.many_body_en(env1_2_2, env2_1_0, hyps, cutoffs)
    calc2 = en.many_body_en(env1_3_2, env2_1_0, hyps, cutoffs)
    kern_finite_diff_20 = (calc1 - calc2) / (2 * delt)

    kern_finite_diff = -(kern_finite_diff_00 + kern_finite_diff_10 + kern_finite_diff_20)

    kern_analytical = en.many_body_force_en(env1_1_0, env2_1_0,
                                            d1, hyps, cutoffs)

    tol = 1e-4

    assert (np.isclose(kern_finite_diff, kern_analytical, atol=tol))


def test_many_body_grad():
    # create env 1
    cell = np.eye(3)
    cutoffs = np.array([1, 1, 1])

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_1 = [1, 1, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    env1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = [1, 1, 1]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    env2 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

    sig = random()
    ls = random()

    d1 = randint(1, 3)
    d2 = randint(1, 3)

    hyps = np.array([sig, ls])

    grad_test = en.many_body_grad(env1, env2, d1, d2, hyps, cutoffs)

    delta = 1e-8
    new_sig = sig + delta
    new_ls = ls + delta

    sig_derv_brute = (en.many_body(env1, env2, d1, d2,
                                   np.array([new_sig, ls]),
                                   cutoffs) -
                      en.many_body(env1, env2, d1, d2,
                                   hyps, cutoffs)) / delta

    l_derv_brute = (en.many_body(env1, env2, d1, d2,
                                 np.array([sig, new_ls]),
                                 cutoffs) -
                    en.many_body(env1, env2, d1, d2,
                                 hyps, cutoffs)) / delta

    tol = 1e-4
    assert (np.isclose(grad_test[1][0], sig_derv_brute, atol=tol))
    assert (np.isclose(grad_test[1][1], l_derv_brute, atol=tol))
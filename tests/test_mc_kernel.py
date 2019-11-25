import sys
from copy import deepcopy
import pytest
import numpy as np
from numpy.random import random, randint

from flare import env, struc, gp
from flare.mc_simple import _str_to_kernel as stk


def generate_envs(cutoffs, delta):
    # create env 1
    cell = np.eye(3)

    positions_1 = np.vstack([[0, 0, 0], random([3, 3])])
    positions_2 = np.copy(positions_1)
    positions_2[0][0] = delta
    positions_3 = np.copy(positions_1)
    positions_3[0][0] = -delta

    species_1 = [1, 2, 1, 1, 1, 1, 2, 1, 2]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)
    test_structure_2 = struc.Structure(cell, species_1, positions_2)
    test_structure_3 = struc.Structure(cell, species_1, positions_3)

    env1_1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = env.AtomicEnvironment(test_structure_2, atom_1, cutoffs)
    env1_3 = env.AtomicEnvironment(test_structure_3, atom_1, cutoffs)

    # create env 2
    positions_1 = np.vstack([[0, 0, 0], random([3, 3])])
    positions_2 = np.copy(positions_1)
    positions_2[0][1] = delta
    positions_3 = np.copy(positions_1)
    positions_3[0][1] = -delta

    atom_2 = 0
    species_2 = [1, 1, 2, 1, 2, 1, 2, 2, 2]

    test_structure_1 = struc.Structure(cell, species_2, positions_1)
    test_structure_2 = struc.Structure(cell, species_2, positions_2)
    test_structure_3 = struc.Structure(cell, species_2, positions_3)

    env2_1 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)
    env2_2 = env.AtomicEnvironment(test_structure_2, atom_2, cutoffs)
    env2_3 = env.AtomicEnvironment(test_structure_3, atom_2, cutoffs)

    return env1_1, env1_2, env1_3, env2_1, env2_2, env2_3


def generate_hm(nbond, ntriplet):

    if bool(nbond>0)!=bool(ntriplet>0):
        return np.array([random(), random()])
    else:
        return np.array([random(), random(), random(), random()])

@pytest.mark.parametrize('kernel_name, nbond, ntriplet',
                         [ ('two_body_mc', 1, 0),
                           ('three_body_mc', 0, 1),
                           ('two_plus_three_mc', 1, 1) ]
                         )

def test_force_en(kernel_name, nbond, ntriplet):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    cutoffs = np.array([1, 1])
    delta = 1e-8
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(cutoffs, delta)

    # set hyperparameters
    d1 = 1
    hyps = generate_hm(nbond, ntriplet)

    force_en_kernel = stk[kernel_name+"_force_en"]
    en_kernel = stk[kernel_name+"_en"]
    if bool('two' in kernel_name) != bool('three' in kernel_name):

        # check force kernel
        calc1 = en_kernel(env1_2, env2_1, hyps, cutoffs)
        calc2 = en_kernel(env1_1, env2_1, hyps, cutoffs)

        kern_finite_diff = (calc1 - calc2) / delta
        if ('two' in kernel_name):
            kern_finite_diff /= 2
        else:
            kern_finite_diff /= 3
    else:
        en2_kernel = stk['two_body_mc_en']
        en3_kernel = stk['three_body_mc_en']
        # check force kernel
        calc1 = en2_kernel(env1_2, env2_1, hyps[0:nbond*2], cutoffs)
        calc2 = en2_kernel(env1_1, env2_1, hyps[0:nbond*2], cutoffs)
        kern_finite_diff = (calc1 - calc2) / 2.0 / delta
        calc1 = en3_kernel(env1_2, env2_1, hyps[nbond*2:], cutoffs)
        calc2 = en3_kernel(env1_1, env2_1, hyps[nbond*2:], cutoffs)
        kern_finite_diff += (calc1 - calc2) / 3.0 / delta

    kern_analytical = force_en_kernel(env1_1, env2_1, d1, hyps, cutoffs)

    tol = 1e-4
    assert(np.isclose(-kern_finite_diff, kern_analytical, atol=tol))

@pytest.mark.parametrize('kernel_name, nbond, ntriplet',
                         [ ('two_body_mc', 1, 0),
                           ('three_body_mc', 0, 1),
                           ('two_plus_three_body_mc', 1, 1) ]
                         )
def test_force(kernel_name, nbond, ntriplet):
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    # create env 1
    delta = 1e-5
    cutoffs = np.array([1, 1])
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(cutoffs, delta)

    # set hyperparameters
    hyps = generate_hm(nbond, ntriplet)
    d1 = 1
    d2 = 2

    kernel = stk[kernel_name]
    if bool('two' in kernel_name) != bool('three' in kernel_name):
        en_kernel = stk[kernel_name+"_en"]
    else:
        en_kernel = stk['two_plus_three_mc_en']

    # check force kernel
    calc1 = en_kernel(env1_2, env2_2, hyps, cutoffs)
    calc2 = en_kernel(env1_3, env2_3, hyps, cutoffs)
    calc3 = en_kernel(env1_2, env2_3, hyps, cutoffs)
    calc4 = en_kernel(env1_3, env2_2, hyps, cutoffs)

    kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delta**2)
    kern_analytical = kernel(env1_1, env2_1,
                             d1, d2, hyps, cutoffs)
    tol = 1e-4
    assert(np.isclose(kern_finite_diff, kern_analytical, atol=tol))


@pytest.mark.parametrize('kernel_name, nbond, ntriplet',
                         [ ('two_body_mc', 1, 0),
                           ('three_body_mc', 0, 1),
                           ('two_plus_three_body_mc', 1, 1) ]
                         )
def test_hyps_grad(kernel_name, nbond, ntriplet):

    delta = 1e-8
    cutoffs = np.array([1, 1])
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(cutoffs, delta)

    hyps = generate_hm(nbond, ntriplet)
    d1 = randint(1, 3)
    d2 = randint(1, 3)

    kernel = stk[kernel_name]
    kernel_grad = stk[kernel_name+"_grad"]

    grad_test = kernel_grad(env1_1, env2_1,
                            d1, d2, hyps, cutoffs)

    tol = 1e-4
    original = kernel(env1_1, env2_1, d1, d2,
                      hyps, cutoffs)
    for i in range(len(hyps)):
        newhyps = np.copy(hyps)
        newhyps[i] += delta
        hgrad = (kernel(env1_1, env2_1, d1, d2, newhyps,
                        cutoffs)-
                 original)/delta
        assert(np.isclose(grad_test[1][i], hgrad, atol=tol))

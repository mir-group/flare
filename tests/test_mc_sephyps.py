import sys
from copy import deepcopy

import pytest
import numpy as np
from numpy.random import random, randint

from flare import env, struc, gp
from flare.kernels.mc_sephyps import _str_to_kernel as stk

from .fake_gp import generate_hm

def generate_envs(cutoffs, delta):
    # create env 1
    cell = np.eye(3)

    positions_1 = np.vstack([[0, 0, 0], random([3, 3])])
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delta
    positions_3 = deepcopy(positions_1)
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
    positions_2 = deepcopy(positions_1)
    positions_2[0][1] = delta
    positions_3 = deepcopy(positions_1)
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

@pytest.mark.parametrize('kernel_name, nbond, ntriplet, constraint',
                         [ ('two_body_mc', 2, 0, True),
                           ('two_body_mc', 2, 0, False),
                           ('three_body_mc', 0, 2, True),
                           ('three_body_mc', 0, 2, False),
                           ('two_plus_three_mc', 2, 2, True),
                           ('two_plus_three_mc', 2, 2, False) ]
                         )
def test_force_en(kernel_name, nbond, ntriplet, constraint):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    cutoffs = np.array([1, 1])
    delta = 1e-8
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(cutoffs, delta)

    # set hyperparameters
    d1 = 1
    hyps, hm, cut = generate_hm(nbond, ntriplet, cutoffs, constraint)

    force_en_kernel = stk[kernel_name+"_force_en"]
    en_kernel = stk[kernel_name+"_en"]
    if bool('two' in kernel_name) != bool('three' in kernel_name):

        # check force kernel
        calc1 = en_kernel(env1_2, env2_1, hyps, cut,
                hyps_mask=hm)
        calc2 = en_kernel(env1_1, env2_1, hyps, cut,
                hyps_mask=hm)

        kern_finite_diff = (calc1 - calc2) / delta
        if ('two' in kernel_name):
            kern_finite_diff /= 2
        else:
            kern_finite_diff /= 3
    else:
        en2_kernel = stk['two_body_mc_en']
        en3_kernel = stk['three_body_mc_en']
        # check force kernel
        hm2 = deepcopy(hm)
        hm3 = deepcopy(hm)
        if ('map' in hm):
            hm2['original'] = np.hstack([hm2['original'][0:nbond*2], hm2['original'][-1]])
            hm2['map'] = np.array([1, 3, 4])
            hm3['original'] = hm3['original'][nbond*2:]
            hm3['map'] = np.array([1, 3, 4])
            nbond = 1

        hm2['ntriplet']=0
        hm3['nbond']=0

        calc1 = en2_kernel(env1_2, env2_1, hyps[0:nbond*2], cutoffs,
                hyps_mask=hm2)
        calc2 = en2_kernel(env1_1, env2_1, hyps[0:nbond*2], cutoffs,
                hyps_mask=hm2)
        kern_finite_diff = (calc1 - calc2) / 2.0 / delta
        calc1 = en3_kernel(env1_2, env2_1, hyps[nbond*2:-1], cutoffs,
                hyps_mask=hm3)
        calc2 = en3_kernel(env1_1, env2_1, hyps[nbond*2:-1], cutoffs,
                hyps_mask=hm3)
        kern_finite_diff += (calc1 - calc2) / 3.0 / delta

    kern_analytical = force_en_kernel(env1_1, env2_1, d1, hyps, cutoffs,
                hyps_mask=hm)

    tol = 1e-4
    assert(np.isclose(-kern_finite_diff, kern_analytical, atol=tol))

@pytest.mark.parametrize('kernel_name, nbond, ntriplet, constraint',
                         [ ('two_body_mc', 2, 0, True),
                           ('two_body_mc', 2, 0, False),
                           ('three_body_mc', 0, 2, True),
                           ('three_body_mc', 0, 2, False),
                           ('two_plus_three_mc', 2, 2, True),
                           ('two_plus_three_mc', 2, 2, False) ]
                         )
def test_force(kernel_name, nbond, ntriplet, constraint):
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    # create env 1
    delta = 1e-5
    cutoffs = np.array([1, 1])
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(cutoffs, delta)

    # set hyperparameters
    hyps, hm, cut = generate_hm(nbond, ntriplet, cutoffs, constraint)
    d1 = 1
    d2 = 2

    kernel = stk[kernel_name]
    if bool('two' in kernel_name) != bool('three' in kernel_name):
        en_kernel = stk[kernel_name+"_en"]
    else:
        en_kernel = stk['two_plus_three_mc_en']

    # check force kernel
    calc1 = en_kernel(env1_2, env2_2, hyps, cut, hyps_mask=hm)
    calc2 = en_kernel(env1_3, env2_3, hyps, cut, hyps_mask=hm)
    calc3 = en_kernel(env1_2, env2_3, hyps, cut, hyps_mask=hm)
    calc4 = en_kernel(env1_3, env2_2, hyps, cut, hyps_mask=hm)

    kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delta**2)
    kern_analytical = kernel(env1_1, env2_1,
                             d1, d2, hyps, cut, hyps_mask=hm)
    tol = 1e-4
    assert(np.isclose(kern_finite_diff, kern_analytical, atol=tol))


@pytest.mark.parametrize('kernel_name, nbond, ntriplet, constraint',
                         [ ('two_body_mc', 2, 0, True),
                           ('two_body_mc', 2, 0, False),
                           ('three_body_mc', 0, 2, True),
                           ('three_body_mc', 0, 2, False),
                           ('two_plus_three_mc', 2, 2, True),
                           ('two_plus_three_mc', 2, 2, False) ]
                         )
def test_hyps_grad(kernel_name, nbond, ntriplet, constraint):

    delta = 1e-8
    cutoffs = np.array([1, 1])
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(cutoffs, delta)

    hyps, hm, cut = generate_hm(nbond, ntriplet, cutoffs, constraint)
    d1 = randint(1, 3)
    d2 = randint(1, 3)

    kernel = stk[kernel_name]
    kernel_grad = stk[kernel_name+"_grad"]

    k, grad = kernel_grad(env1_1, env2_1,
                            d1, d2, hyps, cut, hyps_mask=hm)
    print(kernel_name)
    print("grad", grad)
    print("hyps", hyps)

    tol = 1e-4
    original = kernel(env1_1, env2_1, d1, d2,
                      hyps, cut, hyps_mask=hm)

    nhyps = len(hyps)-1
    if ('map' in hm.keys()):
        if (hm['map'][-1] != (len(hm['original'])-1)):
            nhyps = len(hyps)
        print(hm['map'])
        original_hyps = np.copy(hm['original'])

    for i in range(nhyps):
        newhyps = np.copy(hyps)
        newhyps[i] += delta
        if ('map' in hm.keys()):
            newid = hm['map'][i]
            hm['original'] = np.copy(original_hyps)
            hm['original'][newid] += delta
        hgrad = (kernel(env1_1, env2_1, d1, d2, newhyps,
                        cut, hyps_mask=hm) -
                 original)/delta
        print(i, "hgrad", hgrad)
        assert(np.isclose(grad[i], hgrad, atol=tol))

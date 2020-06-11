import sys
from copy import deepcopy

import pytest
import numpy as np
from numpy.random import random, randint

from flare import env, struc, gp
from flare.kernels.mc_sephyps import _str_to_kernel as stk
from flare.kernels.utils import from_mask_to_args, str_to_kernel_set

from .fake_gp import generate_hm, generate_envs


def test_force_en_multi_vs_simple():
    """Check that the analytical kernel matches the one implemented
    in mc_simple.py"""

    cutoffs = np.ones(3, dtype=np.float64)
    delta = 1e-8
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = \
        generate_envs(cutoffs, delta)

    # set hyperparameters
    d1 = 1
    d2 = 2
    tol = 1e-4

    hyps, hm, cut = generate_hm(1, 1, cutoffs, False)

    # mc_simple
    kernel0, kg0, en_kernel0, force_en_kernel0, _, _ = \
        str_to_kernel_set("2+3+mb+mc", False)
    hyps = np.ones(7, dtype=np.float64)
    args0 = (hyps, cutoffs)

    # mc_sephyps
    kernel, kg, en_kernel, force_en_kernel, _, _ = \
        str_to_kernel_set("2+3+mb+mc", True)
    args1 = from_mask_to_args(hyps, hm, cutoffs)

    funcs = [[kernel0, kg0, en_kernel0, force_en_kernel0],
             [kernel, kg, en_kernel, force_en_kernel]]

    i = 0
    reference = funcs[0][i](env1_1, env2_1, d1, d2, *args0)
    result = funcs[1][i](env1_1, env2_1, d1, d2, *args1)
    assert(np.isclose(reference, result, atol=tol))

    i = 1
    reference = funcs[0][i](env1_1, env2_1, d1, d2, *args0)
    result = funcs[1][i](env1_1, env2_1, d1, d2, *args1)
    assert(np.isclose(reference[0], result[0], atol=tol))
    assert(np.isclose(reference[1], result[1], atol=tol).all())

    i = 2
    reference = funcs[0][i](env1_1, env2_1, *args0)
    result = funcs[1][i](env1_1, env2_1, *args1)
    assert(np.isclose(reference, result, atol=tol))

    i = 3
    reference = funcs[0][i](env1_1, env2_1, d1, *args0)
    result = funcs[1][i](env1_1, env2_1, d1, *args1)
    assert(np.isclose(reference, result, atol=tol))


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
    args0 = from_mask_to_args(hyps, hm, cutoffs)

    force_en_kernel = stk[kernel_name+"_force_en"]
    en_kernel = stk[kernel_name+"_en"]
    if bool('two' in kernel_name) != bool('three' in kernel_name):

        # check force kernel
        calc1 = en_kernel(env1_2, env2_1, *args0)
        calc2 = en_kernel(env1_1, env2_1, *args0)

        kern_finite_diff = (calc1 - calc2) / delta
        if ('two' in kernel_name):
            kern_finite_diff *= 2
        else:
            kern_finite_diff *= 3
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

        args2 = from_mask_to_args(hyps[0:nbond*2], hm2, cutoffs)

        calc1 = en2_kernel(env1_2, env2_1, *args2)
        calc2 = en2_kernel(env1_1, env2_1, *args2)
        kern_finite_diff = 4 * (calc1 - calc2) / 2.0 / delta

        args3 = from_mask_to_args(hyps[nbond*2:-1], hm3, cutoffs)

        calc1 = en3_kernel(env1_2, env2_1, *args3)
        calc2 = en3_kernel(env1_1, env2_1, *args3)
        kern_finite_diff += 9 * (calc1 - calc2) / 3.0 / delta

    kern_analytical = force_en_kernel(env1_1, env2_1, d1, *args0)

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
    args0 = from_mask_to_args(hyps, hm, cutoffs)
    d1 = 1
    d2 = 2

    kernel = stk[kernel_name]
    
    en2_kernel = stk['two_body_mc_en']
    en3_kernel = stk['three_body_mc_en']
    calc1 = 0
    calc2 = 0
    calc3 = 0
    calc4 = 0
    if 'two' in kernel_name:
        calc1 += 4 * en2_kernel(env1_2, env2_2, *args0)
        calc2 += 4 * en2_kernel(env1_3, env2_3, *args0)
        calc3 += 4 * en2_kernel(env1_2, env2_3, *args0)
        calc4 += 4 * en2_kernel(env1_3, env2_2, *args0)
    if 'three' in kernel_name:
        calc1 += 9 * en3_kernel(env1_2, env2_2, *args0)
        calc2 += 9 * en3_kernel(env1_3, env2_3, *args0)
        calc3 += 9 * en3_kernel(env1_2, env2_3, *args0)
        calc4 += 9 * en3_kernel(env1_3, env2_2, *args0)

    kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delta**2)
    kern_analytical = kernel(env1_1, env2_1, d1, d2, *args0)
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

    np.random.seed(0)

    delta = 1e-8
    cutoffs = np.array([1, 1])
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(cutoffs, delta)

    hyps, hm, cut = generate_hm(nbond, ntriplet, cutoffs, constraint)
    args = from_mask_to_args(hyps, hm, cutoffs)
    d1 = 1
    d2 = 2

    kernel = stk[kernel_name]
    kernel_grad = stk[kernel_name+"_grad"]

    # compute analytical values
    k, grad = kernel_grad(env1_1, env2_1, d1, d2, *args)

    print(kernel_name)
    print("grad", grad)
    print("hyps", hyps)

    tol = 1e-4
    original = kernel(env1_1, env2_1, d1, d2, *args)

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
        newargs = from_mask_to_args(newhyps, hm, cutoffs)

        hgrad = (kernel(env1_1, env2_1, d1, d2, *newargs) - original)/delta
        if ('map' in hm.keys()):
            print(i, "hgrad", hgrad, grad[hm['map'][i]])
            assert(np.isclose(grad[hm['map'][i]], hgrad, atol=tol))
        else:
            print(i, "hgrad", hgrad, grad[i])
            assert(np.isclose(grad[i], hgrad, atol=tol))

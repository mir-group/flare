import sys
from copy import deepcopy

import pytest
import numpy as np
from numpy.random import random, randint

from flare import env, struc, gp
from flare.kernels.mc_sephyps import _str_to_kernel as stk
from flare.kernels.utils import from_mask_to_args, str_to_kernel_set

from .fake_gp import generate_hm, generate_envs, another_env

list_to_test = ['2', '3', '2+3', '2+3+mb']

@pytest.mark.parametrize('kernel_name', list_to_test)
def test_force_en_multi_vs_simple(kernel_name):
    """Check that the analytical kernel matches the one implemented
    in mc_simple.py"""

    cutoffs = np.ones(3)
    delta = 1e-8
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(
        cutoffs, delta)

    # set hyperparameters
    d1 = 1
    d2 = 2
    tol = 1e-4

    cutoffs, hyps0, hyps1, hyps2, hm1, hm2 = generate_same_hm(kernel_name)

    # mc_simple
    kernel0, kg0, en_kernel0, force_en_kernel0 = str_to_kernel_set(
        kernel_name, False)
    args0 = (hyps0, cutoffs)

    # mc_sephyps
    kernel, kg, en_kernel, force_en_kernel = str_to_kernel_set(
        kernel_name, True)
    args1 = from_mask_to_args(hyps1, hm1, cutoffs)
    args2 = from_mask_to_args(hyps2, hm2, cutoffs)

    funcs = [[kernel0, kg0, en_kernel0, force_en_kernel0],
             [kernel, kg, en_kernel, force_en_kernel]]

    i = 0
    reference = funcs[0][i](env1_1, env2_1, d1, d2, *args0)
    result = funcs[1][i](env1_1, env2_1, d1, d2, *args1)
    assert(np.isclose(reference, result, atol=tol))
    result = funcs[1][i](env1_1, env2_1, d1, d2, *args2)
    assert(np.isclose(reference, result, atol=tol))

    i = 1
    reference = funcs[0][i](env1_1, env2_1, d1, d2, *args0)
    result = funcs[1][i](env1_1, env2_1, d1, d2, *args1)
    assert(np.isclose(reference[0], result[0], atol=tol))
    assert(np.isclose(reference[1], result[1], atol=tol).all())
    result = funcs[1][i](env1_1, env2_1, d1, d2, *args2)
    assert(np.isclose(reference[0], result[0], atol=tol))
    joint_grad = np.zeros(len(result[1])//2)
    for i in range(joint_grad.shape[0]):
        joint_grad[i] = result[1][i*2] + result[1][i*2+1]
    assert(np.isclose(reference[1], joint_grad, atol=tol).all())

    i = 2
    reference = funcs[0][i](env1_1, env2_1, *args0)
    result = funcs[1][i](env1_1, env2_1, *args1)
    assert(np.isclose(reference, result, atol=tol))
    result = funcs[1][i](env1_1, env2_1, *args2)
    assert(np.isclose(reference, result, atol=tol))

    i = 3
    reference = funcs[0][i](env1_1, env2_1, d1, *args0)
    result = funcs[1][i](env1_1, env2_1, d1, *args1)
    assert(np.isclose(reference, result, atol=tol))
    result = funcs[1][i](env1_1, env2_1, d1, *args2)
    assert(np.isclose(reference, result, atol=tol))


@pytest.mark.parametrize('kernel_name, nbond, ntriplet, constraint',
                         [ ('2+3', 2, 2, True)]
                         )
def test_constraint(kernel_name, nbond, ntriplet, constraint):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    cutoffs = np.array([1, 1])
    delta = 1e-8
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(
        cutoffs, delta)

    # set hyperparameters
    d1 = 1

    hyps, hm, cut = generate_hm(nbond, ntriplet, cutoffs, constraint)
    args0 = from_mask_to_args(hyps, hm, cutoffs)

    force_en_kernel = stk[kernel_name+"_force_en"]
    en_kernel = stk[kernel_name+"_en"]
    en2_kernel = stk['2_en']
    en3_kernel = stk['3_en']
    # check force kernel
    hm2 = deepcopy(hm)
    hm3 = deepcopy(hm)
    if ('map' in hm):
        hm2['original'] = np.hstack(
            [hm2['original'][0:nbond*2], hm2['original'][-1]])
        hm2['map'] = np.array([1, 3, 4])
        hm3['original'] = hm3['original'][nbond*2:]
        hm3['map'] = np.array([1, 3, 4])
        nbond = 1

    hm2['ntriplet'] = 0
    hm3['nbond'] = 0

    args2 = from_mask_to_args(hyps[0:nbond*2], hm2, cutoffs)

    calc1 = en2_kernel(env1_2, env2_1, *args2)
    calc2 = en2_kernel(env1_1, env2_1, *args2)
    kern_finite_diff = (calc1 - calc2) / 2.0 / delta

    args3 = from_mask_to_args(hyps[nbond*2:-1], hm3, cutoffs)

    calc1 = en3_kernel(env1_2, env2_1, *args3)
    calc2 = en3_kernel(env1_1, env2_1, *args3)
    kern_finite_diff += (calc1 - calc2) / 3.0 / delta

    kern_analytical = force_en_kernel(env1_1, env2_1, d1, *args0)

    tol = 1e-4
    assert(np.isclose(-kern_finite_diff, kern_analytical, atol=tol))

@pytest.mark.parametrize('kernel_name', list_to_test)
def test_force_en(kernel_name):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    cutoffs, hyps, hm = generate_diff_hm(kernel_name)
    args = from_mask_to_args(hyps, hm, cutoffs)

    delta = 1e-8
    env1_1, env1_2, env1_3, \
        env1_2_1, env1_3_1, env1_2_2, env1_3_2, env2_1 \
        = another_env(cutoffs, delta, mask=hm)

    # set hyperparameters
    d1 = 1
    # hyps = generate_hm(kernel_name)

    _, __, en_kernel, force_en_kernel = str_to_kernel_set(kernel_name)

    nterm = 0
    for term in ['2', '3', 'mb']:
        if (term in kernel_name):
            nterm += 1

    kern_finite_diff = 0
    if ('mb' in kernel_name):
        _, __, enm_kernel, ___ = str_to_kernel_set('mb')
        mhyps = hyps[(nterm-1)*2:]
        calc1 = enm_kernel(env1_2, env2_1, mhyps, cutoffs)
        calc2 = enm_kernel(env1_3, env2_1, mhyps, cutoffs)
        kern_finite_diff_00 = (calc1 - calc2) / (2 * delta)

        calc1 = enm_kernel(env1_2_1, env2_1, mhyps, cutoffs)
        calc2 = enm_kernel(env1_3_1, env2_1, mhyps, cutoffs)
        kern_finite_diff_10 = (calc1 - calc2) / (2 * delta)

        calc1 = enm_kernel(env1_2_2, env2_1, mhyps, cutoffs)
        calc2 = enm_kernel(env1_3_2, env2_1, mhyps, cutoffs)
        kern_finite_diff_20 = (calc1 - calc2) / (2 * delta)

        mb_diff = (kern_finite_diff_00 +
                   kern_finite_diff_10 + kern_finite_diff_20)

        kern_finite_diff += mb_diff

    if ('2' in kernel_name):
        nbond = 1
        _, __, en2_kernel, ___ = str_to_kernel_set('2mc')
        calc1 = en2_kernel(env1_2, env2_1, hyps[0:nbond * 2], cutoffs)
        calc2 = en2_kernel(env1_1, env2_1, hyps[0:nbond * 2], cutoffs)
        diff2b = (calc1 - calc2) / 2.0 / delta



@pytest.mark.parametrize('kernel_name, nbond, ntriplet, constraint',
                         [('2', 2, 0, True),
                          ('2', 2, 0, False),
                          ('3', 0, 2, True),
                          ('3', 0, 2, False),
                          ('2+3', 2, 2, True),
                          ('2+3', 2, 2, False)]
                         )
def test_force(kernel_name, nbond, ntriplet, constraint):
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    # create env 1
    delta = 1e-5
    cutoffs = np.array([1, 1])
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(
        cutoffs, delta)

    # set hyperparameters
    hyps, hm, cut = generate_hm(nbond, ntriplet, cutoffs, constraint)
    args0 = from_mask_to_args(hyps, hm, cutoffs)
    d1 = 1
    d2 = 2

    kernel = stk[kernel_name]
    if bool('2' in kernel_name) != bool('3' in kernel_name):
        en_kernel = stk[kernel_name+"_en"]
    else:
        en_kernel = stk['2+3_en']

    # check force kernel
    calc1 = en_kernel(env1_2, env2_2, *args0)
    calc2 = en_kernel(env1_3, env2_3, *args0)
    calc3 = en_kernel(env1_2, env2_3, *args0)
    calc4 = en_kernel(env1_3, env2_2, *args0)

    kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delta**2)
    kern_analytical = kernel(env1_1, env2_1,
                             d1, d2, *args0)
    tol = 1e-4
    assert(np.isclose(kern_finite_diff, kern_analytical, atol=tol))


@pytest.mark.parametrize('kernel_name, nbond, ntriplet, constraint',
                         [('2', 2, 0, True),
                          ('2', 2, 0, False),
                          ('3', 0, 2, True),
                          ('3', 0, 2, False),
                          ('2+3', 2, 2, True),
                          ('2+3', 2, 2, False)]
                         )
def test_hyps_grad(kernel_name, nbond, ntriplet, constraint):

    np.random.seed(0)

    delta = 1e-8
    cutoffs = np.array([1, 1])
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(
        cutoffs, delta)

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


def generate_same_hm(kernel_name):
    cutoffs = []
    hyps0 = []
    hyps2 = []
    hm1 = {'nspec': 1, 'spec_mask': np.zeros(118, dtype=int)}
    hm2 = {'nspec': 2, 'spec_mask': np.zeros(118, dtype=int)}
    hm2['spec_mask'][2] = 1
    if ('2' in kernel_name):
        cutoffs = np.ones(1)

        hyps0 += [1, 0.9]
        hm1['nbond'] = 1
        hm1['bond_mask'] = np.zeros(1, dtype=int)

        hyps2 += [1, 1, 0.9, 0.9]
        hm2['nbond'] = 2
        hm2['bond_mask'] = np.ones(4, dtype=int)
        hm2['bond_mask'][0] = 0
        hm2['cutoff_2b'] = np.ones(2)
    if ('3' in kernel_name):
        cutoffs = np.ones(2)

        hyps0 += [1, 0.8]

        hm1['ntriplet'] = 1
        hm1['triplet_mask'] = np.zeros(1, dtype=int)

        hyps2 += [1, 1, 0.8, 0.8]
        hm2['ntriplet'] = 2
        hm2['triplet_mask'] = np.ones(4, dtype=int)
        hm2['triplet_mask'][0] = 0
        hm2['ncut3b'] = 2
        hm2['cut3b_mask'] = np.ones(4, dtype=int)
        hm2['cut3b_mask'][0] = 0
        hm2['cutoff_3b'] = np.ones(2)
    if ('mb' in kernel_name):
        cutoffs = np.ones(3)

        hyps0 += [1, 0.7]

        hm1['nmb'] = 1
        hm1['mb_mask'] = np.zeros(1, dtype=int)

        hyps2 += [1, 1, 0.7, 0.7]
        hm2['nmb'] = 2
        hm2['mb_mask'] = np.ones(4, dtype=int)
        hm2['mb_mask'][0] = 0
        hm2['cutoff_mb'] = np.ones(2)

    hyps0 = np.hstack([hyps0, 0.5])
    hyps2 = np.hstack([hyps2, 0.5])
    hyps1 = np.hstack([hyps0, 0.5])
    return cutoffs, hyps0, hyps1, hyps2, hm1, hm2

def generate_diff_hm(kernel_name):
    cutoffs = []
    hyps2 = []
    hm2 = {'nspec': 2, 'spec_mask': np.zeros(118, dtype=int)}
    hm2['spec_mask'][2] = 1
    if ('2' in kernel_name):
        cutoffs = np.array([1.2])

        hyps2 += [1, 0.8, 0.7, 0.6]
        hm2['nbond'] = 2
        hm2['bond_mask'] = np.ones(4, dtype=int)
        hm2['bond_mask'][0] = 0
        hm2['cutoff_2b'] = np.array([1.2, 1.25])

    if ('3' in kernel_name):
        cutoffs = np.array([1.2, 1.2])

        hyps2 += [1, 0.9, 0.8, 0.7]
        hm2['ntriplet'] = 2
        hm2['triplet_mask'] = np.ones(4, dtype=int)
        hm2['triplet_mask'][0] = 0
        hm2['ncut3b'] = 2
        hm2['cut3b_mask'] = np.ones(4, dtype=int)
        hm2['cut3b_mask'][0] = 0
        hm2['cutoff_3b'] = np.array([1.15, 1.2])

    if ('mb' in kernel_name):
        cutoffs = np.ones(3)

        hyps2 += [1, 0.9, 0.8, 0.7]
        hm2['nmb'] = 2
        hm2['mb_mask'] = np.ones(4, dtype=int)
        hm2['mb_mask'][0] = 0
        hm2['cutoff_mb'] = np.array([1.15, 1.2])

    hyps2 = np.hstack([hyps2, 0.5])
    return cutoffs, hyps2, hm2

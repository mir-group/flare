import sys
from copy import deepcopy

import pytest
import numpy as np
from numpy.random import random, randint
from numpy import isclose

from flare import env, struc, gp
from flare.kernels.mc_sephyps import _str_to_kernel as stk
from flare.kernels.utils import from_mask_to_args, str_to_kernel_set
from flare.cutoffs import quadratic_cutoff_bound

from .fake_gp import generate_mb_envs, generate_mb_twin_envs

list_to_test = ['2', '3', 'mb', '2+3', '2+3+mb']


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('diff_cutoff', [True, False])
def test_force_en_multi_vs_simple(kernel_name, diff_cutoff):
    """Check that the analytical kernel matches the one implemented
    in mc_simple.py"""

    d1 = 1
    d2 = 2
    tol = 1e-4
    cell = 1e7 * np.eye(3)

    # set hyperparameters
    cutoffs, hyps0, hyps1, hyps2, hm1, hm2 = generate_same_hm(
        kernel_name, diff_cutoff)

    delta = 1e-8
    env1 = generate_mb_envs(cutoffs, cell, delta, d1)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2)
    env1 = env1[0][0]
    env2 = env2[0][0]

    # mc_simple
    kernel0, kg0, en_kernel0, force_en_kernel0 = str_to_kernel_set(
        kernel_name, False)
    args0 = (hyps0, cutoffs)

    # mc_sephyps
    # args1 and args 2 use 1 and 2 groups of hyper-parameters
    # if (diff_cutoff), 1 or 2 group of cutoffs
    # but same value as in args0
    kernel, kg, en_kernel, force_en_kernel = str_to_kernel_set(
        kernel_name, True)
    args1 = from_mask_to_args(hyps1, hm1, cutoffs)
    args2 = from_mask_to_args(hyps2, hm2, cutoffs)

    funcs = [[kernel0, kg0, en_kernel0, force_en_kernel0],
             [kernel, kg, en_kernel, force_en_kernel]]

    # compare whether mc_sephyps and mc_simple
    # yield the same values
    i = 0
    reference = funcs[0][i](env1, env2, d1, d2, *args0)
    result = funcs[1][i](env1, env2, d1, d2, *args1)
    assert(isclose(reference, result, rtol=tol))
    print(kernel_name, i, reference, result)
    result = funcs[1][i](env1, env2, d1, d2, *args2)
    assert(isclose(reference, result, rtol=tol))
    print(kernel_name, i, reference, result)

    i = 1
    reference = funcs[0][i](env1, env2, d1, d2, *args0)
    result = funcs[1][i](env1, env2, d1, d2, *args1)
    print(kernel_name, i, reference, result)
    assert(isclose(reference[0], result[0], rtol=tol))
    assert(isclose(reference[1], result[1], rtol=tol).all())

    result = funcs[1][i](env1, env2, d1, d2, *args2)
    print(kernel_name, i, reference, result)
    assert(isclose(reference[0], result[0], rtol=tol))
    joint_grad = np.zeros(len(result[1])//2)
    for i in range(joint_grad.shape[0]):
        joint_grad[i] = result[1][i*2] + result[1][i*2+1]
    assert(isclose(reference[1], joint_grad, rtol=tol).all())

    i = 2
    reference = funcs[0][i](env1, env2, *args0)
    result = funcs[1][i](env1, env2, *args1)
    print(kernel_name, i, reference, result)
    assert(isclose(reference, result, rtol=tol))
    result = funcs[1][i](env1, env2, *args2)
    print(kernel_name, i, reference, result)
    assert(isclose(reference, result, rtol=tol))

    i = 3
    reference = funcs[0][i](env1, env2, d1, *args0)
    result = funcs[1][i](env1, env2, d1, *args1)
    print(kernel_name, i, reference, result)
    assert(isclose(reference, result, rtol=tol))
    result = funcs[1][i](env1, env2, d1, *args2)
    print(kernel_name, i, reference, result)
    assert(isclose(reference, result, rtol=tol))


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('diff_cutoff', [True, False])
def test_check_sig_scale(kernel_name, diff_cutoff):
    """Check whether the grouping is properly assign
    with four environments

    * env1 and env2 are computed from two structures with four
    atoms each. There are two species 1, 2
    * env1_t and env2_t are derived from the same structure, but
      species 2 atoms are removed.
    * only the sigma of 1-1 are non-zero
    * so using env1 and env1_t should produce the same value
    * if the separate group of hyperparameter is properly
      applied, the result should be 2**2 times of
      the reference
    """

    d1 = 1
    d2 = 2
    tol = 1e-4
    scale = 2

    cutoffs, hyps0, hm = generate_diff_hm(kernel_name, diff_cutoff)

    delta = 1e-8
    env1, env1_t = generate_mb_twin_envs(cutoffs, np.eye(3)*100, delta, d1, hm)
    env2, env2_t = generate_mb_twin_envs(cutoffs, np.eye(3)*100, delta, d2, hm)
    env1 = env1[0][0]
    env2 = env2[0][0]
    env1_t = env1_t[0][0]
    env2_t = env2_t[0][0]

    # make the second sigma zero
    hyps1 = np.copy(hyps0)
    hyps0[1::4] = 0  # 1e-8
    hyps1[1::4] = 0  # 1e-8
    hyps1[0::4] *= scale

    kernel, kg, en_kernel, force_en_kernel = str_to_kernel_set(
        kernel_name, True)

    args0 = from_mask_to_args(hyps0, hm, cutoffs)
    args1 = from_mask_to_args(hyps1, hm, cutoffs)

    reference = kernel(env1, env2, d1, d2, *args0)
    result = kernel(env1_t, env2_t, d1, d2, *args1)
    print(kernel.__name__, result, reference)
    if (reference != 0):
        assert isclose(result/reference, scale**2, rtol=tol)

    reference = kg(env1, env2, d1, d2, *args0)
    result = kg(env1_t, env2_t, d1, d2, *args1)
    print(kg.__name__, result, reference)
    if (reference[0] != 0):
        assert isclose(result[0]/reference[0], scale**2, rtol=tol)
    for idx in range(reference[1].shape[0]):
        # check sig0
        if (reference[1][idx] != 0 and (idx % 4) == 0):
            assert isclose(result[1][idx]/reference[1][idx], scale, rtol=tol)
        # check the rest, but skip sig 1
        elif (reference[1][idx] != 0 and (idx % 4) != 1):
            assert isclose(result[1][idx]/reference[1]
                           [idx], scale**2, rtol=tol)

    reference = en_kernel(env1, env2, *args0)
    result = en_kernel(env1_t, env2_t, *args1)
    print(en_kernel.__name__, result, reference)
    if (reference != 0):
        assert isclose(result/reference, scale**2, rtol=tol)

    reference = force_en_kernel(env1, env2, d1, *args0)
    result = force_en_kernel(env1_t, env2_t, d1, *args1)
    print(force_en_kernel.__name__, result, reference)
    if (reference != 0):
        assert isclose(result/reference, scale**2, rtol=tol)


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('diff_cutoff', [True, False])
def test_force_bound_cutoff_compare(kernel_name, diff_cutoff):
    """Check that the analytical kernel matches the one implemented
    in mc_simple.py"""

    d1 = 1
    d2 = 2
    tol = 1e-4
    cell = 1e7 * np.eye(3)
    delta = 1e-8

    cutoffs, hyps, hm = generate_diff_hm(kernel_name, diff_cutoff, False)
    kernel, kg, en_kernel, force_en_kernel = str_to_kernel_set(
        kernel_name, True)
    args = from_mask_to_args(hyps, hm, cutoffs)

    np.random.seed(10)
    env1 = generate_mb_envs(cutoffs, cell, delta, d1, hm)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2, hm)
    env1 = env1[0][0]
    env2 = env2[0][0]

    reference = kernel(env1, env2, d1, d2, *args, quadratic_cutoff_bound)
    result = kernel(env1, env2, d1, d2, *args)
    assert(isclose(reference, result, rtol=tol))

    reference = kg(env1, env2, d1, d2, *args, quadratic_cutoff_bound)
    result = kg(env1, env2, d1, d2, *args)
    assert(isclose(reference[0], result[0], rtol=tol))
    assert(isclose(reference[1], result[1], rtol=tol).all())

    reference = en_kernel(env1, env2, *args, quadratic_cutoff_bound)
    result = en_kernel(env1, env2, *args)
    assert(isclose(reference, result, rtol=tol))

    reference = force_en_kernel(
        env1, env2, d1, *args, quadratic_cutoff_bound)
    result = force_en_kernel(env1, env2, d1, *args)
    assert(isclose(reference, result, rtol=tol))


@pytest.mark.parametrize('kernel_name', ['2+3'])
@pytest.mark.parametrize('diff_cutoff', [True, False])
def test_constraint(kernel_name, diff_cutoff):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    if ('mb' in kernel_name):
        return

    d1 = 1
    d2 = 2
    cell = 1e7 * np.eye(3)
    delta = 1e-8

    cutoffs, hyps, hm = generate_diff_hm(
        kernel_name, diff_cutoff=diff_cutoff, constraint=True)
    _, __, en_kernel, force_en_kernel = str_to_kernel_set(kernel_name, True)
    args0 = from_mask_to_args(hyps, hm, cutoffs)

    np.random.seed(10)
    env1 = generate_mb_envs(cutoffs, cell, delta, d1, hm)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2, hm)

    kern_finite_diff = 0

    if ('2' in kernel_name):
        _, __, en2_kernel, fek2 = str_to_kernel_set('2', True)
        calc1 = en2_kernel(env1[1][0], env2[0][0], *args0)
        calc2 = en2_kernel(env1[0][0], env2[0][0], *args0)
        kern_finite_diff += (calc1 - calc2) / 2.0 / delta

    if ('3' in kernel_name):
        _, __, en3_kernel, fek3 = str_to_kernel_set('3', True)
        calc1 = en3_kernel(env1[1][0], env2[0][0], *args0)
        calc2 = en3_kernel(env1[0][0], env2[0][0], *args0)
        kern_finite_diff += (calc1 - calc2) / 3.0 / delta

    kern_analytical = force_en_kernel(env1[0][0], env2[0][0], d1, *args0)

    tol = 1e-4
    assert(isclose(-kern_finite_diff, kern_analytical, rtol=tol))


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('diff_cutoff', [True, False])
def test_force_en(kernel_name, diff_cutoff):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    delta = 1e-5
    d1 = 1
    d2 = 2
    cell = 1e7 * np.eye(3)
    np.random.seed(0)

    cutoffs, hyps, hm = generate_diff_hm(kernel_name, diff_cutoff, False)
    args = from_mask_to_args(hyps, hm, cutoffs)

    env1 = generate_mb_envs(cutoffs, cell, delta, d1, hm)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2, hm)

    _, __, en_kernel, force_en_kernel = str_to_kernel_set(kernel_name, True)

    kern_analytical = force_en_kernel(env1[0][0], env2[0][0], d1, *args)

    kern_finite_diff = 0
    if ('mb' in kernel_name):
        kernel, _, enm_kernel, efk = str_to_kernel_set('mb', True)

        calc = 0
        for i in range(len(env1[0])):
            calc += enm_kernel(env1[1][i], env2[0][0], *args)
            calc -= enm_kernel(env1[2][i], env2[0][0], *args)

        kern_finite_diff += (calc)/(2*delta)

    if ('2' in kernel_name or '3' in kernel_name):
        args23 = from_mask_to_args(hyps, hm, cutoffs[:2])

    if ('2' in kernel_name):
        kernel, _, en2_kernel, efk = str_to_kernel_set('2b', True)
        calc1 = en2_kernel(env1[1][0], env2[0][0], *args23)
        calc2 = en2_kernel(env1[2][0], env2[0][0], *args23)
        diff2b = (calc1 - calc2) / 2.0 / delta / 2.0

        kern_finite_diff += diff2b

    if ('3' in kernel_name):
        kernel, _, en3_kernel, efk = str_to_kernel_set('3b', True)
        calc1 = en3_kernel(env1[1][0], env2[0][0], *args23)
        calc2 = en3_kernel(env1[2][0], env2[0][0], *args23)
        diff3b = (calc1 - calc2) / 3.0 / delta / 2.0

        kern_finite_diff += diff3b

    print("\nforce_en", kernel_name, kern_finite_diff, kern_analytical)

    tol = 1e-3
    assert (isclose(-kern_finite_diff, kern_analytical, rtol=tol))


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('constraint', [True, False])
def test_force(kernel_name, constraint):
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    d1 = 1
    d2 = 2
    tol = 1e-4
    cell = 1e7 * np.eye(3)
    delta = 1e-5

    cutoffs, hyps, hm = generate_diff_hm(
        kernel_name, diff_cutoff=False, constraint=constraint)
    kernel, kg, en_kernel, fek = str_to_kernel_set(kernel_name, True)
    args = from_mask_to_args(hyps, hm, cutoffs)

    np.random.seed(10)
    env1 = generate_mb_envs(cutoffs, cell, delta, d1, hm)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2, hm)

    # check force kernel
    if ('mb' == kernel_name):
        cal = 0
        for i in range(3):
            for j in range(len(env1[0])):
                cal += en_kernel(env1[1][i], env2[1][j], *args)
                cal += en_kernel(env1[2][i], env2[2][j], *args)
                cal -= en_kernel(env1[1][i], env2[2][j], *args)
                cal -= en_kernel(env1[2][i], env2[1][j], *args)
        kern_finite_diff = cal / (4 * delta ** 2)
    elif ('mb' not in kernel_name):
        calc1 = en_kernel(env1[1][0], env2[1][0], *args)
        calc2 = en_kernel(env1[2][0], env2[2][0], *args)
        calc3 = en_kernel(env1[1][0], env2[2][0], *args)
        calc4 = en_kernel(env1[2][0], env2[1][0], *args)
        kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delta**2)
    else:
        return

    kern_analytical = kernel(env1[0][0], env2[0][0],
                             d1, d2, *args)
    tol = 1e-4
    assert(isclose(kern_finite_diff, kern_analytical, rtol=tol))


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('constraint', [True, False])
def test_hyps_grad(kernel_name, constraint):

    delta = 1e-8
    d1 = 1
    d2 = 2
    tol = 1e-4

    cutoffs, hyps, hm = generate_diff_hm(kernel_name, constraint=constraint)
    args = from_mask_to_args(hyps, hm, cutoffs)
    kernel, kernel_grad, _, __ = str_to_kernel_set(kernel_name, True)

    np.random.seed(0)
    env1 = generate_mb_envs(cutoffs, np.eye(3)*100, delta, d1)
    env2 = generate_mb_envs(cutoffs, np.eye(3)*100, delta, d2)
    env1 = env1[0][0]
    env2 = env2[0][0]

    # compute analytical values
    k, grad = kernel_grad(env1, env2, d1, d2, *args)

    original = kernel(env1, env2, d1, d2, *args)

    nhyps = len(hyps)-1
    if ('map' in hm.keys()):
        if (hm['map'][-1] != (len(hm['original'])-1)):
            nhyps = len(hyps)
        original_hyps = np.copy(hm['original'])

    for i in range(nhyps):
        newhyps = np.copy(hyps)
        newhyps[i] += delta
        if ('map' in hm.keys()):
            newid = hm['map'][i]
            hm['original'] = np.copy(original_hyps)
            hm['original'][newid] += delta
        newargs = from_mask_to_args(newhyps, hm, cutoffs)

        hgrad = (kernel(env1, env2, d1, d2, *newargs) - original)/delta
        if ('map' in hm.keys()):
            print(i, "hgrad", hgrad, grad[hm['map'][i]])
            assert(isclose(grad[hm['map'][i]], hgrad, rtol=tol))
        else:
            print(i, "hgrad", hgrad, grad[i])
            assert(isclose(grad[i], hgrad, rtol=tol))


def generate_same_hm(kernel_name, diff_cutoff=False):
    cutoffs = []
    hyps0 = []
    hyps2 = []
    hm1 = {'nspec': 1, 'spec_mask': np.zeros(118, dtype=int)}
    hm2 = {'nspec': 2, 'spec_mask': np.zeros(118, dtype=int)}
    hm2['spec_mask'][2] = 1
    if ('2' in kernel_name):
        cutoffs = np.ones(1, dtype=float)

        hyps0 += [1, 0.9]
        hm1['nbond'] = 1
        hm1['bond_mask'] = np.zeros(1, dtype=int)

        hyps2 += [1, 1, 0.9, 0.9]
        hm2['nbond'] = 2
        hm2['bond_mask'] = np.ones(4, dtype=int)
        hm2['bond_mask'][0] = 0
        if (diff_cutoff):
            hm1['cutoff_2b'] = np.ones(1, dtype=float)*cutoffs[0]
            hm2['cutoff_2b'] = np.ones(2, dtype=float)*cutoffs[0]
    if ('3' in kernel_name):
        cutoffs = np.ones(2)

        hyps0 += [1, 0.8]

        hm1['ntriplet'] = 1
        hm1['triplet_mask'] = np.zeros(1, dtype=int)

        hyps2 += [1, 1, 0.8, 0.8]
        hm2['ntriplet'] = 2
        hm2['triplet_mask'] = np.ones(8, dtype=int)
        hm2['triplet_mask'][0] = 0
        if (diff_cutoff):
            hm1['ncut3b'] = 1
            hm1['cut3b_mask'] = np.zeros(1, dtype=int)
            hm1['cutoff_3b'] = np.ones(1, dtype=float)*cutoffs[1]
            hm2['ncut3b'] = 2
            hm2['cut3b_mask'] = np.ones(4, dtype=int)
            hm2['cut3b_mask'][0] = 0
            hm2['cutoff_3b'] = np.ones(2, dtype=float)*cutoffs[1]
    if ('mb' in kernel_name):
        cutoffs = np.ones(3)

        hyps0 += [1, 0.7]

        hm1['nmb'] = 1
        hm1['mb_mask'] = np.zeros(1, dtype=int)

        hyps2 += [1, 1, 0.7, 0.7]

        hm2['nmb'] = 2
        hm2['mb_mask'] = np.ones(4, dtype=int)
        hm2['mb_mask'][0] = 0

        if (diff_cutoff):
            hm1['cutoff_mb'] = np.ones(1)*cutoffs[2]
            hm2['cutoff_mb'] = np.ones(2)*cutoffs[2]

    hyps0 = np.hstack([hyps0, 0.5])
    hyps2 = np.hstack([hyps2, 0.5])
    hyps1 = np.hstack([hyps0, 0.5])
    return cutoffs, hyps0, hyps1, hyps2, hm1, hm2


def generate_diff_hm(kernel_name, diff_cutoff=False, constraint=False):
    cutoffs = []
    hyps2 = []
    hm2 = {'nspec': 2, 'spec_mask': np.zeros(118, dtype=int)}
    hm2['spec_mask'][2] = 1
    if ('2' in kernel_name):
        cutoffs = np.array([2.5])

        hyps2 += [1, 0.8, 0.7, 0.05]
        hm2['nbond'] = 2
        hm2['bond_mask'] = np.ones(4, dtype=int)
        hm2['bond_mask'][0] = 0
        if (diff_cutoff):
            hm2['cutoff_2b'] = np.array([1.2, 1.25])

    if ('3' in kernel_name):
        cutoffs = np.array([2.5, 1.2])

        hyps2 += [1, 0.9, 0.8, 0.05]
        hm2['ntriplet'] = 2
        hm2['triplet_mask'] = np.ones(8, dtype=int)
        hm2['triplet_mask'][0] = 0
        if (diff_cutoff):
            hm2['ncut3b'] = 2
            hm2['cut3b_mask'] = np.ones(4, dtype=int)
            hm2['cut3b_mask'][0] = 0
            hm2['cutoff_3b'] = np.array([1.15, 1.2])

    if ('mb' in kernel_name):
        cutoffs = np.array([2, 1.2, 1.2])

        hyps2 += [1, 0.9, 0.8, 0.05]
        hm2['nmb'] = 2
        hm2['mb_mask'] = np.ones(4, dtype=int)
        hm2['mb_mask'][0] = 0
        if (diff_cutoff):
            hm2['cutoff_mb'] = np.array([1.25, 1.2])

    hyps2 = np.hstack([hyps2, 0.5])

    if (constraint is False):
        return cutoffs, hyps2, hm2

    hm2['original'] = hyps2
    hm2['map'] = np.arange(0, len(hyps2), 2)
    hm2['hyps_label'] = np.arange(0, len(hyps2), 2)
    hyps2 = hyps2[hm2['map']]
    if (len(hyps2)-1) in hm2['map']:
        hm2['train_noise'] = True
    else:
        hm2['train_noise'] = False

    return cutoffs, hyps2, hm2

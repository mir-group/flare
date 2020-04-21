import sys
from copy import deepcopy

import pytest
import numpy as np
from numpy.random import random, randint

from flare import env, struc, gp
from flare.kernels.mc_sephyps import _str_to_kernel as stk
from flare.kernels.utils import from_mask_to_args, str_to_kernel_set
from flare.cutoffs import  quadratic_cutoff_bound

from .fake_gp import generate_envs, generate_mb_envs

list_to_test = ['2', '3', 'mb', '2+3', '2+3+mb']

@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('diff_cutoff', [True, False])
def test_force_en_multi_vs_simple(kernel_name, diff_cutoff):
    """Check that the analytical kernel matches the one implemented
    in mc_simple.py"""

    cutoffs = np.ones(3, dtype=np.float64)
    delta = 1e-8
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(
        cutoffs, delta)

    # set hyperparameters
    d1 = 1
    d2 = 2
    tol = 1e-4

    cutoffs, hyps0, hyps1, hyps2, hm1, hm2 = generate_same_hm(kernel_name, diff_cutoff)

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
    print(kernel_name, i, reference, result)
    result = funcs[1][i](env1_1, env2_1, d1, d2, *args2)
    assert(np.isclose(reference, result, atol=tol))
    print(kernel_name, i, reference, result)

    i = 1
    reference = funcs[0][i](env1_1, env2_1, d1, d2, *args0)
    result = funcs[1][i](env1_1, env2_1, d1, d2, *args1)
    print(kernel_name, i, reference, result)
    assert(np.isclose(reference[0], result[0], atol=tol))
    assert(np.isclose(reference[1], result[1], atol=tol).all())
    result = funcs[1][i](env1_1, env2_1, d1, d2, *args2)
    print(kernel_name, i, reference, result)
    assert(np.isclose(reference[0], result[0], atol=tol))
    joint_grad = np.zeros(len(result[1])//2)
    for i in range(joint_grad.shape[0]):
        joint_grad[i] = result[1][i*2] + result[1][i*2+1]
    assert(np.isclose(reference[1], joint_grad, atol=tol).all())

    i = 2
    reference = funcs[0][i](env1_1, env2_1, *args0)
    result = funcs[1][i](env1_1, env2_1, *args1)
    print(kernel_name, i, reference, result)
    assert(np.isclose(reference, result, atol=tol))
    result = funcs[1][i](env1_1, env2_1, *args2)
    print(kernel_name, i, reference, result)
    assert(np.isclose(reference, result, atol=tol))

    i = 3
    reference = funcs[0][i](env1_1, env2_1, d1, *args0)
    result = funcs[1][i](env1_1, env2_1, d1, *args1)
    print(kernel_name, i, reference, result)
    assert(np.isclose(reference, result, atol=tol))
    result = funcs[1][i](env1_1, env2_1, d1, *args2)
    print(kernel_name, i, reference, result)
    assert(np.isclose(reference, result, atol=tol))

@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('diff_cutoff', [True, False])
def test_force_bound_cutoff_compare(kernel_name, diff_cutoff):
    """Check that the analytical kernel matches the one implemented
    in mc_simple.py"""

    cutoffs = np.ones(3, dtype=np.float64)
    delta = 1e-8
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(
        cutoffs, delta)

    # set hyperparameters
    d1 = 1
    d2 = 2
    tol = 1e-4

    cutoffs, hyps, hm = generate_diff_hm(kernel_name, diff_cutoff, False)

    # mc_sephyps
    kernel, kg, en_kernel, force_en_kernel = str_to_kernel_set(
        kernel_name, True)
    args = from_mask_to_args(hyps, hm, cutoffs)

    reference = kernel(env1_1, env2_1, d1, d2, *args, quadratic_cutoff_bound)
    result = kernel(env1_1, env2_1, d1, d2, *args)
    assert(np.isclose(reference, result, atol=tol))

    reference = kg(env1_1, env2_1, d1, d2, *args, quadratic_cutoff_bound)
    result = kg(env1_1, env2_1, d1, d2, *args)
    assert(np.isclose(reference[0], result[0], atol=tol))
    assert(np.isclose(reference[1], result[1], atol=tol).all())

    reference = en_kernel(env1_1, env2_1, *args, quadratic_cutoff_bound)
    result = en_kernel(env1_1, env2_1, *args)
    assert(np.isclose(reference, result, atol=tol))

    reference = force_en_kernel(env1_1, env2_1, d1, *args, quadratic_cutoff_bound)
    result = force_en_kernel(env1_1, env2_1, d1, *args)
    assert(np.isclose(reference, result, atol=tol))


@pytest.mark.parametrize('kernel_name', ['2+3'])
@pytest.mark.parametrize('diff_cutoff', [True, False])
def test_constraint(kernel_name, diff_cutoff):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    if ('mb' in kernel_name):
        return

    cutoffs = np.array([1, 1])
    delta = 1e-8
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(
        cutoffs, delta)

    # set hyperparameters
    d1 = 1

    # hyps, hm, cut = generate_hm(nbond, ntriplet, cutoffs, constraint)
    cut, hyps, hm = generate_diff_hm(kernel_name, diff_cutoff=diff_cutoff, constraint=True)
    args0 = from_mask_to_args(hyps, hm, cut)

    _, __, en_kernel, force_en_kernel = str_to_kernel_set(kernel_name, True)

    kern_finite_diff = 0

    if ('2' in kernel_name):
        _, __, en2_kernel, fek2 = str_to_kernel_set('2', True)
        calc1 = en2_kernel(env1_2, env2_1, *args0)
        calc2 = en2_kernel(env1_1, env2_1, *args0)
        kern_finite_diff += (calc1 - calc2) / 2.0 / delta

    if ('3' in kernel_name):
        _, __, en3_kernel, fek3 = str_to_kernel_set('3', True)
        calc1 = en3_kernel(env1_2, env2_1, *args0)
        calc2 = en3_kernel(env1_1, env2_1, *args0)
        kern_finite_diff += (calc1 - calc2) / 3.0 / delta

    kern_analytical = force_en_kernel(env1_1, env2_1, d1, *args0)

    tol = 1e-4
    assert(np.isclose(-kern_finite_diff, kern_analytical, atol=tol))


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('diff_cutoff', [True, False])
def test_force_en(kernel_name, diff_cutoff):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    delta = 1e-6
    d1 = 1
    d2 = 2

    cutoffs, hyps, hm = generate_diff_hm(kernel_name, diff_cutoff, False)
    args = from_mask_to_args(hyps, hm, cutoffs)

    print("generate environment")
    env1 = generate_mb_envs(cutoffs, np.eye(3)*100, delta, d1, hm)
    env2 = generate_mb_envs(cutoffs, np.eye(3)*100, delta, d2, hm)

    _, __, en_kernel, force_en_kernel = str_to_kernel_set(kernel_name, True)

    print("analytical")
    kern_analytical = force_en_kernel(env1[0][0], env2[0][0], d1, *args)
    print(kern_analytical)

    kern_finite_diff = 0
    if ('mb' in kernel_name):
        kernel, _, enm_kernel, efk = str_to_kernel_set('mb', True)

        calc = 0
        for i in range(3):
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
    assert (np.isclose(-kern_finite_diff, kern_analytical, atol=tol))


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('constraint', [True, False])
def test_force(kernel_name, constraint):
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    if ('mb' in kernel_name):
        return


    # create env 1
    delta = 1e-5
    d1 = 1
    d2 = 2

    cutoffs, hyps, hm = generate_diff_hm(kernel_name, constraint=constraint)
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(
        cutoffs, delta, hm)
    args0 = from_mask_to_args(hyps, hm, cutoffs)

    kernel, g, en_kernel, fek = str_to_kernel_set(kernel_name, True)

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


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('constraint', [True, False])
def test_hyps_grad(kernel_name, constraint):

    np.random.seed(0)

    delta = 1e-8
    cutoffs, hyps, hm = generate_diff_hm(kernel_name, constraint=constraint)

    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(
        cutoffs, delta, hm)

    args = from_mask_to_args(hyps, hm, cutoffs)
    d1 = 1
    d2 = 2

    kernel, kernel_grad, _, __ = str_to_kernel_set(kernel_name, True)

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




@pytest.mark.parametrize('diff_cutoff', [True, False])
def test_many_force(diff_cutoff):
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    # create env 1
    d1 = 1
    d2 = 2
    delta = 1e-6
    tol = 1e-4
    cutoffs = np.ones(3)*1.2
    cell = 1e7 * np.eye(3)

    cutoffs, hyps, hm = generate_diff_hm('mb', diff_cutoff, constraint=False)

    args = from_mask_to_args(hyps, hm, cutoffs)

    env1 = generate_debug_envs(cutoffs, cell, delta, d1, hm)
    env2 = generate_debug_envs(cutoffs, cell, delta, d2, hm)

    kernel, _, en_kernel, ___ = str_to_kernel_set('mb', True)

    cal = 0
    for i in range(3):
        for j in range(3):
            cal += en_kernel(env1[1][i], env2[1][j], *args)
            cal += en_kernel(env1[2][i], env2[2][j], *args)
            cal -= en_kernel(env1[1][i], env2[2][j], *args)
            cal -= en_kernel(env1[2][i], env2[1][j], *args)
    kern_finite_diff = cal / (4 * delta ** 2)
    kern_analytical = kernel(env1[0][0], env2[0][0],
                             d1, d2, *args)
    print("mb force", kern_finite_diff, kern_analytical)
    assert (np.isclose(kern_finite_diff, kern_analytical, atol=tol))

    # delt = 1e-4
    # for ihyp in range(len(hyps)):
    #     newhyps = np.copy(hyps)
    #     newhyps[ihyp] *= (1+delt)
    #     args = from_mask_to_args(newhyps, hm, cutoffs)
    #     cal = 0
    #     for i in range(3):
    #         for j in range(3):
    #             cal += en_kernel(env1[1][i], env2[1][j], *args)
    #             cal += en_kernel(env1[2][i], env2[2][j], *args)
    #             cal -= en_kernel(env1[1][i], env2[2][j], *args)
    #             cal -= en_kernel(env1[2][i], env2[1][j], *args)
    #     kern_finite_diff = cal / (4 * delta ** 2)
    #     kern_analytical = kernel(env1[0][0], env2[0][0],
    #                              d1, d2, *args)
    #     print("mb force", ihyp, kern_finite_diff, kern_analytical)

def generate_debug_envs(cutoffs, cell, delt, d1, mask=None):

    positions = []
    positions += [[np.array([0., 0., 0.]),
                   np.array([0., 0.3, 0.]),
                   np.array([0.3, 0., 0.]),
                   np.array([1., 1., 0.])]]

    noa = len(positions[0])

    positions_2 = deepcopy(positions[0])
    positions_2[0][d1-1] = delt
    positions += [positions_2]

    positions_3 = deepcopy(positions[0])
    positions_3[0][d1-1] = -delt
    positions += [positions_3]

    species_1 = [1, 2, 1, 1]
    test_struc = []
    for i in range(3):
        test_struc += [struc.Structure(cell, species_1, positions[i])]

    env_0 = []
    env_p = []
    env_m = []
    for i in range(noa):
        env_0 += [env.AtomicEnvironment(test_struc[0], i, cutoffs, mask)]
        env_p += [env.AtomicEnvironment(test_struc[1], i, cutoffs, mask)]
        env_m += [env.AtomicEnvironment(test_struc[2], i, cutoffs, mask)]

    return [env_0, env_p, env_m]

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
        cutoffs = np.array([1.25])

        hyps2 += [1, 0.8, 0.7, 0.05]
        hm2['nbond'] = 2
        hm2['bond_mask'] = np.ones(4, dtype=int)
        hm2['bond_mask'][0] = 0
        if (diff_cutoff):
            hm2['cutoff_2b'] = np.array([1.2, 1.25])

    if ('3' in kernel_name):
        cutoffs = np.array([1.25, 1.2])

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
        cutoffs = np.array([1.25, 1.2, 1.2])

        hyps2 += [1, 0.9, 0.8, 0.05]
        hm2['nmb'] = 2
        hm2['mb_mask'] = np.ones(4, dtype=int)
        if (diff_cutoff):
            hm2['mb_mask'][0] = 0
            hm2['cutoff_mb'] = np.array([1.1, 1.2])

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

import sys
from copy import deepcopy
import pytest
import numpy as np
from numpy.random import random, randint

from flare import env, struc, gp
import flare.kernels.mc_simple as mck
from flare.kernels.utils import str_to_kernel_set as stks

from .fake_gp import generate_envs, generate_mb_envs

list_to_test = ['2mc', '3mc', '2+3mc', 'mbmc', '2+3+mbmc']


def generate_hm(kernel_name):
    hyps = []
    for term in ['2', '3', 'mb']:
        if (term in kernel_name):
            hyps += [random(2)]
    hyps += [random()]
    return np.hstack(hyps)


@pytest.mark.parametrize('kernel_name', list_to_test)
def test_force_en(kernel_name):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    cutoffs = np.ones(3)*1.2
    delta = 1e-5
    cell = 1e7 * np.eye(3)

    # set hyperparameters
    d1 = 1

    env1 = generate_mb_envs(cutoffs, cell, delta, d1)
    env2 = generate_mb_envs(cutoffs, cell, delta, d1)

    hyps = generate_hm(kernel_name)

    _, __, en_kernel, force_en_kernel = stks(kernel_name)

    nterm = 0
    for term in ['2', '3', 'mb']:
        if (term in kernel_name):
            nterm += 1

    kern_finite_diff = 0
    if ('mb' in kernel_name):
        _, __, enm_kernel, ___ = stks('mbmc')
        mhyps = hyps[(nterm-1)*2:]
        calc = 0
        for i in range(3):
            calc += enm_kernel(env1[2][i], env2[0][0], mhyps, cutoffs)
            calc -= enm_kernel(env1[1][i], env2[0][0], mhyps, cutoffs)
        mb_diff = calc / (2 * delta)
        kern_finite_diff += mb_diff

    if ('2' in kernel_name):
        nbond = 1
        _, __, en2_kernel, ___ = stks('2mc')
        calc1 = en2_kernel(env1[2][0], env2[0][0], hyps[0:nbond * 2], cutoffs)
        calc2 = en2_kernel(env1[1][0], env2[0][0], hyps[0:nbond * 2], cutoffs)
        diff2b = (calc1 - calc2) / 2.0 / 2.0 / delta

        kern_finite_diff += diff2b
    else:
        nbond = 0

    if ('3' in kernel_name):
        _, __, en3_kernel, ___ = stks('3mc')
        calc1 = en3_kernel(env1[2][0], env2[0][0], hyps[nbond * 2:], cutoffs)
        calc2 = en3_kernel(env1[1][0], env2[0][0], hyps[nbond * 2:], cutoffs)
        diff3b = (calc1 - calc2) / 2.0 / 3.0 / delta

        kern_finite_diff += diff3b

    kern_analytical = force_en_kernel(env1[0][0], env2[0][0], d1, hyps, cutoffs)

    print("\nforce_en", kernel_name, kern_finite_diff, kern_analytical)

    tol = 1e-4
    assert (np.isclose(kern_finite_diff, kern_analytical, atol=tol))


@pytest.mark.parametrize('kernel_name', list_to_test)
def test_force(kernel_name):
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    if ('mb' in kernel_name):
        return 0
    # create env 1
    delta = 1e-5
    cutoffs = np.ones(3)*0.8

    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 \
        = generate_envs(cutoffs, delta)

    # set hyperparameters
    hyps = generate_hm(kernel_name)
    d1 = 1
    d2 = 2

    kernel, _, en_kernel, ___ = stks(kernel_name)

    # check force kernel
    calc1 = en_kernel(env1_2, env2_2, hyps, cutoffs)
    calc2 = en_kernel(env1_3, env2_3, hyps, cutoffs)
    calc3 = en_kernel(env1_2, env2_3, hyps, cutoffs)
    calc4 = en_kernel(env1_3, env2_2, hyps, cutoffs)

    kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4 * delta ** 2)
    kern_analytical = kernel(env1_1, env2_1,
                             d1, d2, hyps, cutoffs)
    tol = 1e-4
    assert (np.isclose(kern_finite_diff, kern_analytical, atol=tol))

def test_many_force():
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    # create env 1
    delta = 1e-5
    cutoffs = np.ones(3)*1.2

    cell = 1e7 * np.eye(3)
    d1 = 1
    d2 = 2
    env1 = generate_mb_envs(cutoffs, cell, delta, d1)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2)

    # set hyperparameters
    hyps = generate_hm('mb')

    kernel, _, en_kernel, ___ = stks('mb')

    cal = 0
    for i in range(3):
        for j in range(3):
            cal += en_kernel(env1[1][i], env2[1][j], hyps, cutoffs)
            cal += en_kernel(env1[2][i], env2[2][j], hyps, cutoffs)
            cal -= en_kernel(env1[1][i], env2[2][j], hyps, cutoffs)
            cal -= en_kernel(env1[2][i], env2[1][j], hyps, cutoffs)
    kern_finite_diff = cal / (4 * delta ** 2)
    kern_analytical = kernel(env1[0][0], env2[0][0],
                             d1, d2, hyps, cutoffs)
    print("mb force", kern_finite_diff, kern_analytical)
    tol = 1e-4
    assert (np.isclose(kern_finite_diff, kern_analytical, atol=tol))


@pytest.mark.parametrize('kernel_name', list_to_test)
def test_hyps_grad(kernel_name):

    delta = 1e-8
    cutoffs = np.ones(3)*1.2

    hyps = generate_hm(kernel_name)
    d1 = randint(1, 3)
    d2 = randint(1, 3)
    print("hyps", hyps)
    cell = 1e7 * np.eye(3)

    env1 = generate_mb_envs(cutoffs, cell, 0, d1)[0][0]
    env2 = generate_mb_envs(cutoffs, cell, 0, d2)[0][0]

    kernel, kernel_grad, _, _ = stks(kernel_name, False)

    grad_test = kernel_grad(env1, env2,
                            d1, d2, hyps, cutoffs)

    tol = 1e-4
    original = kernel(env1, env2, d1, d2,
                      hyps, cutoffs)
    for i in range(len(hyps)-1):
        newhyps = np.copy(hyps)
        newhyps[i] += delta
        hgrad = (kernel(env1, env2, d1, d2, newhyps,
                        cutoffs) -
                 original)/delta
        print("numerical gradients", hgrad)
        print("analytical gradients", grad_test[1][i])
        assert(np.isclose(grad_test[1][i], hgrad, atol=tol))

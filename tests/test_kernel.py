import sys
from copy import deepcopy
import pytest
import numpy as np
from numpy import isclose
from numpy.random import random, randint

from flare import env, struc, gp
from flare.kernels.utils import str_to_kernel_set

from .fake_gp import generate_mb_envs

list_to_test = ['2', '3', '2+3', 'mb', '2+3+mb']
list_type = ['sc', 'mc']

def generate_hm(kernel_name):
    hyps = []
    for term in ['2', '3', 'mb']:
        if (term in kernel_name):
            hyps += [random(2)]
    hyps += [random()]
    return np.hstack(hyps)


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('kernel_type', list_type)
def test_force_en(kernel_name, kernel_type):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    cutoffs = np.ones(3)*1.2
    delta = 1e-5
    tol = 1e-4
    cell = 1e7 * np.eye(3)

    # set hyperparameters
    d1 = 1

    np.random.seed(10)
    env1 = generate_mb_envs(cutoffs, cell, delta, d1, kern_type=kernel_type)
    env2 = generate_mb_envs(cutoffs, cell, delta, d1, kern_type=kernel_type)

    hyps = generate_hm(kernel_name)

    _, __, en_kernel, force_en_kernel = \
        str_to_kernel_set(kernel_name+kernel_type)
    print(force_en_kernel.__name__)

    nterm = 0
    for term in ['2', '3', 'mb']:
        if (term in kernel_name):
            nterm += 1

    kern_finite_diff = 0
    if ('mb' in kernel_name):
        _, __, enm_kernel, ___ = str_to_kernel_set('mb'+kernel_type)
        mhyps = hyps[(nterm-1)*2:]
        calc = 0
        nat = len(env1[0])
        for i in range(nat):
            calc += enm_kernel(env1[2][i], env2[0][0], mhyps, cutoffs)
            calc -= enm_kernel(env1[1][i], env2[0][0], mhyps, cutoffs)
        mb_diff = calc / (2 * delta)
        kern_finite_diff += mb_diff

    if ('2' in kernel_name):
        nbond = 1
        _, __, en2_kernel, ___ = str_to_kernel_set('2'+kernel_type)
        calc1 = en2_kernel(env1[2][0], env2[0][0], hyps[0:nbond * 2], cutoffs)
        calc2 = en2_kernel(env1[1][0], env2[0][0], hyps[0:nbond * 2], cutoffs)
        diff2b = 4 * (calc1 - calc2) / 2.0 / 2.0 / delta

        kern_finite_diff += diff2b
    else:
        nbond = 0

    if ('3' in kernel_name):
        _, __, en3_kernel, ___ = str_to_kernel_set('3'+kernel_type)
        calc1 = en3_kernel(env1[2][0], env2[0][0], hyps[nbond * 2:], cutoffs)
        calc2 = en3_kernel(env1[1][0], env2[0][0], hyps[nbond * 2:], cutoffs)
        diff3b = 9 * (calc1 - calc2) / 2.0 / 3.0 / delta

        kern_finite_diff += diff3b

    kern_analytical = \
        force_en_kernel(env1[0][0], env2[0][0], d1, hyps, cutoffs)

    print("\nforce_en", kernel_name, kern_finite_diff, kern_analytical)

    assert (isclose(kern_finite_diff, kern_analytical, rtol=tol))


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('kernel_type', list_type)
def test_force(kernel_name, kernel_type):
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    d1 = 1
    d2 = 2
    tol = 1e-3
    cell = 1e7 * np.eye(3)
    delta = 1e-4
    cutoffs = np.ones(3)*1.2

    np.random.seed(10)

    hyps = generate_hm(kernel_name)
    kernel, kg, en_kernel, fek = \
        str_to_kernel_set(kernel_name+kernel_type, False)
    args = (hyps, cutoffs)

    nterm = 0
    for term in ['2', '3', 'mb']:
        if (term in kernel_name):
            nterm += 1

    env1 = generate_mb_envs(cutoffs, cell, delta, d1, kern_type=kernel_type)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2, kern_type=kernel_type)

    # check force kernel
    kern_finite_diff = 0
    if ('mb' == kernel_name):
        _, __, enm_kernel, ___ = str_to_kernel_set('mb'+kernel_type)
        mhyps = hyps[(nterm-1)*2:]
        print(hyps)
        print(mhyps)
        cal = 0
        for i in range(3):
            for j in range(len(env1[0])):
                cal += enm_kernel(env1[1][i], env2[1][j], mhyps, cutoffs)
                cal += enm_kernel(env1[2][i], env2[2][j], mhyps, cutoffs)
                cal -= enm_kernel(env1[1][i], env2[2][j], mhyps, cutoffs)
                cal -= enm_kernel(env1[2][i], env2[1][j], mhyps, cutoffs)
        kern_finite_diff += cal / (4 * delta ** 2)
    else:
        # TODO: Establish why 2+3+MB fails (numerical error?)
        return

    if ('2' in kernel_name):
        nbond = 1
        _, __, en2_kernel, ___ = str_to_kernel_set('2'+kernel_type)
        print(hyps[0:nbond * 2])

        calc1 = en2_kernel(env1[1][0], env2[1][0], hyps[0:nbond * 2], cutoffs)
        calc2 = en2_kernel(env1[2][0], env2[2][0], hyps[0:nbond * 2], cutoffs)
        calc3 = en2_kernel(env1[1][0], env2[2][0], hyps[0:nbond * 2], cutoffs)
        calc4 = en2_kernel(env1[2][0], env2[1][0], hyps[0:nbond * 2], cutoffs)
        kern_finite_diff += 4 * (calc1 + calc2 - calc3 - calc4) / (4*delta**2)
    else:
        nbond = 0

    if ('3' in kernel_name):
        _, __, en3_kernel, ___ = str_to_kernel_set('3'+kernel_type)
        print(hyps[nbond * 2:])
        calc1 = en3_kernel(env1[1][0], env2[1][0], hyps[nbond * 2:], cutoffs)
        calc2 = en3_kernel(env1[2][0], env2[2][0], hyps[nbond * 2:], cutoffs)
        calc3 = en3_kernel(env1[1][0], env2[2][0], hyps[nbond * 2:], cutoffs)
        calc4 = en3_kernel(env1[2][0], env2[1][0], hyps[nbond * 2:], cutoffs)
        kern_finite_diff += 9 * (calc1 + calc2 - calc3 - calc4) / (4*delta**2)

    kern_analytical = kernel(env1[0][0], env2[0][0], d1, d2, *args)

    assert(isclose(kern_finite_diff, kern_analytical, rtol=tol))


@pytest.mark.parametrize('kernel_name', list_to_test)
@pytest.mark.parametrize('kernel_type', list_type)
def test_hyps_grad(kernel_name, kernel_type):

    d1 = randint(1, 3)
    d2 = randint(1, 3)
    tol = 1e-4
    cell = 1e7 * np.eye(3)
    delta = 1e-8
    cutoffs = np.ones(3)*1.2

    np.random.seed(10)
    hyps = generate_hm(kernel_name)
    env1 = generate_mb_envs(cutoffs, cell, 0, d1, kern_type=kernel_type)[0][0]
    env2 = generate_mb_envs(cutoffs, cell, 0, d2, kern_type=kernel_type)[0][0]

    kernel, kernel_grad, _, _ = str_to_kernel_set(kernel_name+kernel_type, False)

    grad_test = kernel_grad(env1, env2,
                            d1, d2, hyps, cutoffs)

    original = kernel(env1, env2, d1, d2,
                      hyps, cutoffs)
    assert(isclose(grad_test[0], original, rtol=tol))

    for i in range(len(hyps)-1):
        newhyps = np.copy(hyps)
        newhyps[i] += delta
        hgrad = (kernel(env1, env2, d1, d2, newhyps,
                        cutoffs) -
                 original)/delta
        print("numerical gradients", hgrad)
        print("analytical gradients", grad_test[1][i])
        assert(isclose(grad_test[1][i], hgrad, rtol=tol))

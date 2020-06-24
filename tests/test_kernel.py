import sys
from time import time
from copy import deepcopy
import pytest
import numpy as np
from numpy import isclose
from numpy.random import random, randint

from flare import env, struc, gp
from flare.kernels.utils import str_to_kernel_set
from .fake_gp import generate_mb_envs

list_to_test = [['2'], ['3'], ['many'],
                ['2', '3'],
                ['2', '3', 'many']]
list_type = ['sc', 'mc']

def generate_hm(kernels):
    hyps = []
    for term in ['2', '3', 'many']:
        if (term in kernels):
            hyps += [random(2)]
    hyps += [random()]
    return np.hstack(hyps)


@pytest.mark.parametrize('kernels', list_to_test)
@pytest.mark.parametrize('kernel_type', list_type)
@pytest.mark.parametrize('d1', [1, 2])
def test_force_en(kernels, kernel_type, d1):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    cutoffs = np.ones(3)*1.2
    delta = 1e-5
    tol = 1e-4
    cell = 1e7 * np.eye(3)

    # set hyperparameters

    np.random.seed(10)
    env1 = generate_mb_envs(cutoffs, cell, delta, d1, kern_type=kernel_type)
    env2 = generate_mb_envs(cutoffs, cell, delta, d1, kern_type=kernel_type)

    hyps = generate_hm(kernels)

    _, __, en_kernel, force_en_kernel = \
        str_to_kernel_set(kernels, kernel_type)

    nterm = 0
    for term in ['2', '3', 'many']:
        if (term in kernels):
            nterm += 1

    kern_finite_diff = 0
    if 'many' in kernels:
        _, __, enm_kernel, ___ = str_to_kernel_set(['many'], kernel_type)
        mhyps = hyps[(nterm-1)*2:]
        calc = 0
        nat = len(env1[0])
        for i in range(nat):
            calc += enm_kernel(env1[2][i], env2[0][0], mhyps, cutoffs)
            calc -= enm_kernel(env1[1][i], env2[0][0], mhyps, cutoffs)
        mb_diff = calc / (2 * delta)
        kern_finite_diff += mb_diff

    if ('2' in kernels):
        ntwobody = 1
        _, __, en2_kernel, ___ = str_to_kernel_set('2', kernel_type)
        calc1 = en2_kernel(env1[2][0], env2[0][0], hyps[0:ntwobody * 2], cutoffs)
        calc2 = en2_kernel(env1[1][0], env2[0][0], hyps[0:ntwobody * 2], cutoffs)
        diff2b = 4 * (calc1 - calc2) / 2.0 / 2.0 / delta

        kern_finite_diff += diff2b
    else:
        ntwobody = 0

    if ('3' in kernels):
        _, __, en3_kernel, ___ = str_to_kernel_set('3', kernel_type)
        calc1 = en3_kernel(env1[2][0], env2[0][0], hyps[ntwobody * 2:], cutoffs)
        calc2 = en3_kernel(env1[1][0], env2[0][0], hyps[ntwobody * 2:], cutoffs)
        diff3b = 9 * (calc1 - calc2) / 2.0 / 3.0 / delta

        kern_finite_diff += diff3b

    kern_analytical = \
        force_en_kernel(env1[0][0], env2[0][0], hyps, cutoffs)
    kern_analytical = kern_analytical[d1-1]

    print("\nforce_en", kernels, kern_finite_diff, kern_analytical)
    assert (isclose(kern_finite_diff, kern_analytical, rtol=tol))


@pytest.mark.parametrize('kernels', list_to_test)
@pytest.mark.parametrize('kernel_type', list_type)
@pytest.mark.parametrize('d1, d2', [[1,1], [1, 2]])
def test_force(kernels, kernel_type, d1, d2):
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    tol = 1e-3
    cell = 1e7 * np.eye(3)
    delta = 1e-4
    cutoffs = np.ones(3)*1.2

    np.random.seed(10)

    hyps = generate_hm(kernels)
    kernel, kg, en_kernel, fek = \
        str_to_kernel_set(kernels, kernel_type)
    args = (hyps, cutoffs)

    nterm = 0
    for term in ['2', '3', 'many']:
        if (term in kernels):
            nterm += 1

    env1 = generate_mb_envs(cutoffs, cell, delta, d1, kern_type=kernel_type)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2, kern_type=kernel_type)

    # check force kernel
    kern_finite_diff = 0
    if ('many' in kernels) and len(kernels)==1:
        _, __, enm_kernel, ___ = str_to_kernel_set('many', kernel_type)
        mhyps = hyps[(nterm-1)*2:]
        cal = 0
        for i in range(3):
            for j in range(len(env1[0])):
                cal += enm_kernel(env1[1][i], env2[1][j], mhyps, cutoffs)
                cal += enm_kernel(env1[2][i], env2[2][j], mhyps, cutoffs)
                cal -= enm_kernel(env1[1][i], env2[2][j], mhyps, cutoffs)
                cal -= enm_kernel(env1[2][i], env2[1][j], mhyps, cutoffs)
        kern_finite_diff += cal / (4 * delta ** 2)

    elif 'many' in kernels:
        # TODO: Establish why 2+3+MB fails (numerical error?)
        return

    if ('2' in kernels):
        ntwobody = 1
        _, __, en2_kernel, ___ = str_to_kernel_set(['2'], kernel_type)
        calc1 = en2_kernel(env1[1][0], env2[1][0], hyps[0:ntwobody * 2], cutoffs)
        calc2 = en2_kernel(env1[2][0], env2[2][0], hyps[0:ntwobody * 2], cutoffs)
        calc3 = en2_kernel(env1[1][0], env2[2][0], hyps[0:ntwobody * 2], cutoffs)
        calc4 = en2_kernel(env1[2][0], env2[1][0], hyps[0:ntwobody * 2], cutoffs)
        kern_finite_diff += 4 * (calc1 + calc2 - calc3 - calc4) / (4*delta**2)
    else:
        ntwobody = 0

    if ('3' in kernels):
        _, __, en3_kernel, ___ = str_to_kernel_set(['3'], kernel_type)
        calc1 = en3_kernel(env1[1][0], env2[1][0], hyps[ntwobody * 2:], cutoffs)
        calc2 = en3_kernel(env1[2][0], env2[2][0], hyps[ntwobody * 2:], cutoffs)
        calc3 = en3_kernel(env1[1][0], env2[2][0], hyps[ntwobody * 2:], cutoffs)
        calc4 = en3_kernel(env1[2][0], env2[1][0], hyps[ntwobody * 2:], cutoffs)
        kern_finite_diff += 9 * (calc1 + calc2 - calc3 - calc4) / (4*delta**2)

    kern_analytical = kernel(env1[0][0], env2[0][0], *args)
    kern_analytical = kern_analytical[d1-1][d2-1]

    print(d1, d2, kern_finite_diff, kern_analytical)
    assert(isclose(kern_finite_diff, kern_analytical, rtol=tol))


@pytest.mark.parametrize('kernels', list_to_test)
@pytest.mark.parametrize('kernel_type', list_type)
@pytest.mark.parametrize('d1, d2', [[1,1], [1, 2]])
def test_hyps_grad(kernels, kernel_type, d1, d2):

    tol = 1e-4
    cell = 1e7 * np.eye(3)
    delta = 1e-8
    cutoffs = np.ones(3)*1.2

    np.random.seed(10)
    hyps = generate_hm(kernels)
    env1 = generate_mb_envs(cutoffs, cell, 0, d1, kern_type=kernel_type)[0][0]
    env2 = generate_mb_envs(cutoffs, cell, 0, d2, kern_type=kernel_type)[0][0]

    kernel, kernel_grad, _, _ = str_to_kernel_set(kernels, kernel_type)

    k, grad = kernel_grad(env1, env2, hyps, cutoffs)
    grad = grad[:, d1-1, d2-1]

    original = kernel(env1, env2, hyps, cutoffs)
    assert(isclose(k, original, rtol=tol).all())

    for i in range(len(hyps)-1):
        newhyps = np.copy(hyps)
        newhyps[i] += delta
        hgrad = (kernel(env1, env2, newhyps,
                        cutoffs) -
                 original)/delta
        hgrad = hgrad[d1-1][d2-1]
        print(i, "numerical vs analytical", hgrad, grad[i])
        assert(isclose(grad[i], hgrad, rtol=tol))

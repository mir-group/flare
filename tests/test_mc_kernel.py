import sys
from copy import deepcopy
import pytest
import numpy as np
from numpy.random import random, randint

from flare import env, struc, gp
from flare.kernels.utils import str_to_kernel_set as stks

from .fake_gp import generate_envs

def generate_hm(nbond, ntriplet):

    if bool(nbond>0)!=bool(ntriplet>0):
        return np.array([random(), random()])
    else:
        return np.array([random(), random(), random(), random()])


@pytest.mark.parametrize('kernel_name, nbond, ntriplet',
                         [ ('2mc', 1, 0),
                           ('3mc', 0, 1),
                           ('2+3mc', 1, 1) ]
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

    _, __, en_kernel, force_en_kernel = stks(kernel_name)
    if bool('2' in kernel_name) != bool('3' in kernel_name):

        # check force kernel
        calc1 = en_kernel(env1_2, env2_1, hyps, cutoffs)
        calc2 = en_kernel(env1_1, env2_1, hyps, cutoffs)

        kern_finite_diff = (calc1 - calc2) / delta
        if ('2' in kernel_name):
            kern_finite_diff /= 2
        else:
            kern_finite_diff /= 3
    else:
        _, __, en2_kernel, ___ = stks('2mc')
        _, __, en3_kernel, ___ = stks('3mc')
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
                         [ ('2_mc', 1, 0),
                           ('3_mc', 0, 1),
                           ('2+3_mc', 1, 1) ]
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

    kernel, _, __, ___ = stks(kernel_name)
    if bool('2' in kernel_name) != bool('3' in kernel_name):
        _, __, en_kernel, ___ = stks(kernel_name)
    else:
        _, __, en_kernel, ___ = stks('2+3_mc')

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
                         [ ('mc2', 1, 0),
                           ('mc3', 0, 1),
                           ('mc23', 1, 1) ]
                         )
def test_hyps_grad(kernel_name, nbond, ntriplet):

    delta = 1e-8
    cutoffs = np.array([1, 1])
    env1_1, env1_2, env1_3, env2_1, env2_2, env2_3 = generate_envs(cutoffs, delta)

    hyps = generate_hm(nbond, ntriplet)
    d1 = randint(1, 3)
    d2 = randint(1, 3)

    kernel, kernel_grad, _, _ = stks(kernel_name, False)

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
        print(grad_test, hgrad)
        assert(np.isclose(grad_test[1][i], hgrad, atol=tol))

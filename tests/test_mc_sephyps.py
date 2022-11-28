import sys
from copy import deepcopy

import pytest
import numpy as np
from numpy.random import random, randint
from numpy import isclose

from flare.kernels.utils import from_mask_to_args, str_to_kernel_set
from flare.kernels.cutoffs import quadratic_cutoff_bound
from flare.utils.parameters import Parameters
from flare.utils.parameter_helper import ParameterHelper

from .fake_gp import generate_mb_envs, generate_mb_twin_envs

list_to_test = [
    ["twobody"],
    ["threebody"],
    ["twobody", "threebody"],
    ["twobody", "threebody", "manybody"],
]
multi_cut = [False, True]


@pytest.mark.parametrize("kernels", list_to_test)
@pytest.mark.parametrize("multi_cutoff", multi_cut)
def test_force_en_multi_vs_simple(kernels, multi_cutoff):
    """Check that the analytical kernel matches the one implemented
    in mc_simple.py"""

    d1 = 1
    d2 = 2
    tol = 1e-4
    cell = 1e7 * np.eye(3)

    # set hyperparameters
    cutoffs, hyps1, hyps2, hm1, hm2 = generate_same_hm(kernels, multi_cutoff)

    delta = 1e-8
    env1 = generate_mb_envs(cutoffs, cell, delta, d1)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2)
    env1 = env1[0][0]
    env2 = env2[0][0]

    # mc_simple
    kernel0, kg0, en_kernel0, force_en_kernel0, _, _, _ = str_to_kernel_set(
        kernels, "mc", None
    )
    args0 = from_mask_to_args(hyps1, cutoffs)

    # mc_sephyps
    # args1 and args 2 use 1 and 2 groups of hyper-parameters
    # if (diff_cutoff), 1 or 2 group of cutoffs
    # but same value as in args0
    kernel, kg, en_kernel, force_en_kernel, _, _, _ = str_to_kernel_set(
        kernels, "mc", hm2
    )
    args1 = from_mask_to_args(hyps1, cutoffs, hm1)
    args2 = from_mask_to_args(hyps2, cutoffs, hm2)

    funcs = [
        [kernel0, kg0, en_kernel0, force_en_kernel0],
        [kernel, kg, en_kernel, force_en_kernel],
    ]

    # compare whether mc_sephyps and mc_simple
    # yield the same values
    i = 2
    reference = funcs[0][i](env1, env2, *args0)
    result = funcs[1][i](env1, env2, *args1)
    print(kernels, i, reference, result)
    assert isclose(reference, result, rtol=tol)
    result = funcs[1][i](env1, env2, *args2)
    print(kernels, i, reference, result)
    assert isclose(reference, result, rtol=tol)

    i = 3
    reference = funcs[0][i](env1, env2, d1, *args0)
    result = funcs[1][i](env1, env2, d1, *args1)
    print(kernels, i, reference, result)
    assert isclose(reference, result, rtol=tol)
    result = funcs[1][i](env1, env2, d1, *args2)
    print(kernels, i, reference, result)
    assert isclose(reference, result, rtol=tol)

    i = 0
    reference = funcs[0][i](env1, env2, d1, d2, *args0)
    result = funcs[1][i](env1, env2, d1, d2, *args1)
    assert isclose(reference, result, rtol=tol)
    print(kernels, i, reference, result)
    result = funcs[1][i](env1, env2, d1, d2, *args2)
    assert isclose(reference, result, rtol=tol)
    print(kernels, i, reference, result)

    i = 1
    reference = funcs[0][i](env1, env2, d1, d2, *args0)
    result = funcs[1][i](env1, env2, d1, d2, *args1)
    print(kernels, i, reference, result)
    assert isclose(reference[0], result[0], rtol=tol)
    assert isclose(reference[1], result[1], rtol=tol).all()

    result = funcs[1][i](env1, env2, d1, d2, *args2)
    print(kernels, i, reference, result)
    assert isclose(reference[0], result[0], rtol=tol)
    joint_grad = np.zeros(len(result[1]) // 2)
    for i in range(joint_grad.shape[0]):
        joint_grad[i] = result[1][i * 2] + result[1][i * 2 + 1]
    assert isclose(reference[1], joint_grad, rtol=tol).all()


@pytest.mark.parametrize("kernels", list_to_test)
@pytest.mark.parametrize("diff_cutoff", multi_cut)
def test_check_sig_scale(kernels, diff_cutoff):
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

    cutoffs, hyps0, hm = generate_diff_hm(kernels, diff_cutoff)

    delta = 1e-8
    env1, env1_t = generate_mb_twin_envs(cutoffs, np.eye(3) * 100, delta, d1, hm)
    env2, env2_t = generate_mb_twin_envs(cutoffs, np.eye(3) * 100, delta, d2, hm)
    env1 = env1[0][0]
    env2 = env2[0][0]
    env1_t = env1_t[0][0]
    env2_t = env2_t[0][0]

    # make the second sigma zero
    hyps1 = np.copy(hyps0)
    hyps0[0::4] = 0  # 1e-8
    hyps1[0::4] = 0  # 1e-8
    hyps1[1::4] *= scale

    kernel, kg, en_kernel, force_en_kernel, _, _, _ = str_to_kernel_set(
        kernels, "mc", hm
    )

    args0 = from_mask_to_args(hyps0, cutoffs, hm)
    args1 = from_mask_to_args(hyps1, cutoffs, hm)

    reference = en_kernel(env1, env2, *args0)
    result = en_kernel(env1_t, env2_t, *args1)
    print(en_kernel.__name__, result, reference)
    if reference != 0:
        assert isclose(result / reference, scale**2, rtol=tol)

    reference = force_en_kernel(env1, env2, d1, *args0)
    result = force_en_kernel(env1_t, env2_t, d1, *args1)
    print(force_en_kernel.__name__, result, reference)
    if reference != 0:
        assert isclose(result / reference, scale**2, rtol=tol)

    reference = kernel(env1, env2, d1, d2, *args0)
    result = kernel(env1_t, env2_t, d1, d2, *args1)
    print(kernel.__name__, result, reference)
    if reference != 0:
        assert isclose(result / reference, scale**2, rtol=tol)

    reference = kg(env1, env2, d1, d2, *args0)
    result = kg(env1_t, env2_t, d1, d2, *args1)
    print(kg.__name__, result, reference)
    if reference[0] != 0:
        assert isclose(result[0] / reference[0], scale**2, rtol=tol)
    for idx in range(reference[1].shape[0]):
        # check sig0
        if reference[1][idx] != 0 and (idx % 4) == 0:
            assert isclose(result[1][idx] / reference[1][idx], scale, rtol=tol)
        # check the rest, but skip sig 1
        elif reference[1][idx] != 0 and (idx % 4) != 1:
            assert isclose(result[1][idx] / reference[1][idx], scale**2, rtol=tol)


@pytest.mark.parametrize("kernels", list_to_test)
@pytest.mark.parametrize("diff_cutoff", multi_cut)
def test_force_bound_cutoff_compare(kernels, diff_cutoff):
    """Check that the analytical kernel matches the one implemented
    in mc_simple.py"""

    d1 = 1
    d2 = 2
    tol = 1e-4
    cell = 1e7 * np.eye(3)
    delta = 1e-8

    cutoffs, hyps, hm = generate_diff_hm(kernels, diff_cutoff)
    kernel, kg, en_kernel, force_en_kernel, _, _, _ = str_to_kernel_set(
        kernels, "mc", hm
    )
    args = from_mask_to_args(hyps, cutoffs, hm)

    np.random.seed(10)
    env1 = generate_mb_envs(cutoffs, cell, delta, d1, hm)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2, hm)
    env1 = env1[0][0]
    env2 = env2[0][0]

    reference = kernel(env1, env2, d1, d2, *args, quadratic_cutoff_bound)
    result = kernel(env1, env2, d1, d2, *args)
    assert isclose(reference, result, rtol=tol)

    reference = kg(env1, env2, d1, d2, *args, quadratic_cutoff_bound)
    result = kg(env1, env2, d1, d2, *args)
    assert isclose(reference[0], result[0], rtol=tol)
    assert isclose(reference[1], result[1], rtol=tol).all()

    reference = en_kernel(env1, env2, *args, quadratic_cutoff_bound)
    result = en_kernel(env1, env2, *args)
    assert isclose(reference, result, rtol=tol)

    reference = force_en_kernel(env1, env2, d1, *args, quadratic_cutoff_bound)
    result = force_en_kernel(env1, env2, d1, *args)
    assert isclose(reference, result, rtol=tol)


@pytest.mark.parametrize("kernels", [["twobody", "threebody"]])
@pytest.mark.parametrize("diff_cutoff", multi_cut)
def test_constraint(kernels, diff_cutoff):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    if "manybody" in kernels:
        return

    d1 = 1
    d2 = 2
    cell = 1e7 * np.eye(3)
    delta = 1e-8

    cutoffs, hyps, hm = generate_diff_hm(
        kernels, diff_cutoff=diff_cutoff, constraint=True
    )

    _, __, en_kernel, force_en_kernel, _, _, _ = str_to_kernel_set(kernels, "mc", hm)

    args0 = from_mask_to_args(hyps, cutoffs, hm)

    np.random.seed(10)
    env1 = generate_mb_envs(cutoffs, cell, delta, d1, hm)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2, hm)

    kern_finite_diff = 0

    if "twobody" in kernels:
        _, _, en2_kernel, fek2, _, _, _ = str_to_kernel_set(["twobody"], "mc", hm)
        calc1 = en2_kernel(env1[1][0], env2[0][0], *args0)
        calc2 = en2_kernel(env1[0][0], env2[0][0], *args0)
        kern_finite_diff += 4 * (calc1 - calc2) / 2.0 / delta

    if "threebody" in kernels:
        _, _, en3_kernel, fek3, _, _, _ = str_to_kernel_set(["threebody"], "mc", hm)
        calc1 = en3_kernel(env1[1][0], env2[0][0], *args0)
        calc2 = en3_kernel(env1[0][0], env2[0][0], *args0)
        kern_finite_diff += 9 * (calc1 - calc2) / 3.0 / delta

    kern_analytical = force_en_kernel(env1[0][0], env2[0][0], d1, *args0)

    tol = 1e-4
    print(kern_finite_diff, kern_analytical)
    assert isclose(-kern_finite_diff, kern_analytical, rtol=tol)


@pytest.mark.parametrize("kernels", list_to_test)
@pytest.mark.parametrize("diff_cutoff", multi_cut)
def test_force_en(kernels, diff_cutoff):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    delta = 1e-5
    d1 = 1
    d2 = 2
    cell = 1e7 * np.eye(3)
    np.random.seed(0)

    cutoffs, hyps, hm = generate_diff_hm(kernels, diff_cutoff)
    args = from_mask_to_args(hyps, cutoffs, hm)

    env1 = generate_mb_envs(cutoffs, cell, delta, d1, hm)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2, hm)

    _, _, en_kernel, force_en_kernel, _, _, _ = str_to_kernel_set(kernels, "mc", hm)

    kern_analytical = force_en_kernel(env1[0][0], env2[0][0], d1, *args)

    kern_finite_diff = 0
    if "manybody" in kernels:
        kernel, _, enm_kernel, efk, _, _, _ = str_to_kernel_set(["manybody"], "mc", hm)

        calc = 0
        for i in range(len(env1[0])):
            calc += enm_kernel(env1[1][i], env2[0][0], *args)
            calc -= enm_kernel(env1[2][i], env2[0][0], *args)

        kern_finite_diff += (calc) / (2 * delta)

    if "twobody" in kernels or "threebody" in kernels:
        args23 = from_mask_to_args(hyps, cutoffs, hm)

    if "twobody" in kernels:
        kernel, _, en2_kernel, efk, _, _, _ = str_to_kernel_set(["2b"], "mc", hm)
        calc1 = en2_kernel(env1[1][0], env2[0][0], *args23)
        calc2 = en2_kernel(env1[2][0], env2[0][0], *args23)
        diff2b = 4 * (calc1 - calc2) / 2.0 / delta / 2.0

        kern_finite_diff += diff2b

    if "threebody" in kernels:
        kernel, _, en3_kernel, efk, _, _, _ = str_to_kernel_set(["3b"], "mc", hm)
        calc1 = en3_kernel(env1[1][0], env2[0][0], *args23)
        calc2 = en3_kernel(env1[2][0], env2[0][0], *args23)
        diff3b = 9 * (calc1 - calc2) / 3.0 / delta / 2.0

        kern_finite_diff += diff3b

    tol = 1e-3
    print("\nforce_en", kernels, kern_finite_diff, kern_analytical)
    assert isclose(-kern_finite_diff, kern_analytical, rtol=tol)


@pytest.mark.parametrize("kernels", list_to_test)
@pytest.mark.parametrize("diff_cutoff", multi_cut)
def test_force(kernels, diff_cutoff):
    """Check that the analytical force kernel matches finite difference of
    energy kernel."""

    d1 = 1
    d2 = 2
    tol = 1e-3
    cell = 1e7 * np.eye(3)
    delta = 1e-4
    cutoffs = np.ones(3) * 1.2

    np.random.seed(10)

    cutoffs, hyps, hm = generate_diff_hm(kernels, diff_cutoff)
    kernel, kg, en_kernel, fek, _, _, _ = str_to_kernel_set(kernels, "mc", hm)

    nterm = 0
    for term in ["twobody", "threebody", "manybody"]:
        if term in kernels:
            nterm += 1

    np.random.seed(10)
    env1 = generate_mb_envs(cutoffs, cell, delta, d1, hm)
    env2 = generate_mb_envs(cutoffs, cell, delta, d2, hm)

    # check force kernel
    kern_finite_diff = 0
    if "manybody" in kernels and len(kernels) == 1:
        _, _, enm_kernel, _, _, _, _ = str_to_kernel_set(["manybody"], "mc", hm)
        mhyps, mcutoffs, mhyps_mask = Parameters.get_component_mask(
            hm, "manybody", hyps=hyps
        )
        margs = from_mask_to_args(mhyps, mcutoffs, mhyps_mask)
        cal = 0
        for i in range(3):
            for j in range(len(env1[0])):
                cal += enm_kernel(env1[1][i], env2[1][j], *margs)
                cal += enm_kernel(env1[2][i], env2[2][j], *margs)
                cal -= enm_kernel(env1[1][i], env2[2][j], *margs)
                cal -= enm_kernel(env1[2][i], env2[1][j], *margs)
        kern_finite_diff += cal / (4 * delta**2)
    elif "manybody" in kernels:
        # TODO: Establish why 2+3+MB fails (numerical error?)
        return

    if "twobody" in kernels:
        ntwobody = 1
        _, _, en2_kernel, _, _, _, _ = str_to_kernel_set(["twobody"], "mc", hm)
        bhyps, bcutoffs, bhyps_mask = Parameters.get_component_mask(
            hm, "twobody", hyps=hyps
        )
        args2 = from_mask_to_args(bhyps, bcutoffs, bhyps_mask)

        calc1 = en2_kernel(env1[1][0], env2[1][0], *args2)
        calc2 = en2_kernel(env1[2][0], env2[2][0], *args2)
        calc3 = en2_kernel(env1[1][0], env2[2][0], *args2)
        calc4 = en2_kernel(env1[2][0], env2[1][0], *args2)
        kern_finite_diff += 4 * (calc1 + calc2 - calc3 - calc4) / (4 * delta**2)
    else:
        ntwobody = 0

    if "threebody" in kernels:
        _, _, en3_kernel, _, _, _, _ = str_to_kernel_set(["threebody"], "mc", hm)

        thyps, tcutoffs, thyps_mask = Parameters.get_component_mask(
            hm, "threebody", hyps=hyps
        )
        args3 = from_mask_to_args(thyps, tcutoffs, thyps_mask)

        calc1 = en3_kernel(env1[1][0], env2[1][0], *args3)
        calc2 = en3_kernel(env1[2][0], env2[2][0], *args3)
        calc3 = en3_kernel(env1[1][0], env2[2][0], *args3)
        calc4 = en3_kernel(env1[2][0], env2[1][0], *args3)
        kern_finite_diff += 9 * (calc1 + calc2 - calc3 - calc4) / (4 * delta**2)

    args = from_mask_to_args(hyps, cutoffs, hm)
    kern_analytical = kernel(env1[0][0], env2[0][0], d1, d2, *args)

    assert isclose(kern_finite_diff, kern_analytical, rtol=tol)


@pytest.mark.parametrize("kernels", list_to_test)
@pytest.mark.parametrize("diff_cutoff", multi_cut)
@pytest.mark.parametrize("constraint", [True, False])
def test_hyps_grad(kernels, diff_cutoff, constraint):

    delta = 1e-8
    d1 = 1
    d2 = 2
    tol = 1e-4

    np.random.seed(10)
    cutoffs, hyps, hm = generate_diff_hm(kernels, diff_cutoff, constraint=constraint)
    args = from_mask_to_args(hyps, cutoffs, hm)
    kernel, kernel_grad, _, _, _, _, _ = str_to_kernel_set(kernels, "mc", hm)

    np.random.seed(0)
    env1 = generate_mb_envs(cutoffs, np.eye(3) * 100, delta, d1)
    env2 = generate_mb_envs(cutoffs, np.eye(3) * 100, delta, d2)
    env1 = env1[0][0]
    env2 = env2[0][0]

    k, grad = kernel_grad(env1, env2, d1, d2, *args)

    original = kernel(env1, env2, d1, d2, *args)

    nhyps = len(hyps)
    if hm["train_noise"]:
        nhyps -= 1
    original_hyps = Parameters.get_hyps(hm, hyps=hyps)

    for i in range(nhyps):
        newhyps = np.copy(hyps)
        newhyps[i] += delta
        if "map" in hm.keys():
            newid = hm["map"][i]
            hm["original_hyps"] = np.copy(original_hyps)
            hm["original_hyps"][newid] += delta
        newargs = from_mask_to_args(newhyps, cutoffs, hm)

        hgrad = (kernel(env1, env2, d1, d2, *newargs) - original) / delta
        if "map" in hm:
            print(i, "hgrad", hgrad, grad[hm["map"][i]])
            assert isclose(grad[hm["map"][i]], hgrad, rtol=tol)
        else:
            print(i, "hgrad", hgrad, grad[i])
            assert isclose(grad[i], hgrad, rtol=tol)


def generate_same_hm(kernels, multi_cutoff=False):
    """
    generate hyperparameter and masks that are effectively the same
    but with single or multi group expression
    """
    pm1 = ParameterHelper(species=["H", "He"], parameters={"noise": 0.05})

    pm2 = ParameterHelper(species=["H", "He"], parameters={"noise": 0.05})

    if "twobody" in kernels:
        para = 2.5 + 0.1 * random(3)
        pm1.set_parameters("cutoff_twobody", para[-1])
        pm1.define_group("twobody", "twobody0", ["*", "*"], para[:-1])

        pm2.set_parameters("cutoff_twobody", para[-1])
        pm2.define_group("twobody", "twobody0", ["*", "*"], para[:-1])
        pm2.define_group("twobody", "twobody1", ["H", "H"], para[:-1])

        if multi_cutoff:
            pm2.set_parameters("twobody0", para)
            pm2.set_parameters("twobody1", para)

    if "threebody" in kernels:
        para = 1.2 + 0.1 * random(3)
        pm1.set_parameters("cutoff_threebody", para[-1])
        pm1.define_group("threebody", "threebody0", ["*", "*", "*"], para[:-1])

        pm2.set_parameters("cutoff_threebody", para[-1])
        pm2.define_group("threebody", "threebody0", ["*", "*", "*"], para[:-1])
        pm2.define_group("threebody", "threebody1", ["H", "H", "H"], para[:-1])

        if multi_cutoff:
            pm2.define_group("cut3b", "c1", ["*", "*"], parameters=para)
            pm2.define_group("cut3b", "c2", ["H", "H"], parameters=para)

    if "manybody" in kernels:
        para = 1.2 + 0.1 * random(3)

        pm1.set_parameters("cutoff_manybody", para[-1])
        pm1.define_group("manybody", "manybody0", ["*", "*"], para[:-1])

        pm2.set_parameters("cutoff_manybody", para[-1])
        pm2.define_group("manybody", "manybody0", ["*", "*"], para[:-1])
        pm2.define_group("manybody", "manybody1", ["H", "H"], para[:-1])

        if multi_cutoff:
            pm2.set_parameters("manybody0", para)
            pm2.set_parameters("manybody1", para)

    hm1 = pm1.as_dict()
    hyps1 = hm1["hyps"]
    cut = hm1["cutoffs"]

    hm2 = pm2.as_dict()
    hyps2 = hm2["hyps"]
    cut = hm2["cutoffs"]

    return cut, hyps1, hyps2, hm1, hm2


def generate_diff_hm(kernels, diff_cutoff=False, constraint=False):

    pm = ParameterHelper(species=["H", "He"], parameters={"noise": 0.05})

    if "twobody" in kernels:
        para1 = 2.5 + 0.1 * random(3)
        para2 = 2.5 + 0.1 * random(3)
        pm.set_parameters("cutoff_twobody", para1[-1])
        pm.define_group("twobody", "twobody0", ["*", "*"])
        pm.set_parameters("twobody0", para1[:-1], not constraint)
        pm.define_group("twobody", "twobody1", ["H", "H"], para2[:-1])

        if diff_cutoff:
            pm.set_parameters("twobody0", para1, not constraint)
            pm.set_parameters("twobody1", para2)

    if "threebody" in kernels:
        para1 = 1.2 + 0.1 * random(3)
        para2 = 1.2 + 0.1 * random(3)
        pm.set_parameters("cutoff_threebody", para1[-1])
        pm.define_group("threebody", "threebody0", ["*", "*", "*"], para1[:-1])
        pm.set_parameters("threebody0", para1[:-1], not constraint)
        pm.define_group("threebody", "threebody1", ["H", "H", "H"], para2[:-1])

        if diff_cutoff:
            pm.define_group("cut3b", "c1", ["*", "*"], parameters=para1)
            pm.define_group("cut3b", "c2", ["H", "H"], parameters=para2)

    if "manybody" in kernels:
        para1 = 1.2 + 0.1 * random(3)
        para2 = 1.2 + 0.1 * random(3)

        pm.set_parameters("cutoff_manybody", para1[-1])
        pm.define_group("manybody", "manybody0", ["*", "*"])
        pm.set_parameters("manybody0", para1[:-1], not constraint)
        pm.define_group("manybody", "manybody1", ["H", "H"], para2[:-1])

        if diff_cutoff:
            pm.set_parameters("manybody0", para1, not constraint)
            pm.set_parameters("manybody1", para2)

    hm = pm.as_dict()
    hyps = hm["hyps"]
    cut = hm["cutoffs"]

    return cut, hyps, hm

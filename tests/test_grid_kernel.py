import numpy as np
import pytest
import sys

from copy import deepcopy
from itertools import combinations_with_replacement, permutations
from numpy import isclose
from numpy.random import random, randint

from flare.descriptors.env import AtomicEnvironment
from flare.kernels.utils import from_mask_to_args, str_to_kernel_set
from flare.kernels.cutoffs import quadratic_cutoff_bound, quadratic_cutoff
from flare.atoms import FLARE_Atoms
from flare.utils.parameter_helper import ParameterHelper

from tests.fake_gp import generate_mb_envs, generate_mb_twin_envs
from tests.test_mc_sephyps import generate_same_hm, generate_diff_hm
import flare.bffs.mgp.map2b as m2
import flare.bffs.mgp.map3b as m3
from flare.bffs.mgp.grid_kernels import grid_kernel

# multi_cut = [False, True]
bodies = [2, 3]
hyps_list = [True, False]


@pytest.mark.parametrize("bodies", bodies)
@pytest.mark.parametrize("same_hyps", hyps_list)
@pytest.mark.parametrize("prefix", ["energy"])  # , 'force'])
def test_start(parameter, bodies, same_hyps, prefix):

    if bodies == 2:
        xb_map = m2
        kernel_name = "twobody"
        grid_dim = 1
    elif bodies == 3:
        xb_map = m3
        kernel_name = "threebody"
        grid_dim = 3

    env1, env2, hm1, hm2 = parameter[kernel_name]

    # get environments, hyps and arguments for training data
    env = env1 if same_hyps else env2
    hm = hm1 if same_hyps else hm2
    kernel = grid_kernel
    hyps_mask = None if same_hyps else hm
    # # debug
    # for k in env.__dict__:
    #     print(k, env.__dict__[k])

    # get all possible triplet grids
    list_of_triplet = list(combinations_with_replacement([1, 2], 3))
    # # debug
    # print(list_of_triplet)

    for comb in list_of_triplet:
        for species in set(permutations(comb)):

            grid_env, grid = get_grid_env(species, parameter, kernel_name, same_hyps)

            # # debug
            # print(species)
            # for k in grid_env.__dict__:
            #     print(k, grid_env.__dict__[k])
            # print(grid)

            reference = get_reference(
                grid_env, species, parameter, kernel_name, same_hyps
            )

            coords = np.zeros((1, grid_dim * 3))
            coords[:, 0] = np.ones_like(coords[:, 0])

            fj, fdj = xb_map.bonds_cutoff(
                grid,
                hm["cutoffs"][kernel_name],
                coords,
                derivative=True,
                cutoff_func=quadratic_cutoff,
            )
            fdj = fdj[:, [0]]

            hyps, r_cut = xb_map.get_hyps_for_kern(
                hm["hyps"], hm["cutoffs"], hyps_mask, grid_env.ctype, grid_env.etypes
            )

            kern_type = f"{prefix}_force"
            kern_vec = kernel(
                env,
                bodies,
                kern_type,
                xb_map.get_bonds_for_kern,
                xb_map.bonds_cutoff,
                grid_env.ctype,
                grid_env.etypes,
                hyps,
                r_cut,
                grid,
                fj,
                fdj,
            )
            kern_vec = np.hstack(kern_vec)
            print("species, reference, kern_vec, reference-kern_vec")
            print(species, reference, kern_vec, reference - kern_vec)


@pytest.fixture(scope="module")
def parameter():

    np.random.seed(10)
    xb_list = ["twobody", "threebody"]
    all_dict = {}
    for b in xb_list:
        kernels = [b]

        delta = 1e-8
        cutoffs, hyps1, hyps2, hm1, hm2 = generate_same_hm(kernels, multi_cutoff=False)
        (
            cutoffs,
            hyps2,
            hm2,
        ) = generate_diff_hm(kernels, diff_cutoff=False, constraint=False)

        cell = 1e7 * np.eye(3)
        env1 = generate_mb_envs(hm1["cutoffs"], cell, delta, 1)
        env2 = generate_mb_envs(hm2["cutoffs"], cell, delta, 2)
        env1 = env1[0][0]
        env2 = env2[0][0]

        all_dict[b] = (env1, env2, hm1, hm2)

    yield all_dict
    del all_dict


def get_grid_env(species, parameter, kernel_name, same_hyps):
    """generate a single triplet environment"""

    env1, env2, hm1, hm2 = parameter[kernel_name]
    hm = hm1 if same_hyps else hm2

    big_cell = np.eye(3) * 100
    r1 = 0.5
    r2 = 0.5

    if kernel_name == "twobody":
        positions = [[0, 0, 0], [r1, 0, 0]]
        grid_struc = FLARE_Atoms(
            symbols=species[:2], cell=big_cell, positions=positions
        )
        env = AtomicEnvironment(grid_struc, 0, hm["cutoffs"], hm)
        grid = np.array([[r1]])
    elif kernel_name == "threebody":
        positions = [[0, 0, 0], [r1, 0, 0], [0, r2, 0]]
        grid_struc = FLARE_Atoms(symbols=species, cell=big_cell, positions=positions)
        env = AtomicEnvironment(grid_struc, 0, hm["cutoffs"], hm)
        env.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])
        grid = np.array([[r1, r2, np.sqrt(r1**2 + r2**2)]])
    return env, grid


def get_reference(grid_env, species, parameter, kernel_name, same_hyps):

    env1, env2, hm1, hm2 = parameter[kernel_name]
    env = env1 if same_hyps else env2
    hm = hm1 if same_hyps else hm2

    kernel, kg, en_kernel, force_en_kernel, _, _, _ = str_to_kernel_set(
        hm["kernels"], "mc", None if same_hyps else hm
    )
    args = from_mask_to_args(hm["hyps"], hm["cutoffs"], None if same_hyps else hm)

    energy_force = np.zeros(3, dtype=np.float)
    # force_force = np.zeros(3, dtype=np.float)
    # force_energy = np.zeros(3, dtype=np.float)
    # energy_energy = np.zeros(3, dtype=np.float)
    for i in range(3):
        energy_force[i] = force_en_kernel(env, grid_env, i + 1, *args)
        # force_energy[i] = force_en_kernel(env, grid_env, i, *args)
        # force_force[i] = kernel(grid_env, env, 0, i, *args)
    #     result = funcs[1][i](env1, env2, d1, *args1)
    return energy_force  # , force_energy, force_force, energy_energy

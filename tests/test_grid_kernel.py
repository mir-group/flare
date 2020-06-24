import numpy as np
import pytest
import sys

from copy import deepcopy
from itertools import combinations_with_replacement, permutations
from numpy import isclose
from numpy.random import random, randint

from flare.env import AtomicEnvironment
from flare.kernels.utils import from_mask_to_args, str_to_kernel_set
from flare.kernels.cutoffs import quadratic_cutoff_bound
from flare.parameters import Parameters
from flare.struc import Structure
from flare.utils.parameter_helper import ParameterHelper

from tests.fake_gp import generate_mb_envs, generate_mb_twin_envs
from tests.test_mc_sephyps import generate_same_hm, generate_diff_hm
from flare.mgp.utils import get_triplets, get_kernel_term
from flare.mgp.grid_kernels_3b import triplet_cutoff, grid_kernel_sephyps, \
        grid_kernel, get_triplets_for_kern

# multi_cut = [False, True]
hyps_list = [True, False]

@pytest.mark.parametrize('same_hyps', hyps_list)
@pytest.mark.parametrize('prefix', ['energy']) #, 'force'])
def test_start(parameter, same_hyps, prefix):

    env1, env2, hm1, hm2 = parameter

    # get environments, hyps and arguments for training data
    env = env1 if same_hyps else env2
    hm = hm1 if same_hyps else hm2
    kernel = grid_kernel if same_hyps else grid_kernel_sephyps
    args = from_mask_to_args(hm['hyps'], hm['cutoffs'], None if same_hyps else hm)
    # # debug
    # for k in env.__dict__:
    #     print(k, env.__dict__[k])

    # get all possible triplet grids
    list_of_triplet = list(combinations_with_replacement([1, 2], 3))
    # # debug
    # print(list_of_triplet)

    for comb in list_of_triplet:
        for species in  set(permutations(comb)):

            grid_env, grid = get_grid_env(species, parameter, same_hyps)

            # # debug
            # print(species)
            # for k in grid_env.__dict__:
            #     print(k, grid_env.__dict__[k])
            # print(grid)

            reference = get_reference(grid_env, species, parameter, same_hyps)


            coords = np.zeros((1, 9), dtype=np.float64)
            coords[:, 0] += 1.0

            fj, fdj = triplet_cutoff(grid, hm['cutoffs']['threebody'],
                                     coords, derivative=True)
            fdj = fdj[:, [0]]

            kern_type = f'{prefix}_force'
            kern_vec = kernel(kern_type, env, grid, fj, fdj,
                              grid_env.ctype, grid_env.etypes,
                              *args)
            kern_vec = np.hstack(kern_vec)
            print('species, reference, kern_vec, reference-kern_vec')
            print(species, reference, kern_vec, reference-kern_vec)

@pytest.fixture(scope='module')
def parameter():

    np.random.seed(10)
    kernels = ['threebody']

    delta = 1e-8
    cutoffs, hyps1, hyps2, hm1, hm2 = generate_same_hm(
        kernels, multi_cutoff=False)
    cutoffs, hyps2, hm2, = generate_diff_hm(
            kernels, diff_cutoff=False, constraint=False)

    cell = 1e7 * np.eye(3)
    env1 = generate_mb_envs(hm1['cutoffs'], cell, delta, 1)
    env2 = generate_mb_envs(hm2['cutoffs'], cell, delta, 2)
    env1 = env1[0][0]
    env2 = env2[0][0]

    all_list = (env1, env2, hm1, hm2)

    yield all_list
    del all_list

def get_grid_env(species, parameter, same_hyps):
    '''generate a single triplet environment'''

    env1, env2, hm1, hm2 = parameter

    big_cell = np.eye(3) * 100
    r1 = 0.5
    r2 = 0.5
    positions = [[0, 0, 0], [r1, 0, 0], [0, r2, 0]]
    grid_struc = Structure(big_cell, species, positions)
    if same_hyps:
        env = AtomicEnvironment(grid_struc, 0, hm1['cutoffs'], hm1)
    else:
        env = AtomicEnvironment(grid_struc, 0, hm2['cutoffs'], hm2)

    env.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])

    grid = np.array([[r1, r2, np.sqrt(r1**2+r2**2)]])
    return env, grid

def get_reference(grid_env, species, parameter, same_hyps):

    env1, env2, hm1, hm2 = parameter
    env = env1 if same_hyps else env2
    hm = hm1 if same_hyps else hm2

    kernel, kg, en_kernel, force_en_kernel = str_to_kernel_set(
            hm['kernels'], "mc", None if same_hyps else hm)
    args = from_mask_to_args(hm['hyps'], hm['cutoffs'], None if same_hyps else hm)

    energy_force = force_en_kernel(env, grid_env, *args)
        # force_energy[i] = force_en_kernel(env, grid_env, i, *args)
        # force_force[i] = kernel(grid_env, env, 0, i, *args)
#     result = funcs[1][i](env1, env2, *args1)
    return energy_force # , force_energy, force_force, energy_energy


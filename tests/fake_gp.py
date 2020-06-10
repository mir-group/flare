from copy import deepcopy
import pytest
import numpy as np
from numpy.random import random, randint, permutation

from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.utils.parameter_helper import ParameterHelper


def get_random_structure(cell, unique_species, noa):
    """Create a random test structure """
    np.random.seed(0)

    positions = []
    forces = []
    species = []

    for n in range(noa):
        positions.append(np.random.uniform(-1, 1, 3))
        forces.append(np.random.uniform(-1, 1, 3))
        species.append(unique_species[np.random.randint(0,
                                                        len(unique_species))])

    test_structure = Structure(cell, species, positions)

    return test_structure, forces


def generate_hm(ntwobody, nthreebody, nmanybody=1, constraint=False, multihyps=True):

    cutoff = 0.8
    if (multihyps is False):
        kernels = []
        parameters = {}
        if (ntwobody > 0):
            kernels += ['twobody']
            parameters['cutoff_twobody'] = cutoff
        if (nthreebody > 0):
            kernels += ['threebody']
            parameters['cutoff_threebody'] = cutoff
        if (nmanybody > 0):
            kernels += ['manybody']
            parameters['cutoff_manybody'] = cutoff
        pm = ParameterHelper(kernels=kernels, random=True,
                parameters=parameters)
        hm = pm.as_dict()
        hyps = hm['hyps']
        cut = hm['cutoffs']
        return hyps, hm, cut

    pm = ParameterHelper(species=['H', 'He'], parameters={'noise':0.05})
    if (ntwobody > 0):
        pm.define_group('twobody', 'b1', ['*', '*'], parameters=random(2))
        pm.set_parameters('cutoff_twobody', cutoff)
    if (nthreebody > 0):
        pm.define_group('threebody', 't1', ['*', '*', '*'], parameters=random(2))
        pm.set_parameters('cutoff_threebody', cutoff)
    if (nmanybody > 0):
        pm.define_group('manybody', 'manybody1', ['*', '*'], parameters=random(2))
        pm.set_parameters('cutoff_manybody', cutoff)
    if (ntwobody > 1):
        pm.define_group('twobody', 'b2', ['H', 'H'], parameters=random(2))
    if (nthreebody > 1):
        pm.define_group('threebody', 't2', ['H', 'H', 'H'], parameters=random(2))

    if (constraint is False):
        hm = pm.as_dict()
        hyps = hm['hyps']
        cut = hm['cutoffs']
        return hyps, hm, cut

    pm.set_constraints('b1', opt=[True, False])
    pm.set_constraints('t1', opt=[False, True])
    hm = pm.as_dict()
    hyps = hm['hyps']
    cut = hm['cutoffs']
    return hyps, hm, cut


def get_gp(bodies, kernel_type='mc', multihyps=True, cellabc=[1, 1, 1.5], 
           force_only=False) -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""
    print("\nSetting up...\n")

    # params
    cell = np.diag(cellabc)
    unique_species = [2, 1, 3]
    noa = 5

    ntwobody = 0
    nthreebody = 0
    prefix = bodies
    if ('2' in bodies or 'two' in bodies):
        ntwobody = 1
    if ('3' in bodies or 'three' in bodies):
        nthreebody = 1

    hyps, hm, _ = generate_hm(ntwobody, nthreebody, nmanybody=0, multihyps=multihyps)
    cutoffs = hm['cutoffs']
    kernels = hm['kernels']
    hl = hm['hyp_labels']

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  noa)
    energy = 3.14

    # test update_db
    gaussian = \
        GaussianProcess(kernels=kernels,
                        component=kernel_type,
                        hyps=hyps,
                        hyp_labels=hl,
                        cutoffs=cutoffs, hyps_mask=hm,
                        parallel=False, n_cpus=1)
    if force_only:
        gaussian.update_db(test_structure, forces)
    else:
        gaussian.update_db(test_structure, forces, energy=energy)
    gaussian.check_L_alpha()

    #print(gaussian.alpha)

    return gaussian


def get_params():
    parameters = {'unique_species': [2, 1],
                  'cutoffs': {'twobody': 0.8},
                  'noa': 5,
                  'cell': np.eye(3),
                  'db_pts': 30}
    return parameters


def get_tstp(hm=None) -> AtomicEnvironment:
    """Create test point for kernel to compare against"""
    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoffs = {'twobody':0.8, 'threebody': 0.8, 'manybody': 0.8}
    noa = 10

    test_structure_2, _ = get_random_structure(cell, unique_species,
                                               noa)

    test_pt = AtomicEnvironment(test_structure_2, 0,
                                cutoffs, cutoffs_mask=hm)
    return test_pt


def generate_mb_envs(cutoffs, cell, delt, d1, mask=None, kern_type='mc'):
    positions0 = np.array([[0., 0., 0.],
                           [0., 0.3, 0.],
                           [0.3, 0., 0.],
                           [0.0, 0., 0.3],
                           [1., 1., 0.]])
    positions0[1:] += 0.1*np.random.random([4, 3])
    threebody = [1, 1, 2, 1]
    np.random.shuffle(threebody)
    species_1 = np.hstack([threebody, randint(1, 2)])
    if kern_type == 'sc':
        species_1 = np.ones(species_1.shape)
    return generate_mb_envs_pos(positions0, species_1, cutoffs, cell, delt, d1, mask)


def generate_mb_twin_envs(cutoffs, cell, delt, d1, mask=None):

    positions0 = np.array([[0., 0., 0.],
                           [0., 0.3, 0.],
                           [0.3, 0., 0.],
                           [0.0, 0., 0.3],
                           [1., 1., 0.]])
    positions0[1:] += 0.1*np.random.random([4, 3])

    species_1 = np.array([1, 2, 1, 1, 1], dtype=np.int)
    env1 = generate_mb_envs_pos(
        positions0, species_1, cutoffs, cell, delt, d1, mask)

    positions1 = positions0[[0, 2, 3, 4]]
    species_2 = species_1[[0, 2, 3, 4]]
    env2 = generate_mb_envs_pos(
        positions1, species_2, cutoffs, cell, delt, d1, mask)

    return env1, env2


def generate_mb_envs_pos(positions0, species_1, cutoffs, cell, delt, d1, mask=None):

    positions = [positions0]

    noa = len(positions0)

    positions_2 = deepcopy(positions0)
    positions_2[0][d1-1] = delt
    positions += [positions_2]

    positions_3 = deepcopy(positions[0])
    positions_3[0][d1-1] = -delt
    positions += [positions_3]

    test_struc = []
    for i in range(3):
        test_struc += [Structure(cell, species_1, positions[i])]

    env_0 = []
    env_p = []
    env_m = []
    for i in range(noa):
        env_0 += [AtomicEnvironment(test_struc[0], i, cutoffs, cutoffs_mask=mask)]
        env_p += [AtomicEnvironment(test_struc[1], i, cutoffs, cutoffs_mask=mask)]
        env_m += [AtomicEnvironment(test_struc[2], i, cutoffs, cutoffs_mask=mask)]
    return [env_0, env_p, env_m]

def generate_envs(cutoffs, delta):
    """
    create environment with perturbation on
    direction i
    """
    # create env 1
    # perturb the x direction of atom 0 for +- delta
    cell = np.eye(3)*(np.max(list(cutoffs.values()))+0.1)
    atom_1 = 0
    pos_1 = np.vstack([[0, 0, 0], random([3, 3])])
    pos_2 = deepcopy(pos_1)
    pos_2[0][0] = delta
    pos_3 = deepcopy(pos_1)
    pos_3[0][0] = -delta

    species_1 = [1, 2, 1, 1] # , 1, 1, 2, 1, 2]
    test_structure_1 = Structure(cell, species_1, pos_1)
    test_structure_2 = Structure(cell, species_1, pos_2)
    test_structure_3 = Structure(cell, species_1, pos_3)

    env1_1 = AtomicEnvironment(test_structure_1, atom_1, cutoffs)
    env1_2 = AtomicEnvironment(test_structure_2, atom_1, cutoffs)
    env1_3 = AtomicEnvironment(test_structure_3, atom_1, cutoffs)


    # create env 2
    # perturb the y direction
    pos_1 = np.vstack([[0, 0, 0], random([3, 3])])
    pos_2 = deepcopy(pos_1)
    pos_2[0][1] = delta
    pos_3 = deepcopy(pos_1)
    pos_3[0][1] = -delta

    atom_2 = 0
    species_2 = [1, 1, 2, 1] #, 2, 1, 2, 2, 2]

    test_structure_1 = Structure(cell, species_2, pos_1)
    test_structure_2 = Structure(cell, species_2, pos_2)
    test_structure_3 = Structure(cell, species_2, pos_3)

    env2_1 = AtomicEnvironment(test_structure_1, atom_2, cutoffs)
    env2_2 = AtomicEnvironment(test_structure_2, atom_2, cutoffs)
    env2_3 = AtomicEnvironment(test_structure_3, atom_2, cutoffs)

    return env1_1, env1_2, env1_3, env2_1, env2_2, env2_3

def another_env(cutoffs, delt):

    cell = 10.0 * np.eye(3)

    # atomic structure 1
    pos_1 = np.vstack([[0, 0, 0], 0.1*random([3, 3])])
    pos_1[1, 1] += 1
    pos_1[2, 0] += 1
    pos_1[3, :2] += 1
    pos_2 = deepcopy(pos_1)
    pos_2[0][0] = delt
    pos_3 = deepcopy(pos_1)
    pos_3[0][0] = -delt


    species_1 = [1, 1, 1, 1]

    test_structure_1 = Structure(cell, species_1, pos_1)
    test_structure_2 = Structure(cell, species_1, pos_2)
    test_structure_3 = Structure(cell, species_1, pos_3)

    # atom 0, original position
    env1_1_0 = AtomicEnvironment(test_structure_1, 0, cutoffs)
    # atom 0, 0 perturbe along x
    env1_2_0 = AtomicEnvironment(test_structure_2, 0, cutoffs)
    # atom 1, 0 perturbe along x
    env1_2_1 = AtomicEnvironment(test_structure_2, 1, cutoffs)
    # atom 2, 0 perturbe along x
    env1_2_2 = AtomicEnvironment(test_structure_2, 2, cutoffs)

    # atom 0, 0 perturbe along -x
    env1_3_0 = AtomicEnvironment(test_structure_3, 0, cutoffs)
    # atom 1, 0 perturbe along -x
    env1_3_1 = AtomicEnvironment(test_structure_3, 1, cutoffs)
    # atom 2, 0 perturbe along -x
    env1_3_2 = AtomicEnvironment(test_structure_3, 2, cutoffs)

    # create env 2
    pos_1 = np.vstack([[0, 0, 0], 0.1*random([3, 3])])
    pos_1[1, 1] += 1
    pos_1[2, 0] += 1
    pos_1[3, :2] += 1
    pos_2 = deepcopy(pos_1)
    pos_2[0][0] = delt
    pos_3 = deepcopy(pos_1)
    pos_3[0][0] = -delt

    species_2 = [1, 2, 2, 1]

    test_structure_1 = Structure(cell, species_2, pos_1)

    env2_1_0 = AtomicEnvironment(test_structure_1, 0, cutoffs)

    return env1_1_0, env1_2_0, env1_3_0, \
           env1_2_1, env1_3_1, env1_2_2, env1_3_2, env2_1_0


from copy import deepcopy
import pytest
import numpy as np
from numpy.random import random, randint, permutation

from flare import env, struc, gp
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.otf_parser import OtfAnalysis

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


def generate_hm(nbond, ntriplet, cutoffs=[1, 1], constraint=False, multihyps=True):

    if (multihyps is False):
        hyps_label = []
        if (nbond > 0):
            nbond = 1
            hyps_label += ['Length', 'Signal Var.']
        if (ntriplet >0):
            ntriplet = 1
            hyps_label += ['Length', 'Signal Var.']
        hyps_label += ['Noise Var.']
        return random((nbond+ntriplet)*2+1), {'hyps_label':hyps_label}, deepcopy(cutoffs)

    specs_mask = np.zeros(118, dtype=int)
    specs_mask[1] = 0
    specs_mask[2] = 1
    nspecs = 2

    specs_mask = np.zeros(118, dtype=int)
    specs_mask[1] = 0
    specs_mask[2] = 1
    nspecs = 2

    cut = []
    cut += [cutoffs[0]]
    cut += [cutoffs[1]]

    if (nbond>1):
        sig1 = random(nbond)
        ls1 = random(nbond)
        bond_mask = np.ones(nspecs**2, dtype=int)
        bond_mask[0] = 0
        bond_name = ["sig2"]*nbond+["ls2"]*nbond
    else:
        sig1 = [random()]
        ls1 = [random()]
        bond_mask = np.zeros(nspecs**2, dtype=int)
        bond_name = ["sig2"]+["ls2"]

    if (ntriplet>1):
        sig2 = random(ntriplet)
        ls2 = random(ntriplet)
        triplet_mask = np.ones(nspecs**3, dtype=int)
        triplet_mask[0] = 0
        triplet_name = ["sig3"]*ntriplet+["ls3"]*ntriplet
    else:
        sig2 = [random()]
        ls2 = [random()]
        triplet_mask = np.zeros(nspecs**3, dtype=int)
        triplet_name = ["sig3"]+["ls3"]

    sigman = [0.05]

    if (nbond>0 and ntriplet>0):
        hyps = np.hstack([sig1, ls1, sig2, ls2, sigman])
        hyps_label = np.hstack([bond_name, triplet_name, ['noise']])
    elif (nbond>0):
        hyps = np.hstack([sig1, ls1, sigman])
        hyps_label = np.hstack([bond_name, ['noise']])
    else:
        hyps = np.hstack([sig2, ls2, sigman])
        hyps_label = np.hstack([triplet_name, ['noise']])

    hyps_mask = {'nspec': nspecs,
                 'spec_mask': specs_mask,
                 'nbond': nbond,
                 'bond_mask': bond_mask,
                 'ntriplet': ntriplet,
                 'triplet_mask': triplet_mask,
                 'hyps_label': hyps_label,
                 'train_noise': True}
    if (constraint is False):
        return hyps, hyps_mask, cut

    hyps_mask['map'] = []
    hyps_mask['original'] = hyps
    hm = hyps_mask['map']
    count = 0
    newhyps = []
    newlabel = []
    if (nbond>1):
        # fix type 0, and only compute type 1 of bonds
        hm += [1]
        newhyps += [hyps[1]]
        newlabel += [hyps_label[1]]
        hm += [3]
        newhyps += [hyps[3]]
        newlabel += [hyps_label[3]]
        count += 4
    elif (nbond>0):
        # fix sigma, and only vary ls
        hm += [1]
        newhyps += [hyps[1]]
        newlabel += [hyps_label[1]]
        count += 2
    if (ntriplet>1):
        # fix type 0, and only compute type 1 of triplets
        hm += [1+count]
        newhyps += [hyps[1+count]]
        newlabel += [hyps_label[1+count]]
        hm += [3+count]
        newhyps += [hyps[3+count]]
        newlabel += [hyps_label[3+count]]
    elif (ntriplet>0):
        # fix sigma, and only vary ls
        hm += [1+count]
        newhyps += [hyps[1+count]]
        newlabel += [hyps_label[1+count]]
    hm += [len(hyps)-1]
    newhyps += [hyps[-1]]
    newlabel += ['noise']
    hyps = np.hstack(newhyps)
    hyps_mask['hyps_label'] = np.hstack(hyps_label)

    return hyps, hyps_mask, cut


def get_gp(bodies, kernel_type='mc', multihyps=True) -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""
    print("\nSetting up...\n")

    # params
    cell = np.diag(np.array([1, 1, 1.5]))
    unique_species = [2, 1]
    cutoffs = np.array([0.8, 0.8])
    noa = 5

    nbond = 0
    ntriplet = 0
    prefix = bodies
    if ('2' in bodies or 'two' in bodies):
        nbond = 1
    if ('3' in bodies or 'three' in bodies):
        ntriplet = 1

    hyps, hm, _ = generate_hm(nbond, ntriplet, multihyps=multihyps)

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  noa)

    hl = hm['hyps_label']
    if (multihyps is False):
        hm = None

    # test update_db
    gaussian = \
        GaussianProcess(kernel_name=f'{prefix}{kernel_type}',
                        hyps=hyps,
                        hyp_labels=hl,
                        cutoffs=cutoffs, multihyps=multihyps, hyps_mask=hm,
                        parallel=False, n_cpus=1)
    gaussian.update_db(test_structure, forces)
    gaussian.check_L_alpha()

    return gaussian

def get_params():
    parameters = {'unique_species': [2, 1],
                  'cutoff': 0.8,
                  'noa': 5,
                  'cell': np.eye(3),
                  'db_pts': 30}
    return parameters

def get_tstp() -> AtomicEnvironment:
    """Create test point for kernel to compare against"""
    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoffs = np.ones(3)*0.8
    noa = 10

    test_structure_2, _ = get_random_structure(cell, unique_species,
                                               noa)

    test_pt = AtomicEnvironment(test_structure_2, 0,
                                cutoffs)
    return test_pt


def generate_envs(cutoffs, delta, mask=None):
    """
    create environment with perturbation on
    direction i
    """
    # create env 1
    # perturb the x direction of atom 0 for +- delta
    cell = np.eye(3)*np.max(cutoffs+0.1)
    atom_1 = 0
    pos_1 = np.vstack([[0, 0, 0], random([3, 3])])
    pos_2 = deepcopy(pos_1)
    pos_2[0][0] = delta
    pos_3 = deepcopy(pos_1)
    pos_3[0][0] = -delta

    species_1 = [1, 2, 1, 1] # , 1, 1, 2, 1, 2]
    test_structure_1 = struc.Structure(cell, species_1, pos_1)
    test_structure_2 = struc.Structure(cell, species_1, pos_2)
    test_structure_3 = struc.Structure(cell, species_1, pos_3)

    env1_1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs, mask)
    env1_2 = env.AtomicEnvironment(test_structure_2, atom_1, cutoffs, mask)
    env1_3 = env.AtomicEnvironment(test_structure_3, atom_1, cutoffs, mask)


    # create env 2
    # perturb the y direction
    pos_1 = np.vstack([[0, 0, 0], random([3, 3])])
    pos_2 = deepcopy(pos_1)
    pos_2[0][1] = delta
    pos_3 = deepcopy(pos_1)
    pos_3[0][1] = -delta

    atom_2 = 0
    species_2 = [1, 1, 2, 1] #, 2, 1, 2, 2, 2]

    test_structure_1 = struc.Structure(cell, species_2, pos_1)
    test_structure_2 = struc.Structure(cell, species_2, pos_2)
    test_structure_3 = struc.Structure(cell, species_2, pos_3)

    env2_1 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs, mask)
    env2_2 = env.AtomicEnvironment(test_structure_2, atom_2, cutoffs, mask)
    env2_3 = env.AtomicEnvironment(test_structure_3, atom_2, cutoffs, mask)

    return env1_1, env1_2, env1_3, env2_1, env2_2, env2_3


def generate_mb_envs(cutoffs, cell, delt, d1, mask=None):

    positions = []
    positions += [[np.array([0., 0., 0.]),
                   np.array([0., 0.3, 0.]) + 0.1 *
                   np.array([random(), random(), random()]),
                   np.array([0.3, 0., 0.]) + 0.1 *
                   np.array([random(), random(), random()]),
                   np.array([1., 1., 0.]) + 0.1 * np.array([random(), random(), random()])]]

    noa = len(positions[0])

    positions_2 = deepcopy(positions[0])
    positions_2[0][d1-1] = delt
    positions += [positions_2]

    positions_3 = deepcopy(positions[0])
    positions_3[0][d1-1] = -delt
    positions += [positions_3]

    triplet = [1, 1, 2]
    np.random.shuffle(triplet)
    species_1 = np.hstack([triplet, randint(1, 2)])
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

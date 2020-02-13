import pytest
import numpy as np
from numpy.random import random, randint, permutation

import flare.kernels.mc_sephyps as en

from flare import env, struc, gp
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare import mc_simple
from flare.otf_parser import OtfAnalysis
from flare.kernels.mc_sephyps import _str_to_kernel as stk

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


def generate_hm(nbond, ntriplet, cutoffs=[1, 1], constraint=False):

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

    if (nbond==2):
        sig1 = random(nbond)
        ls1 = random(nbond)
        bond_mask = np.ones(nspecs**2, dtype=int)
        bond_mask[0] = 0
        bond_name = ["sig2"]*2+["ls2"]*2
    else:
        sig1 = [random()]
        ls1 = [random()]
        bond_mask = np.zeros(nspecs**2, dtype=int)
        bond_name = ["sig2"]+["ls2"]

    if (ntriplet==2):
        sig2 = random(ntriplet)
        ls2 = random(ntriplet)
        triplet_mask = np.ones(nspecs**3, dtype=int)
        triplet_mask[0] = 0
        triplet_name = ["sig3"]*2+["ls3"]*2
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
    if (nbond>0):
        # fix type 0, and only compute type 1 of bonds
        hm += [1]
        newhyps += [hyps[1]]
        newlabel += [hyps_label[1]]
        hm += [3]
        newhyps += [hyps[3]]
        newlabel += [hyps_label[3]]
        count += 4
    if (ntriplet>0):
        # fix type 0, and only compute type 1 of triplets
        hm += [1+count]
        newhyps += [hyps[1+count]]
        newlabel += [hyps_label[1+count]]
        hm += [3+count]
        newhyps += [hyps[3+count]]
        newlabel += [hyps_label[3+count]]
    hm += [len(hyps)-1]
    newhyps += [hyps[-1]]
    newlabel += ['noise']
    hyps = np.hstack(newhyps)
    hyps_mask['hyps_label'] = np.hstack(hyps_label)

    return hyps, hyps_mask, cut


def gp2b() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""
    print("\nSetting up...\n")

    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoffs = np.array([0.8, 0.8])
    noa = 5

    nbond = 1
    ntriplet = 0
    hyps, hm, _ = generate_hm(nbond, ntriplet)

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  noa)

    # test update_db
    gaussian = \
        GaussianProcess(kernel=en.two_body_mc,
                        kernel_grad=en.two_body_mc_grad,
                        hyps=hyps,
                        hyp_labels=hm['hyps_label'],
                        cutoffs=cutoffs, multihyps=True, hyps_mask=hm,
                        par=False, n_cpus=1)
    gaussian.update_db(test_structure, forces)

    return gaussian

def gp3b() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""
    print("\nSetting up...\n")

    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoffs = np.array([0.8, 0.8])
    noa = 5

    nbond = 0
    ntriplet = 1
    hyps, hm, _ = generate_hm(nbond, ntriplet)

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  noa)

    # test update_db
    gaussian = \
        GaussianProcess(kernel=en.three_body_mc,
                        kernel_grad=en.three_body_mc_grad,
                        hyps=hyps,
                        hyp_labels=hm['hyps_label'],
                        cutoffs=cutoffs, multihyps=True, hyps_mask=hm)
    gaussian.update_db(test_structure, forces)

    return gaussian


def gp23b() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""
    print("\nSetting up...\n")

    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoffs = np.array([0.8, 0.8])
    noa = 5

    nbond = 1
    ntriplet = 1
    hyps, hm, _ = generate_hm(nbond, ntriplet)

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  noa)
    efk = en.two_plus_three_mc_force_en
    ek = en.two_plus_three_mc_en

    # test update_db
    gaussian = \
        GaussianProcess(kernel=en.two_plus_three_body_mc,
                        kernel_grad=en.two_plus_three_body_mc_grad,
                        hyps=hyps,
                        energy_force_kernel = efk,
                        energy_kernel=ek,
                        opt_algorithm = 'BFGS',
                        hyp_labels=hm['hyps_label'],
                        cutoffs=cutoffs, multihyps=True, hyps_mask=hm)
    gaussian.update_db(test_structure, forces)

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
    cutoff = 0.8
    noa = 10

    test_structure_2, _ = get_random_structure(cell, unique_species,
                                               noa)

    test_pt = AtomicEnvironment(test_structure_2, 0,
                                np.array([cutoff, cutoff]))
    return test_pt

import numpy as np

from numpy import array
from numba import njit
from math import exp, floor
from typing import Callable

from flare.env import AtomicEnvironment
from flare.kernels.cutoffs import quadratic_cutoff
from flare.kernels.utils import str_to_kernel_set as stks
from flare.parameters import Parameters


def get_kernel_term(GP, term):
    """
    Args
        term (str): 'twobody' or 'threebody'
    """
    kernel, _, ek, efk = stks([term], GP.component, GP.hyps_mask)

    hyps, cutoffs, hyps_mask = Parameters.get_component_mask(GP.hyps_mask, term, hyps=GP.hyps)

    return (kernel, ek, efk, cutoffs, hyps, hyps_mask)

def get_permutations(c2, ej1, ej2):
    perm_list = [[0, 1, 2]]
    if c2 == ej1:
        perm_list += [[0, 2, 1]]

    if c2 == ej2:
        perm_list += [[2, 1, 0]]

    if ej1 == ej2:
        perm_list += [[1, 0, 2]]

    if (c2 == ej1) and (ej1 == ej2):
        perm_list += [[1, 2, 0]]
        perm_list += [[2, 0, 1]]

    return np.array(perm_list)



def get_l_bound(curr_l_bound, structure, two_d=False):
    positions = structure.positions
    if two_d:
        cell = structure.cell[:2]
    else:
        cell = structure.cell

    min_dist = curr_l_bound
    for ind1, pos01 in enumerate(positions):
        for i1 in range(2):
            for vec1 in cell:
                pos1 = pos01 + i1 * vec1

                for ind2, pos02 in enumerate(positions):
                    for i2 in range(2):
                        for vec2 in cell:
                            pos2 = pos02 + i2 * vec2

                            if np.all(pos1 == pos2):
                                continue
                            dist12 = np.linalg.norm(pos1-pos2)
                            if dist12 < min_dist:
                                min_dist = dist12
                                min_atoms = (ind1, ind2)
    return min_dist


@njit
def get_bonds(ctype, etypes, bond_array):
    exist_species = []
    bond_lengths = []
    bond_dirs = []
    for i in range(len(bond_array)):
        bond = bond_array[i]
        if ctype <= etypes[i]:
            spc = [ctype, etypes[i]]
            b_dir = bond[1:]
        else:
            spc = [etypes[i], ctype]
            b_dir = bond[1:]

        if spc in exist_species:
            ind = exist_species.index(spc)
            bond_lengths[ind].append([bond[0]])
            bond_dirs[ind].append(b_dir)
        else:
            exist_species.append(spc)
            bond_lengths.append([[bond[0]]])
            bond_dirs.append([b_dir])
    return exist_species, bond_lengths, bond_dirs

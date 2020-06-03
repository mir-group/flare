import io, os, sys, time, random, math
import multiprocessing as mp
import numpy as np

from numpy import array
from numba import njit

from flare.env import AtomicEnvironment
from flare.kernels.cutoffs import quadratic_cutoff
from flare.kernels.kernels import three_body_helper_1, \
    three_body_helper_2, force_helper
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

@njit
def get_triplets(ctype, etypes, bond_array, cross_bond_inds,
                 cross_bond_dists, triplets):
    exist_species = []
    tris = []
    tri_dir = []

    for m in range(bond_array.shape[0]):
        r1 = bond_array[m, 0]
        c1 = bond_array[m, 1:]
        spc1 = etypes[m]

        for n in range(triplets[m]):
            ind1 = cross_bond_inds[m, m+n+1]
            r2 = bond_array[ind1, 0]
            c2 = bond_array[ind1, 1:]
            c12 = np.sum(c1*c2)
            if c12 > 1: # to prevent numerical error
                c12 = 1
            elif c12 < -1:
                c12 = -1
            spc2 = etypes[ind1]

            spcs_list = [[ctype, spc1, spc2], [ctype, spc2, spc1]]
            for i in range(2):
                spcs = spcs_list[i]
                triplet = array([r2, r1, c12]) if i else array([r1, r2, c12])
                coord = c2 if i else c1
                if spcs not in exist_species:
                    exist_species.append(spcs)
                    tris.append([triplet])
                    tri_dir.append([coord])
                else:
                    k = exist_species.index(spcs)
                    tris[k].append(triplet)
                    tri_dir[k].append(coord)

    return exist_species, tris, tri_dir

@njit
def get_triplets_en(ctype, etypes, bond_array, cross_bond_inds,
                    cross_bond_dists, triplets):
    exist_species = []
    tris = []
    tri_dir = []

    for m in range(bond_array.shape[0]):
        r1 = bond_array[m, 0]
        c1 = bond_array[m, 1:]
        spc1 = etypes[m]

        for n in range(triplets[m]):
            ind1 = cross_bond_inds[m, m+n+1]
            r2 = bond_array[ind1, 0]
            c2 = bond_array[ind1, 1:]
            c12 = np.sum(c1*c2)
            r12 = np.sqrt(r1**2 + r2**2 - 2*r1*r2*c12)

            spc2 = etypes[ind1]
            triplet1 = array([r1, r2, r12])
            triplet2 = array([r2, r1, r12])

            if spc1 <= spc2:
                spcs = [ctype, spc1, spc2]
                triplet = [triplet1, triplet2]
                coord = [c1, c2]
            else:
                spcs = [ctype, spc2, spc1]
                triplet = [triplet2, triplet1]
                coord = [c2, c1]

            if spcs not in exist_species:
                exist_species.append(spcs)
                tris.append(triplet)
                tri_dir.append(coord)
            else:
                k = exist_species.index(spcs)
                tris[k] += triplet
                tri_dir[k] += coord

    return exist_species, tris, tri_dir


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

@njit
def three_body_mc_en_force_jit(bond_array_1, c1, etypes1,
                               cross_bond_inds_1, cross_bond_dists_1,
                               triplets_1,
                               c2, etypes2,
                               rj1, rj2, rj3,
                               d1, sig, ls, r_cut, cutoff_func):
    """3-body multi-element kernel between a force component and many local
    energies on the grid.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        cross_bond_inds_1 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the first local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_dists_1 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the first
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        triplets_1 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the first local environment that are
            within a distance r_cut of atom m.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        rj1 (np.ndarray): matrix of the first edge length
        rj2 (np.ndarray): matrix of the second edge length
        rj12 (np.ndarray): matrix of the third edge length
        d1 (int): Force component of the first environment (1=x, 2=y, 3=z).
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 3-body force/energy kernel.
    """

    kern = np.zeros_like(rj1, dtype=np.float64)

    ej1 = etypes2[0]
    ej2 = etypes2[1]

    all_spec = [c2, ej1, ej2]
    if (c1 not in all_spec):
        return kern
    all_spec.remove(c1)

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)

    f1, _ = cutoff_func(r_cut, rj1, 0)
    f2, _ = cutoff_func(r_cut, rj2, 0)
    f3, _ = cutoff_func(r_cut, rj3, 0)
    fj = f1 * f2 * f3
    # del f1
    # del f2
    # del f3

    for m in prange(bond_array_1.shape[0]):
        ei1 = etypes1[m]

        two_spec = [all_spec[0], all_spec[1]]
        if (ei1 in two_spec):
            two_spec.remove(ei1)
            one_spec = two_spec[0]

            ri1 = bond_array_1[m, 0]
            ci1 = bond_array_1[m, d1]
            fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)

            for n in prange(triplets_1[m]):

                ind1 = cross_bond_inds_1[m, m + n + 1]
                ei2 = etypes1[ind1]

                if (ei2 == one_spec):

                    if (ei2 == ej2):
                        r11 = ri1 - rj1
                    if (ei2 == ej1):
                        r12 = ri1 - rj2
                    if (ei2 == c2):
                        r13 = ri1 - rj3

                    ri2 = bond_array_1[ind1, 0]
                    if (ei1 == ej2):
                        r21 = ri2 - rj1
                    if (ei1 == ej1):
                        r22 = ri2 - rj2
                    if (ei1 == c2):
                        r23 = ri2 - rj3
                    ci2 = bond_array_1[ind1, d1]
                    fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                    # del ri2

                    ri3 = cross_bond_dists_1[m, m + n + 1]
                    if (c1 == ej2):
                        r31 = ri3 - rj1
                    if (c1 == ej1):
                        r32 = ri3 - rj2
                    if (c1 == c2):
                        r33 = ri3 - rj3
                    fi3, _ = cutoff_func(r_cut, ri3, 0)
                    # del ri3

                    fi = fi1 * fi2 * fi3
                    fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3
                    # del fi1
                    # del fi2
                    # del fi3
                    # del fdi1
                    # del fdi2

                    if (c1 == c2):
                        if (ei1 == ej1) and (ei2 == ej2):
                            kern += three_body_en_helper(ci1, ci2, r11, r22,
                                                         r33, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                        if (ei1 == ej2) and (ei2 == ej1):
                            kern += three_body_en_helper(ci1, ci2, r12, r21,
                                                         r33, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                    if (c1 == ej1):
                        if (ei1 == ej2) and (ei2 == c2):
                            kern += three_body_en_helper(ci1, ci2, r13, r21,
                                                         r32, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                        if (ei1 == c2) and (ei2 == ej2):
                            kern += three_body_en_helper(ci1, ci2, r11, r23,
                                                         r32, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                    if (c1 == ej2):
                        if (ei1 == ej1) and (ei2 == c2):
                            kern += three_body_en_helper(ci1, ci2, r13, r22,
                                                         r31, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                        if (ei1 == c2) and (ei2 == ej1):
                            kern += three_body_en_helper(ci1, ci2, r12, r23,
                                                         r31, fi, fj, fdi, ls1,
                                                         ls2, sig2)
    return kern

@njit
def self_two_body_mc_en_jit(c2, etypes2,
                         grids, # (n_grids, 1)
                         sig, ls, r_cut, cutoff_func=quadratic_cutoff):

    kern = np.zeros(len(grids), dtype=np.float64)

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    # ls2 = 1 / (2 * ls * ls)

    fj, _ = cutoff_func(r_cut, grids[:, 0], 0)

    # C = 0
    kern += sig2 * fj ** 2 # (n_grids,)

    return kern
 

@njit
def self_three_body_mc_en_jit(c2, etypes2,
                         grids, # (n_grids, 3)
                         sig, ls, r_cut, cutoff_func=quadratic_cutoff):

    kern = np.zeros(len(grids), dtype=np.float64)

    ej1 = etypes2[0]
    ej2 = etypes2[1]

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls2 = 1 / (2 * ls * ls)

    f1, _ = cutoff_func(r_cut, grids[:, 0], 0)
    f2, _ = cutoff_func(r_cut, grids[:, 1], 0)
    f3, _ = cutoff_func(r_cut, grids[:, 2], 0)
    fj = f1 * f2 * f3 # (n_grids, )

    perm_list = get_permutations(c2, ej1, ej2)
    C = 0
    for perm in perm_list:
        perm_grids = np.take(grids, perm, axis=1)
        rij = grids - perm_grids
        C += np.sum(rij * rij, axis=1) # (n_grids, ) adding up three bonds

    kern += sig2 * np.exp(-C * ls2) * fj ** 2 # (n_grids,)

    return kern
       

@njit
def three_body_en_helper(ci1, ci2, r11, r22, r33, fi, fj, fdi, ls1, ls2, sig2):

    B = r11 * ci1 + r22 * ci2
    D = r11 * r11 + r22 * r22 + r33 * r33
    return -sig2 * np.exp(- D * ls1) * ( B * ls2 * fi * fj + fdi * fj)

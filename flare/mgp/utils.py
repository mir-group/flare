import numpy as np
from numpy import array
from numba import njit
import io, os, sys, time, random, logging
import multiprocessing as mp
import cProfile

import flare.gp as gp
import flare.env as env
import flare.struc as struc
from flare.env import AtomicEnvironment
from flare.kernels.kernels import three_body_helper_1, \
    three_body_helper_2, force_helper
from flare.cutoffs import quadratic_cutoff


def save_GP(GP, prefix):
    np.save(prefix+'alpha', GP.alpha)
    np.save(prefix+'hyps', GP.hyps)
    np.save(prefix+'l_mat', GP.l_mat)


def load_GP(GP, prefix):
    GP.alpha = np.load(prefix+'alpha.npy')
    GP.hyps = np.load(prefix+'hyps.npy')
    GP.l_mat = np.load(prefix+'l_mat.npy')
    l_mat_inv = np.linalg.inv(GP.l_mat)
    GP.ky_mat_inv = l_mat_inv.T @ l_mat_inv


def save_grid(bond_lens, bond_ens_diff, bond_vars_diff, prefix):
    np.save(prefix+'-bond_lens', bond_lens)
    np.save(prefix+'-bond_ens_diff', bond_ens_diff)
    np.save(prefix+'-bond_vars_diff', bond_vars_diff)


def load_grid(prefix):
    bond_lens = np.load(prefix+'bond_lens.npy')
    bond_ens_diff = np.load(prefix+'bond_ens_diff.npy')
    bond_vars_diff = np.load(prefix+'bond_vars_diff.npy')
    return bond_lens, bond_ens_diff, bond_vars_diff


def merge(prefix, a_num, g_num):
    grid_means = np.zeros((g_num, g_num, a_num))
    grid_vars = np.zeros((g_num, g_num, a_num, g_num, g_num, a_num))
    for a12 in range(a_num):
        grid_means[:,:,a12] = np.load(prefix+str((a12, 0))+'-bond_means.npy')
        for a34 in range(a_num):
            grid_vars[:,:,a12,:,:,a34] = np.load(prefix+str((a12, a34))+'-bond_vars.npy')
    return grid_means, grid_vars


def svd_grid(matr, rank):
    u, s, vh = np.linalg.svd(matr, full_matrices=False)
    return u[:,:rank], s[:rank], vh[:rank, :]


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
        spc = sorted([ctype, etypes[i]])
        if spc in exist_species:
            ind = exist_species.index(spc)
            bond_lengths[ind].append([bond[0]])
            bond_dirs[ind].append(bond[1:])
        else:
            exist_species.append(spc)
            bond_lengths.append([[bond[0]]])
            bond_dirs.append([bond[1:]])
    return exist_species, bond_lengths, bond_dirs


#@njit
def add_triplets(spcs_list, exist_species, tris, tri_dir,
        r1, r2, a12, c1, c2):
    for i in range(2):
        spcs = spcs_list[i]
        if spcs not in exist_species:
            exist_species.append(spcs)
            tris.append([])
            tri_dir.append([])

        k = exist_species.index(spcs)
        triplet = (r2, r1, a12) if i else (r1, r2, a12)
        coord = c2 if i else c1
        tris[k].append(triplet)
        tri_dir[k].append(coord)
    return exist_species, tris, tri_dir


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

#            if spc1 == spc2:
#                spcs_list = [[ctype, spc1, spc2], [ctype, spc1, spc2]]
#            elif ctype == spc1: # spc1 != spc2
#                spcs_list = [[ctype, spc1, spc2], [spc2, ctype, spc1]]
#            elif ctype == spc2: # spc1 != spc2
#                spcs_list = [[spc1, spc2, ctype], [spc2, ctype, spc1]]
#            else: # all different
#                spcs_list = [[ctype, spc1, spc2], [ctype, spc2, spc1]]

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
def self_two_body_mc_jit(bond_array, c, etypes,
                    d, sig, ls, r_cut, cutoff_func):

    """Multicomponent two-body force/force kernel accelerated with Numba's
    njit decorator.

    Loops over bonds in two environments and adds to the kernel if bonds are
    of the same type.
    """

    kern = 0

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig * sig

    for m in range(bond_array.shape[0]):
        ri = bond_array[m, 0]
        ci = bond_array[m, d]
        fi, fdi = cutoff_func(r_cut, ri, ci)
        e1 = etypes[m]

        for n in range(m, bond_array.shape[0]):
            e2 = etypes[n]

            # check if bonds agree
            if (c == c and e1 == e2) or (c == e2 and c == e1):
                rj = bond_array[n, 0]
                cj = bond_array[n, d]
                fj, fdj = cutoff_func(r_cut, rj, cj)
                r11 = ri - rj

                A = ci * cj
                B = r11 * ci
                C = r11 * cj
                D = r11 * r11

                kern0 = force_helper(A, B, C, D, fi, fj, fdi, fdj,
                                     ls1, ls2, ls3, sig2)
                kern += kern0 if m == n else 2 * kern0

    return kern


@njit
def self_three_body_mc_jit(bond_array, cross_bond_inds, cross_bond_dists,
                 triplets, c, etypes, d, sig, ls, r_cut, cutoff_func):
    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig*sig
    ls1 = 1 / (2*ls*ls)
    ls2 = 1 / (ls*ls)
    ls3 = ls2*ls2

    for m in range(bond_array.shape[0]):
        ri1 = bond_array[m, 0]
        ci1 = bond_array[m, d]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
        ei1 = etypes[m]

        for n in range(triplets[m]):
            ind1 = cross_bond_inds[m, m+n+1]
            ri2 = bond_array[ind1, 0]
            ci2 = bond_array[ind1, d]
            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
            ei2 = etypes[ind1]

            ri3 = cross_bond_dists[m, m+n+1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            fi = fi1*fi2*fi3
            fdi = fdi1*fi2*fi3+fi1*fdi2*fi3

            for p in range(m, bond_array.shape[0]):
                rj1 = bond_array[p, 0]
                cj1 = bond_array[p, d]
                fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                ej1 = etypes[p]

                for q in range(triplets[p]):
                    ind2 = cross_bond_inds[p, p+1+q]
                    rj2 = bond_array[ind2, 0]
                    cj2 = bond_array[ind2, d]
                    fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                    ej2 = etypes[ind2]

                    rj3 = cross_bond_dists[p, p+1+q]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)

                    fj = fj1*fj2*fj3
                    fdj = fdj1*fj2*fj3+fj1*fdj2*fj3

                    r11 = ri1-rj1
                    r12 = ri1-rj2
                    r13 = ri1-rj3
                    r21 = ri2-rj1
                    r22 = ri2-rj2
                    r23 = ri2-rj3
                    r31 = ri3-rj1
                    r32 = ri3-rj2
                    r33 = ri3-rj3

                    if (c == c):
                        if (ei1 == ej1) and (ei2 == ej2):
                            kern0 = \
                                three_body_helper_1(ci1, ci2, cj1, cj2, r11,
                                                    r22, r33, fi, fj, fdi, fdj,
                                                    ls1, ls2, ls3, sig2)
                        if (ei1 == ej2) and (ei2 == ej1):
                            kern0 = \
                                three_body_helper_1(ci1, ci2, cj2, cj1, r12,
                                                    r21, r33, fi, fj, fdi, fdj,
                                                    ls1, ls2, ls3, sig2)
                    if (c == ej1):
                        if (ei1 == ej2) and (ei2 == c):
                            kern0 = \
                                three_body_helper_2(ci2, ci1, cj2, cj1, r21,
                                                    r13, r32, fi, fj, fdi,
                                                    fdj, ls1, ls2, ls3, sig2)
                        if (ei1 == c) and (ei2 == ej2):
                            kern0 = \
                                three_body_helper_2(ci1, ci2, cj2, cj1, r11,
                                                    r23, r32, fi, fj, fdi,
                                                    fdj, ls1, ls2, ls3, sig2)
                    if (c == ej2):
                        if (ei1 == ej1) and (ei2 == c):
                            kern0 = \
                                three_body_helper_2(ci2, ci1, cj1, cj2, r22,
                                                    r13, r31, fi, fj, fdi,
                                                    fdj, ls1, ls2, ls3, sig2)
                        if (ei1 == c) and (ei2 == ej1):
                            kern0 = \
                                three_body_helper_2(ci1, ci2, cj1, cj2, r12,
                                                    r23, r31, fi, fj, fdi,
                                                    fdj, ls1, ls2, ls3, sig2)
                    kern += kern0 if m == p else 2 * kern0

    return kern

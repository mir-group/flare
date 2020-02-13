import numpy as np
from numpy import array
from numba import njit
import io, os, sys, time, random, math
import multiprocessing as mp

import flare.gp as gp
import flare.env as env
import flare.struc as struc
import flare.kernels.mc_simple as mc_simple
import flare.kernels.mc_sephyps as mc_sephyps
import flare.kernels.kernels as sc
from flare.env import AtomicEnvironment
from flare.kernels.kernels import three_body_helper_1, \
    three_body_helper_2, force_helper
from flare.cutoffs import quadratic_cutoff
from flare.kernels.utils import str_to_kernel_set as stks



def get_2bkernel(GP):
    if 'mc' in GP.kernel_name:
        kernel, _, _, efk = stks('2mc', GP.multihyps)
    else:
        kernel, _, _, efk = stks('2', GP.multihyps)

    cutoffs = [GP.cutoffs[0]]

    original_hyps = np.copy(GP.hyps)
    if (GP.multihyps is True):
        o_hyps_mask = GP.hyps_mask
        if ('map' in o_hyps_mask.keys()):
            ori_hyps = o_hyps_mask['original']
            hm = o_hyps_mask['map']
            for i, h in enumerate(original_hyps):
                ori_hyps[hm[i]]=h
        else:
            ori_hyps = original_hyps
        n2b = o_hyps_mask['nbond']
        hyps = np.hstack([ori_hyps[:n2b*2], ori_hyps[-1]])
        hyps_mask = {'nbond':n2b, 'ntriplet':0,
              'nspec':o_hyps_mask['nspec'],
              'spec_mask':o_hyps_mask['spec_mask'],
              'bond_mask': o_hyps_mask['bond_mask']}
    else:
        hyps = [GP.hyps[0], GP.hyps[1], GP.hyps[-1]]
        hyps_mask = None
    return (kernel, efk, cutoffs, hyps, hyps_mask)


def get_3bkernel(GP):

    if 'mc' in GP.kernel_name:
        kernel, _, _, efk = stks('3mc', GP.multihyps)
    else:
        kernel, _, _, efk = stks('3', GP.multihyps)

    if 'two' in GP.kernel_name:
        base = 2
    else:
        base = 0

    cutoffs = np.copy(GP.cutoffs)

    original_hyps = np.copy(GP.hyps)
    if (GP.multihyps is True):
        o_hyps_mask = GP.hyps_mask
        if ('map' in o_hyps_mask.keys()):
            ori_hyps = o_hyps_mask['original']
            hm = o_hyps_mask['map']
            for i, h in enumerate(original_hyps):
                ori_hyps[hm[i]]=h
        else:
            ori_hyps = original_hyps
        n2b = o_hyps_mask['nbond']
        n3b = o_hyps_mask['ntriplet']
        hyps = ori_hyps[n2b*2:]
        hyps_mask = {'ntriplet':n3b,'nbond':0,
                'nspec':o_hyps_mask['nspec'],
                'spec_mask':o_hyps_mask['spec_mask'],
                'triplet_mask': o_hyps_mask['triplet_mask']}
    else:
        hyps = [GP.hyps[0+base], GP.hyps[1+base], GP.hyps[-1]]
        hyps_mask = None

    return (kernel, efk, cutoffs, hyps, hyps_mask)


def en_kern_vec(training_data, x: AtomicEnvironment,
                energy_force_kernel, hyps, cutoffs, hyps_mask=None):
    """Compute the vector of energy/force kernels between an atomic \
ronment and the environments in the training set."""

    ds = [1, 2, 3]
    size = len(training_data) * 3
    k_v = np.zeros(size, )

    for m_index in range(size):
        x_2 = training_data[int(math.floor(m_index / 3))]
        d_2 = ds[m_index % 3]
        if (hyps_mask is None):
            k_v[m_index] = energy_force_kernel(x_2, x, d_2,
                                               hyps, cutoffs)
        else:
            k_v[m_index] = energy_force_kernel(x_2, x, d_2,
                                               hyps, cutoffs,
                                               hyps_mask=hyps_mask)

    return k_v




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

#            if spc1 == spc2:
#                spcs_list = [[ctype, spc1, spc2], [ctype, spc1, spc2]]
#            elif ctype == spc1: # spc1 != spc2
#                spcs_list = [[ctype, spc1, spc2], [spc2, ctype, spc1]]
#            elif ctype == spc2: # spc1 != spc2
#                spcs_list = [[spc1, spc2, ctype], [spc2, ctype, spc1]]
#            else: # all different
#                spcs_list = [[ctype, spc1, spc2], [ctype, spc2, spc1]]

            if spc1 <= spc2:
                spcs = [ctype, spc1, spc2]
                triplet = array([r1, r2, r12])
                coord = [c1, c2]
            else:
                spcs = [ctype, spc2, spc1]
                triplet = array([r2, r1, r12])
                coord = [c2, c1]

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

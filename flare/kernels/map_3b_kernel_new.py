"""Multi-element 2-, 3-, and 2+3-body kernels that restrict all signal
variance hyperparameters to a single value."""
import numpy as np
from numba import njit, prange
from math import exp, floor
from typing import Callable

from flare.env import AtomicEnvironment
import flare.kernels.cutoffs as cf

def three_body_mc_en(env1: AtomicEnvironment, grids, fj, c2, etypes2, perm_list,
                     hyps: 'ndarray', cutoffs: 'ndarray',
                     cutoff_func: Callable = cf.quadratic_cutoff) \
        -> float:
    """3-body multi-element kernel between a force component and many local
    energies on the grid.

    Args:
        env1 (AtomicEnvironment): First local environment.
        rj1 (np.ndarray): matrix of the first edge length
        rj2 (np.ndarray): matrix of the second edge length
        rj12 (np.ndarray): matrix of the third edge length
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        d1 (int): Force component of the first environment (1=x, 2=y, 3=z).
        hyps (np.ndarray): Hyperparameters of the kernel function (sig1, ls1,
            sig2, ls2).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 3-body force/energy kernel.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    return three_body_mc_en_jit(env1.bond_array_3, env1.ctype,
                                env1.etypes,
                                env1.cross_bond_inds,
                                env1.cross_bond_dists,
                                env1.triplet_counts,
                                c2, etypes2, perm_list,
                                grids, fj,
                                sig, ls, r_cut, cutoff_func) / 9.

def three_body_mc_en_sephyps(env1, grids, fj, c2, etypes2, perm_list,
                             cutoff_2b, cutoff_3b, nspec, spec_mask,
                             nbond, bond_mask, ntriplet, triplet_mask,
                             ncut3b, cut3b_mask,
                             sig2, ls2, sig3, ls3,
                             cutoff_func=cf.quadratic_cutoff) -> float:
    """3-body multi-element kernel between a force component and many local
    energies on the grid.

    Args:
        env1 (AtomicEnvironment): First local environment.
        rj1 (np.ndarray): matrix of the first edge length
        rj2 (np.ndarray): matrix of the second edge length
        rj12 (np.ndarray): matrix of the third edge length
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        d1 (int): Force component of the first environment (1=x, 2=y, 3=z).
        cutoff_2b: dummy
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond: dummy
        bond_mask: dummy
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        sig2: dummy
        ls2: dummy
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        cutoff_func (Callable): Cutoff function of the kernel.

    Returns:
        float:
            Value of the 3-body force/energy kernel.
    """

    ej1 = etypes2[0]
    ej2 = etypes2[1]
    bc1 = spec_mask[c2]
    bc2 = spec_mask[ej1]
    bc3 = spec_mask[ej2]
    ttype = triplet_mask[nspec * nspec * bc1 + nspec*bc2 + bc3]
    ls = ls3[ttype]
    sig = sig3[ttype]
    r_cut = cutoff_3b

    return three_body_mc_en_jit(env1.bond_array_3, env1.ctype,
                                env1.etypes,
                                env1.cross_bond_inds,
                                env1.cross_bond_dists,
                                env1.triplet_counts,
                                c2, etypes2, perm_list,
                                grids, fj,
                                sig, ls, r_cut, cutoff_func) / 9.


def three_body_mc_en_force(env1: AtomicEnvironment, grids, fj, c2, etypes2, perm_list,
                           d1: int, hyps: 'ndarray', cutoffs: 'ndarray',
                           cutoff_func: Callable = cf.quadratic_cutoff) \
        -> float:
    """3-body multi-element kernel between a force component and many local
    energies on the grid.

    Args:
        env1 (AtomicEnvironment): First local environment.
        rj1 (np.ndarray): matrix of the first edge length
        rj2 (np.ndarray): matrix of the second edge length
        rj12 (np.ndarray): matrix of the third edge length
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        d1 (int): Force component of the first environment (1=x, 2=y, 3=z).
        hyps (np.ndarray): Hyperparameters of the kernel function (sig1, ls1,
            sig2, ls2).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 3-body force/energy kernel.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    return three_body_mc_en_force_jit(env1.bond_array_3, env1.ctype,
                                      env1.etypes,
                                      env1.cross_bond_inds,
                                      env1.cross_bond_dists,
                                      env1.triplet_counts,
                                      c2, etypes2, perm_list,
                                      grids, fj,
                                      d1, sig, ls, r_cut, cutoff_func) / 3

def three_body_mc_en_force_sephyps(env1, grids, fj, c2, etypes2, perm_list,
                                   d1, cutoff_2b, cutoff_3b, cutoff_mb,
                                   nspec, spec_mask,
                                   nbond, bond_mask,
                                   ntriplet, triplet_mask,
                                   ncut3b, cut3b_mask,
                                   nmb, mb_mask,
                                   sig2, ls2, sig3, ls3, sigm, lsm,
                                   cutoff_func=cf.quadratic_cutoff) -> float:
    """3-body multi-element kernel between a force component and many local
    energies on the grid.

    Args:
        env1 (AtomicEnvironment): First local environment.
        rj1 (np.ndarray): matrix of the first edge length
        rj2 (np.ndarray): matrix of the second edge length
        rj12 (np.ndarray): matrix of the third edge length
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        d1 (int): Force component of the first environment (1=x, 2=y, 3=z).
        cutoff_2b: dummy
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond: dummy
        bond_mask: dummy
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        sig2: dummy
        ls2: dummy
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        cutoff_func (Callable): Cutoff function of the kernel.

    Returns:
        float:
            Value of the 3-body force/energy kernel.
    """

    ej1 = etypes2[0]
    ej2 = etypes2[1]
    bc1 = spec_mask[c2]
    bc2 = spec_mask[ej1]
    bc3 = spec_mask[ej2]
    ttype = triplet_mask[nspec * nspec * bc1 + nspec*bc2 + bc3]
    ls = ls3[ttype]
    sig = sig3[ttype]
    r_cut = cutoff_3b

    return three_body_mc_en_force_jit(env1.bond_array_3, env1.ctype,
                                      env1.etypes,
                                      env1.cross_bond_inds,
                                      env1.cross_bond_dists,
                                      env1.triplet_counts,
                                      c2, etypes2, perm_list,
                                      grids, fj,
                                      d1, sig, ls, r_cut, cutoff_func) / 3

#@njit
def three_body_mc_en_force_jit(bond_array_1, c1, etypes1,
                               cross_bond_inds_1, cross_bond_dists_1,
                               triplets_1,
                               c2, etypes2, perm_list,
                               grids, fj,
                               d1, sig, ls, r_cut, cutoff_func):

    kern = np.zeros(grids.shape[0], dtype=np.float64)

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)

    triplet_coord_list = get_triplets_for_kern(bond_array_1, c1, etypes1,
        cross_bond_inds_1, cross_bond_dists_1, triplets_1,
        c2, etypes2, perm_list, d1)

    if len(triplet_coord_list) == 0: # no triplets
        return kern

    triplet_list = triplet_coord_list[:, :3] # (n_triplets, 3)
    coord_list = triplet_coord_list[:, 3:]

    fi, fdi = triplet_cutoff_grad(triplet_list, r_cut, coord_list) # (n_triplets, 1)
    fifj = fi @ fj.T # (n_triplets, n_grids)
    fdij = fdi @ fj.T

    B = 0
    D = 0
    for d in range(3):
        rj, ri = np.meshgrid(grids[:, d], triplet_list[:, d])
        rij = ri - rj
        D += rij * rij # (n_triplets, n_grids)

        # column-wise multiplication
        # coord_list[:, [d]].shape = (n_triplets, 1)
        B += rij * coord_list[:, [d]] # (n_triplets, n_grids)

    kern = - np.sum(sig2 * np.exp(- D * ls1) * (B * ls2 * fifj + fdij), axis=0) # (n_grids,)

    return kern


@njit
def three_body_mc_en_jit(bond_array_1, c1, etypes1,
                         cross_bond_inds_1,
                         cross_bond_dists_1,
                         triplets_1,
                         c2, etypes2, perm_list,
                         grids, fj,
                         sig, ls, r_cut, cutoff_func):

    kern = np.zeros(grids.shape[0], dtype=np.float64)

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls2 = 1 / (2 * ls * ls)

    triplet_coord_list = get_triplets_for_kern(bond_array_1, c1, etypes1,
        cross_bond_inds_1, cross_bond_dists_1, triplets_1,
        c2, etypes2, perm_list, d1)

    if len(triplet_coord_list) == 0: # no triplets
        return kern

    triplet_list = triplet_coord_list[:, :3] # (n_triplets, 3)

    fi = triplet_cutoff(triplet_list, r_cut) # (n_triplets, 1)
    fifj = fi @ fj.T # (n_triplets, n_grids)

    C = 0
    for d in range(3):
        rj, ri = np.meshgrid(grids[:, d], triplet_list[:, d])
        rij = ri - rj
        C += rij * rij # (n_triplets, n_grids)

    kern = np.sum(sig2 * np.exp(-C * ls2) * fifj, axis=0) # (n_grids,)

    return kern


@njit
def triplet_cutoff(triplets, r_cut, cutoff_func=cf.quadratic_cutoff):
    f0, _ = cutoff_func(r_cut, triplets, 0) # (n_grid, 3)
    fj = f0[:, 0] * f0[:, 1] * f0[:, 2] # (n_grid,)
    return np.expand_dims(fj, axis=1) # (n_grid, 1)

@njit
def triplet_cutoff_grad(triplets, r_cut, coords, cutoff_func=cf.quadratic_cutoff):
    f0, df0 = cutoff_func(r_cut, triplets, coords) # (n_grid, 3)
    fj = f0[:, 0] * f0[:, 1] * f0[:, 2] # (n_grid,)
    fj = np.expand_dims(fj, axis=1)
    dfj = df0[:, 0] *  f0[:, 1] *  f0[:, 2] + \
           f0[:, 0] * df0[:, 1] *  f0[:, 2] + \
           f0[:, 0] *  f0[:, 1] * df0[:, 2]
    dfj = np.expand_dims(dfj, axis=1)
    return fj, dfj

#@njit
def get_triplets_for_kern(bond_array_1, c1, etypes1,
                          cross_bond_inds_1, cross_bond_dists_1,
                          triplets_1,
                          c2, etypes2, perm_list, d1):

    triplet_list = np.empty((0, 6), dtype=np.float64)

    ej1 = etypes2[0]
    ej2 = etypes2[1]

    all_spec = [c2, ej1, ej2]
    if (c1 not in all_spec):
        return []
    c1_ind = all_spec.index(c1)
    ind_list = [0, 1, 2]
    ind_list.remove(c1_ind)
    all_spec.remove(c1)

    for m in range(bond_array_1.shape[0]):
        two_inds = ind_list.copy()

        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        ei1 = etypes1[m]

        two_spec = [all_spec[0], all_spec[1]]
        if (ei1 in two_spec):

            ei1_ind = ind_list[0] if ei1 == two_spec[0] else ind_list[1]
            two_spec.remove(ei1)
            two_inds.remove(ei1_ind)
            one_spec = two_spec[0]
            ei2_ind = two_inds[0]

            for n in range(triplets_1[m]):
                ind1 = cross_bond_inds_1[m, m + n + 1]
                ei2 = etypes1[ind1]
                if (ei2 == one_spec):

                    order = [c1_ind, ei1_ind, ei2_ind]
                    ri2 = bond_array_1[ind1, 0]
                    ci2 = bond_array_1[ind1, d1]

                    ri3 = cross_bond_dists_1[m, m + n + 1]

                    # align this triplet to the same species order as r1, r2, r12
                    tri = np.take(np.array([ri1, ri2, ri3]), order)
                    crd = np.take(np.array([ci1, ci2,   0]), order)

                    # append permutations
                    for perm in perm_list:
                        tricrd = np.hstack((np.take(tri, perm), np.take(crd, perm)))
                        triplet_list = np.vstack((triplet_list, tricrd))

    return triplet_list

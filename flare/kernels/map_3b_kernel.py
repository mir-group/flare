"""Multi-element 2-, 3-, and 2+3-body kernels that restrict all signal
variance hyperparameters to a single value."""
import numpy as np
from numba import njit, prange
from math import exp, floor
from typing import Callable

from flare.env import AtomicEnvironment
import flare.kernels.cutoffs as cf

def three_body_mc_en(env1: AtomicEnvironment, r1, r2, r12, ctype_2, etypes_2,
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
        ctype_2 (int): Species of the central atom of the second local environment.
        etypes_2 (np.ndarray): Species of atoms in the second local
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
                                ctype_2, etypes_2,
                                r1, r2, r12,
                                sig, ls, r_cut, cutoff_func) / 9.

def three_body_mc_en_sephyps(env1, r1, r2, r12, ctype_2, etypes_2,
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
        ctype_2 (int): Species of the central atom of the second local environment.
        etypes_2 (np.ndarray): Species of atoms in the second local
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

    ej1 = etypes_2[0]
    ej2 = etypes_2[1]
    bctype_1 = spec_mask[ctype_2]
    bctype_2 = spec_mask[ej1]
    bc3 = spec_mask[ej2]
    ttype = triplet_mask[nspec * nspec * bctype_1 + nspec*bctype_2 + bc3]
    ls = ls3[ttype]
    sig = sig3[ttype]
    r_cut = cutoff_3b

    return three_body_mc_en_jit(env1.bond_array_3, env1.ctype,
                                env1.etypes,
                                env1.cross_bond_inds,
                                env1.cross_bond_dists,
                                env1.triplet_counts,
                                ctype_2, etypes_2,
                                r1, r2, r12,
                                sig, ls, r_cut, cutoff_func) / 9.


def three_body_mc_en_force(env1: AtomicEnvironment, r1, r2, r12, ctype_2, etypes_2,
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
        ctype_2 (int): Species of the central atom of the second local environment.
        etypes_2 (np.ndarray): Species of atoms in the second local
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
                                      ctype_2, etypes_2,
                                      r1, r2, r12,
                                      d1, sig, ls, r_cut, cutoff_func) / 3

def three_body_mc_en_force_sephyps(env1, r1, r2, r12, ctype_2, etypes_2,
                                   d1, cutoff_2b, cutoff_3b, nspec, spec_mask,
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
        ctype_2 (int): Species of the central atom of the second local environment.
        etypes_2 (np.ndarray): Species of atoms in the second local
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

    ej1 = etypes_2[0]
    ej2 = etypes_2[1]
    bctype_1 = spec_mask[ctype_2]
    bctype_2 = spec_mask[ej1]
    bc3 = spec_mask[ej2]
    ttype = triplet_mask[nspec * nspec * bctype_1 + nspec*bctype_2 + bc3]
    ls = ls3[ttype]
    sig = sig3[ttype]
    r_cut = cutoff_3b

    return three_body_mc_en_force_jit(env1.bond_array_3, env1.ctype,
                                      env1.etypes,
                                      env1.cross_bond_inds,
                                      env1.cross_bond_dists,
                                      env1.triplet_counts,
                                      ctype_2, etypes_2,
                                      r1, r2, r12,
                                      d1, sig, ls, r_cut, cutoff_func) / 3


# @njit
# def three_body_mc_force_en_jit(bond_array_1, ctype_1, etypes_1,
#                                cross_bond_inds_1, cross_bond_dists_1,
#                                triplets_1,
#                                ctype_2, etypes_2,
#                                rj1, rj2, rj3,
#                                d1, sig, ls, r_cut, cutoff_func):
#     """3-body multi-element kernel between a force component and many local
#     energies on the grid.
#
#     Args:
#         bond_array_1 (np.ndarray): 3-body bond array of the first local
#             environment.
#         ctype_1 (int): Species of the central atom of the first local environment.
#         etypes_1 (np.ndarray): Species of atoms in the first local
#             environment.
#         cross_bond_inds_1 (np.ndarray): Two dimensional array whose row m
#             contains the indices of atoms n > m in the first local
#             environment that are within a distance r_cut of both atom n and
#             the central atom.
#         cross_bond_dists_1 (np.ndarray): Two dimensional array whose row m
#             contains the distances from atom m of atoms n > m in the first
#             local environment that are within a distance r_cut of both atom
#             n and the central atom.
#         triplets_1 (np.ndarray): One dimensional array of integers whose entry
#             m is the number of atoms in the first local environment that are
#             within a distance r_cut of atom m.
#         ctype_2 (int): Species of the central atom of the second local environment.
#         etypes_2 (np.ndarray): Species of atoms in the second local
#             environment.
#         rj1 (np.ndarray): matrix of the first edge length
#         rj2 (np.ndarray): matrix of the second edge length
#         rj12 (np.ndarray): matrix of the third edge length
#         d1 (int): Force component of the first environment (1=x, 2=y, 3=z).
#         sig (float): 3-body signal variance hyperparameter.
#         ls (float): 3-body length scale hyperparameter.
#         r_cut (float): 3-body cutoff radius.
#         cutoff_func (Callable): Cutoff function.
#
#     Returns:
#         float:
#             Value of the 3-body force/energy kernel.
#     """
#
#     kern = np.zeros_like(rj1, dtype=np.float64)
#
#     ei1 = etypes_2[0]
#     ei2 = etypes_2[1]
#
#     all_spec = [ctype_2, ei1, ei2]
#     if (ctype_1 not in all_spec):
#         return kern
#     all_spec.remove(ctype_1)
#
#     # pre-compute constants that appear in the inner loop
#     sig2 = sig * sig
#     ls1 = 1 / (2 * ls * ls)
#     ls2 = 1 / (ls * ls)
#
#     f1, fdi1 = cutoff_func(r_cut, ri1, ci1)
#
#     f2, fdi2 = cutoff_func(r_cut, ri2, ci2)
#     f3, fdi3 = cutoff_func(r_cut, ri3, ci3)
#     fi = f1 * f2 * f3
#     fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3
#     # del f1
#     # del f2
#     # del f3
#
#     for m in prange(bond_array_1.shape[0]):
#         ei1 = etypes_1[m]
#
#         two_spec = [all_spec[0], all_spec[1]]
#         if (ei1 in two_spec):
#             two_spec.remove(ei1)
#             one_spec = two_spec[0]
#
#             rj1 = bond_array_1[m, 0]
#             fj1, _ = cutoff_func(r_cut, rj1, 0)
#
#             for n in prange(triplets_1[m]):
#
#                 ind1 = cross_bond_inds_1[m, m + n + 1]
#                 ej2 = etypes_1[ind1]
#
#                 if (ej2 == one_spec):
#
#                     if (ei2 == ej2):
#                         r11 = ri1 - rj1
#                     if (ei2 == ej1):
#                         r12 = ri1 - rj2
#                     if (ei2 == ctype_2):
#                         r13 = ri1 - rj3
#
#                     rj2 = bond_array_1[ind1, 0]
#                     if (ei1 == ej2):
#                         r21 = ri2 - rj1
#                     if (ei1 == ej1):
#                         r22 = ri2 - rj2
#                     if (ei1 == ctype_2):
#                         r23 = ri2 - rj3
#                     cj2 = bond_array_1[ind1, d1]
#                     fj2, _ = cutoff_func(r_cut, rj2, 0)
#                     # del ri2
#
#                     rj3 = cross_bond_dists_1[m, m + n + 1]
#                     if (ctype_1 == ej2):
#                         r31 = ri3 - rj1
#                     if (ctype_1 == ej1):
#                         r32 = ri3 - rj2
#                     if (ctype_1 == ctype_2):
#                         r33 = ri3 - rj3
#                     fj3, _ = cutoff_func(r_cut, rj3, 0)
#                     # del ri3
#
#                     fj = fj1 * fj2 * fj3
#                     # del fj1
#                     # del fj2
#                     # del fj3
#
#                     if (ctype_1 == ctype_2):
#                         if (ei1 == ej1) and (ei2 == ej2):
#                             kern += three_body_en_helper(ci1, ci2, r11, r22,
#                                                          r33, fi, fj, fdi, ls1,
#                                                          ls2, sig2)
#                         if (ei1 == ej2) and (ei2 == ej1):
#                             kern += three_body_en_helper(ci1, ci2, r12, r21,
#                                                          r33, fi, fj, fdi, ls1,
#                                                          ls2, sig2)
#                     if (ctype_1 == ej1):
#                         if (ei1 == ej2) and (ei2 == ctype_2):
#                             kern += three_body_en_helper(ci1, ci2, r13, r21,
#                                                          r32, fi, fj, fdi, ls1,
#                                                          ls2, sig2)
#                         if (ei1 == ctype_2) and (ei2 == ej2):
#                             kern += three_body_en_helper(ci1, ci2, r11, r23,
#                                                          r32, fi, fj, fdi, ls1,
#                                                          ls2, sig2)
#                     if (ctype_1 == ej2):
#                         if (ei1 == ej1) and (ei2 == ctype_2):
#                             kern += three_body_en_helper(ci1, ci2, r13, r22,
#                                                          r31, fi, fj, fdi, ls1,
#                                                          ls2, sig2)
#                         if (ei1 == ctype_2) and (ei2 == ej1):
#                             kern += three_body_en_helper(ci1, ci2, r12, r23,
#                                                          r31, fi, fj, fdi, ls1,
#                                                          ls2, sig2)
#     return kern

@njit
def three_body_mc_en_force_jit(bond_array_1, ctype_1, etypes_1,
                               cross_bond_inds_1, cross_bond_dists_1,
                               triplets_1,
                               ctype_2, etypes_2,
                               rj1, rj2, rj3,
                               d1, sig, ls, r_cut, cutoff_func):
    """3-body multi-element kernel between a force component and many local
    energies on the grid.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        ctype_1 (int): Species of the central atom of the first local environment.
        etypes_1 (np.ndarray): Species of atoms in the first local
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
        ctype_2 (int): Species of the central atom of the second local environment.
        etypes_2 (np.ndarray): Species of atoms in the second local
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

    ej1 = etypes_2[0]
    ej2 = etypes_2[1]

    all_spec = [ctype_2, ej1, ej2]
    if (ctype_1 not in all_spec):
        return kern
    all_spec.remove(ctype_1)

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
        ei1 = etypes_1[m]

        two_spec = [all_spec[0], all_spec[1]]
        if (ei1 in two_spec):
            two_spec.remove(ei1)
            one_spec = two_spec[0]

            ri1 = bond_array_1[m, 0]
            ci1 = bond_array_1[m, d1]
            fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)

            for n in prange(triplets_1[m]):

                ind1 = cross_bond_inds_1[m, m + n + 1]
                ei2 = etypes_1[ind1]

                if (ei2 == one_spec):

                    if (ei2 == ej2):
                        r11 = ri1 - rj1
                    if (ei2 == ej1):
                        r12 = ri1 - rj2
                    if (ei2 == ctype_2):
                        r13 = ri1 - rj3

                    ri2 = bond_array_1[ind1, 0]
                    if (ei1 == ej2):
                        r21 = ri2 - rj1
                    if (ei1 == ej1):
                        r22 = ri2 - rj2
                    if (ei1 == ctype_2):
                        r23 = ri2 - rj3
                    ci2 = bond_array_1[ind1, d1]
                    fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                    # del ri2

                    ri3 = cross_bond_dists_1[m, m + n + 1]
                    if (ctype_1 == ej2):
                        r31 = ri3 - rj1
                    if (ctype_1 == ej1):
                        r32 = ri3 - rj2
                    if (ctype_1 == ctype_2):
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

                    if (ctype_1 == ctype_2):
                        if (ei1 == ej1) and (ei2 == ej2):
                            kern += three_body_en_helper(ci1, ci2, r11, r22,
                                                         r33, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                        if (ei1 == ej2) and (ei2 == ej1):
                            kern += three_body_en_helper(ci1, ci2, r12, r21,
                                                         r33, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                    if (ctype_1 == ej1):
                        if (ei1 == ej2) and (ei2 == ctype_2):
                            kern += three_body_en_helper(ci1, ci2, r13, r21,
                                                         r32, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                        if (ei1 == ctype_2) and (ei2 == ej2):
                            kern += three_body_en_helper(ci1, ci2, r11, r23,
                                                         r32, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                    if (ctype_1 == ej2):
                        if (ei1 == ej1) and (ei2 == ctype_2):
                            kern += three_body_en_helper(ci1, ci2, r13, r22,
                                                         r31, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                        if (ei1 == ctype_2) and (ei2 == ej1):
                            kern += three_body_en_helper(ci1, ci2, r12, r23,
                                                         r31, fi, fj, fdi, ls1,
                                                         ls2, sig2)
    return kern

@njit
def three_body_mc_en_jit(bond_array_1, ctype_1, etypes_1,
                         cross_bond_inds_1,
                         cross_bond_dists_1,
                         triplets_1,
                         ctype_2, etypes_2,
                         rj1, rj2, rj3,
                         sig, ls, r_cut, cutoff_func):
    """3-body multi-element kernel between two local energies accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        ctype_1 (int): Species of the central atom of the first local environment.
        etypes_1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
            environment.
        ctype_2 (int): Species of the central atom of the second local environment.
        etypes_2 (np.ndarray): Species of atoms in the second local
            environment.
        cross_bond_inds_1 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the first local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_inds_2 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the second local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_dists_1 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the first
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        cross_bond_dists_2 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the second
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        triplets_1 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the first local environment that are
            within a distance r_cut of atom m.
        triplets_2 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the second local environment that are
            within a distance r_cut of atom m.
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 3-body local energy kernel.
    """

    kern = np.zeros_like(rj1, dtype=np.float64)

    ej1 = etypes_2[0]
    ej2 = etypes_2[1]

    all_spec = [ctype_2, ej1, ej2]
    if (ctype_1 not in all_spec):
        return kern
    all_spec.remove(ctype_1)

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls2 = 1 / (2 * ls * ls)


    f1, _ = cutoff_func(r_cut, rj1, 0)
    f2, _ = cutoff_func(r_cut, rj2, 0)
    f3, _ = cutoff_func(r_cut, rj3, 0)
    fj = f1 * f2 * f3

    for m in prange(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        fi1, _ = cutoff_func(r_cut, ri1, 0)
        ei1 = etypes_1[m]

        two_spec = [all_spec[0], all_spec[1]]
        if (ei1 in two_spec):
            two_spec.remove(ei1)
            one_spec = two_spec[0]

            for n in prange(triplets_1[m]):
                ei2 = etypes_1[ind1]
                if (ei2 == one_spec):

                    if (ei2 == ej2):
                        r11 = ri1 - rj1
                    if (ei2 == ej1):
                        r12 = ri1 - rj2
                    if (ei2 == ctype_2):
                        r13 = ri1 - rj3

                    ri2 = bond_array_1[ind1, 0]
                    if (ei1 == ej2):
                        r21 = ri2 - rj1
                    if (ei1 == ej1):
                        r22 = ri2 - rj2
                    if (ei1 == ctype_2):
                        r23 = ri2 - rj3
                    ci2 = bond_array_1[ind1, d1]
                    fi2, _ = cutoff_func(r_cut, ri2, ci2)
                    # del ri2

                    ri3 = cross_bond_dists_1[m, m + n + 1]
                    if (ctype_1 == ej2):
                        r31 = ri3 - rj1
                    if (ctype_1 == ej1):
                        r32 = ri3 - rj2
                    if (ctype_1 == ctype_2):
                        r33 = ri3 - rj3
                    fi3, _ = cutoff_func(r_cut, ri3, 0)

                    fi = fi1 * fi2 * fi3

                    if (ctype_1 == ctype_2):
                        if (ei1 == ej1) and (ei2 == ej2):
                            C1 = r11 * r11 + r22 * r22 + r33 * r33
                            kern += sig2 * np.exp(-C1 * ls2) * fi * fj
                        if (ei1 == ej2) and (ei2 == ej1):
                            C3 = r12 * r12 + r21 * r21 + r33 * r33
                            kern += sig2 * np.exp(-C3 * ls2) * fi * fj
                    if (ctype_1 == ej1):
                        if (ei1 == ej2) and (ei2 == ctype_2):
                            C5 = r13 * r13 + r21 * r21 + r32 * r32
                            kern += sig2 * np.exp(-C5 * ls2) * fi * fj
                        if (ei1 == ctype_2) and (ei2 == ej2):
                            C2 = r11 * r11 + r23 * r23 + r32 * r32
                            kern += sig2 * np.exp(-C2 * ls2) * fi * fj
                    if (ctype_1 == ej2):
                        if (ei1 == ej1) and (ei2 == ctype_2):
                            C6 = r13 * r13 + r22 * r22 + r31 * r31
                            kern += sig2 * np.exp(-C6 * ls2) * fi * fj
                        if (ei1 == ctype_2) and (ei2 == ej1):
                            C4 = r12 * r12 + r23 * r23 + r31 * r31
                            kern += sig2 * np.exp(-C4 * ls2) * fi * fj

    return kern


@njit
def three_body_en_helper(ci1, ci2, r11, r22, r33, fi, fj, fdi, ls1, ls2, sig2):

    B = r11 * ci1 + r22 * ci2
    D = r11 * r11 + r22 * r22 + r33 * r33
    return -sig2 * np.exp(- D * ls1) * ( B * ls2 * fi * fj + fdi * fj)

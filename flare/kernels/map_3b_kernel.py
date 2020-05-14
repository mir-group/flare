"""Multi-element 2-, 3-, and 2+3-body kernels that restrict all signal
variance hyperparameters to a single value."""
import numpy as np
from numba import njit
from math import exp, floor
from flare.env import AtomicEnvironment
import flare.cutoffs as cf
from typing import Callable

from flare.kernels.utils import from_mask_to_args, from_grad_to_mask
from flare.gp_algebra import _global_training_data, _global_training_labels


def three_body_mc_force_en(env1: AtomicEnvironment, r1, r2, r12, c2, etypes2,
                           d1: int, hyps: 'ndarray', cutoffs: 'ndarray',
                           cutoff_func: Callable = cf.quadratic_cutoff) \
        -> float:
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    return three_body_mc_force_en_jit(env1.bond_array_3, env1.ctype,
                                      env1.etypes,
                                      env1.cross_bond_inds,
                                      env1.cross_bond_dists,
                                      env1.triplet_counts,
                                      c2, etypes2,
                                      r1, r2, r12,
                                      d1, sig, ls, r_cut, cutoff_func) / 3

def three_body_mc_force_en_sephyps(env1, r1, r2, r12, c2, etypes2,
                                   d1, cutoffs, nspec, spec_mask,
                                   nbond, bond_mask, ntriplet, triplet_mask,
                                   sig2, ls2, sig3, ls3,
                                   cutoff_func=cf.quadratic_cutoff) -> float:

    ej1 = etypes2[0]
    ej2 = etypes2[1]
    bc1 = spec_mask[c2]
    bc2 = spec_mask[ej1]
    bc3 = spec_mask[ej2]
    ttype = triplet_mask[nspec * nspec * bc1 + nspec*bc2 + bc3]
    ls = ls3[ttype]
    sig = sig3[ttype]
    r_cut = cutoffs[1]

    return three_body_mc_force_en_jit(env1.bond_array_3, env1.ctype,
                                      env1.etypes,
                                      env1.cross_bond_inds,
                                      env1.cross_bond_dists,
                                      env1.triplet_counts,
                                      c2, etypes2,
                                      r1, r2, r12,
                                      d1, sig, ls, r_cut, cutoff_func) / 3


@njit(parallel=True)
def three_body_mc_force_en_jit(bond_array_1, c1, etypes1,
                               cross_bond_inds_1, cross_bond_dists_1,
                               triplets_1,
                               c2, etypes2,
                               rj1, rj2, rj3,
                               d1, sig, ls, r_cut, cutoff_func):
    """3-body multi-element kernel between a force component and a local
    energy accelerated with Numba.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
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

    for m in range(bond_array_1.shape[0]):
        ei1 = etypes1[m]

        two_spec = [all_spec[0], all_spec[1]]
        if (ei1 in two_spec):
            two_spec.remove(ei1)
            one_spec = two_spec[0]

            ri1 = bond_array_1[m, 0]
            ci1 = bond_array_1[m, d1]
            fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)

            for n in range(triplets_1[m]):

                ind1 = cross_bond_inds_1[m, m + n + 1]
                ei2 = etypes1[ind1]

                if (ei2 == one_spec):

                    ri2 = bond_array_1[ind1, 0]
                    ci2 = bond_array_1[ind1, d1]
                    fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)

                    ri3 = cross_bond_dists_1[m, m + n + 1]
                    fi3, _ = cutoff_func(r_cut, ri3, 0)

                    fi = fi1 * fi2 * fi3
                    fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

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
def three_body_en_helper(ci1, ci2, r11, r22, r33, fi, fj, fdi, ls1, ls2, sig2):

    B = r11 * ci1 + r22 * ci2
    D = r11 * r11 + r22 * r22 + r33 * r33
    E = np.exp(-D * ls1)
    F = E * B * ls2
    G = -F * fi * fj
    H = -E * fdi * fj
    I = sig2 * (G + H)
    return I

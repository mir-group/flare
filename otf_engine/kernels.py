import numpy as np
import math
from math import exp
from math import factorial
from itertools import combinations
from itertools import permutations
from numba import njit
from struc import Structure
from env import ChemicalEnvironment
import gp
import struc
import env
import time

# -----------------------------------------------------------------------------
#               kernels and gradients acting on environment objects
# -----------------------------------------------------------------------------


# get n body single component kernel between two environments
def energy_force_sc(env1, env2, bodies, d1, d2, hyps, cutoffs=None):
    combs = get_comb_array(env1.bond_array.shape[0], bodies-1)
    perms = get_perm_array(env2.bond_array.shape[0], bodies-1)

    return energy_force_jit_sc(env1.bond_array, env1.cross_bond_dists, combs,
                               env2.bond_array, env2.cross_bond_dists, perms,
                               d1, d2, hyps)


def combo_kernel_sc(env1: ChemicalEnvironment, env2: ChemicalEnvironment,
                    bodies_array: list, d1: int, d2: int,
                    hyps: np.ndarray, cutoffs: np.ndarray) -> float:
    kern = 0
    for count, (bodies, cutoff) in enumerate(zip(bodies_array, cutoffs)):
        hyp_ind = 2 * count
        hyps_curr = hyps[hyp_ind:hyp_ind+2]
        kern += n_body_sc_cutoff(env1, env2, bodies, d1, d2, hyps_curr,
                                 cutoff)
    return kern


def combo_kernel_sc_grad(env1: ChemicalEnvironment, env2: ChemicalEnvironment,
                         bodies_array: list, d1: int, d2: int,
                         hyps: np.ndarray, cutoffs: np.ndarray) -> float:
    kern = 0
    kern_grad = np.zeros(hyps.size)
    for count, (bodies, cutoff) in enumerate(zip(bodies_array, cutoffs)):
        hyp_ind = 2 * count
        hyps_curr = hyps[hyp_ind:hyp_ind+2]
        kern_curr = n_body_sc_cutoff_grad(env1, env2, bodies, d1, d2,
                                          hyps_curr, cutoff)
        kern += kern_curr[0]
        kern_grad[hyp_ind:hyp_ind+2] = kern_curr[1]
    return kern, kern_grad


def n_body_sc_cutoff(env1: ChemicalEnvironment, env2: ChemicalEnvironment,
                     bodies: int, d1: int, d2: int, hyps: np.ndarray,
                     cutoff: float) -> float:
    bond_array_1, cross_bond_1 = \
        get_restricted_arrays(env1.bond_array, env1.cross_bond_dists, cutoff)
    bond_array_2, cross_bond_2 = \
        get_restricted_arrays(env2.bond_array, env2.cross_bond_dists, cutoff)
    combs = get_comb_array(bond_array_1.shape[0], bodies-1)
    perms = get_perm_array(bond_array_2.shape[0], bodies-1)
    return n_body_jit_sc(bond_array_1, cross_bond_1, combs,
                         bond_array_2, cross_bond_2, perms,
                         d1, d2, hyps)


def n_body_sc_cutoff_grad(env1, env2, bodies, d1, d2, hyps, cutoff):
    bond_array_1, cross_bond_1 = \
        get_restricted_arrays(env1.bond_array, env1.cross_bond_dists, cutoff)
    bond_array_2, cross_bond_2 = \
        get_restricted_arrays(env2.bond_array, env2.cross_bond_dists, cutoff)
    combs = get_comb_array(bond_array_1.shape[0], bodies-1)
    perms = get_perm_array(bond_array_2.shape[0], bodies-1)
    return n_body_sc_grad_array(bond_array_1, cross_bond_1, combs,
                                bond_array_2, cross_bond_2, perms,
                                d1, d2, hyps)


# get n body single component kernel between two environments
def n_body_sc(env1, env2, bodies, d1, d2, hyps, cutoffs=None):
    combs = get_comb_array(env1.bond_array.shape[0], bodies-1)
    perms = get_perm_array(env2.bond_array.shape[0], bodies-1)

    return n_body_jit_sc(env1.bond_array, env1.cross_bond_dists, combs,
                         env2.bond_array, env2.cross_bond_dists, perms,
                         d1, d2, hyps)


def n_body_sc_grad(env1, env2, bodies, d1, d2, hyps, cutoffs=None):
    combs = get_comb_array(env1.bond_array.shape[0], bodies-1)
    perms = get_perm_array(env2.bond_array.shape[0], bodies-1)

    return n_body_sc_grad_array(env1.bond_array, env1.cross_bond_dists, combs,
                                env2.bond_array, env2.cross_bond_dists, perms,
                                d1, d2, hyps)


# get three body kernel between two environments
def three_body(env1, env2, d1, d2, sig, ls):
    return three_body_jit(env1.bond_array, env1.bond_types,
                          env1.cross_bond_dists, env1.cross_bond_types,
                          env2.bond_array, env2.bond_types,
                          env2.cross_bond_dists, env2.cross_bond_types,
                          d1, d2, sig, ls)


# get two body kernel between two environments
def two_body(env1, env2, d1, d2, sig, ls):
    return two_body_jit(env1.bond_array, env1.bond_types,
                        env2.bond_array, env2.bond_types,
                        d1, d2, sig, ls)


# -----------------------------------------------------------------------------
#                               kernel gradients
# -----------------------------------------------------------------------------


@njit
def n_body_sc_grad_array(bond_array_1, cross_bond_dists_1, combinations,
                         bond_array_2, cross_bond_dists_2, permutations,
                         d1, d2, hyps):

    sig = hyps[0]
    ls = hyps[1]

    kern = 0
    sig_derv = 0
    ls_derv = 0

    for m in range(combinations.shape[0]):
        comb = combinations[m]
        for n in range(permutations.shape[0]):
            perm = permutations[n]
            A_cp = 0
            B_cp_1 = 0
            B_cp_2 = 0
            C_cp = 0

            for q, (c_ind, p_ind) in enumerate(zip(comb, perm)):
                rdiff = bond_array_1[c_ind, 0] - bond_array_2[p_ind, 0]
                coord1 = bond_array_1[c_ind, d1]
                coord2 = bond_array_2[p_ind, d2]

                A_cp += coord1 * coord2
                B_cp_1 += rdiff * coord1
                B_cp_2 += rdiff * coord2
                C_cp += rdiff * rdiff

                for c_ind_2, p_ind_2 in zip(comb[q+1:], perm[q+1:]):
                    cb_diff = cross_bond_dists_1[c_ind, c_ind_2] - \
                        cross_bond_dists_2[p_ind, p_ind_2]
                    C_cp += cb_diff * cb_diff

            B_cp = B_cp_1 * B_cp_2

            kern += (sig*sig / (ls**4)) * (A_cp * ls * ls - B_cp) * \
                exp(-C_cp / (2 * ls * ls))

            sig_derv += (2*sig / (ls**4)) * (A_cp * ls * ls - B_cp) * \
                exp(-C_cp / (2 * ls * ls))

            ls_derv += ((sig*sig)/(ls**7)) * \
                (-B_cp*C_cp+(4*B_cp+A_cp*C_cp)*ls*ls-2*A_cp*ls**4) * \
                exp(-C_cp / (2 * ls * ls))

    kern_grad = np.array([sig_derv, ls_derv])

    return kern, kern_grad


def two_body_grad_from_env(bond_array_1, bond_types_1, bond_array_2,
                           bond_types_2, d1, d2, hyps):
    sig = hyps[0]
    ls = hyps[1]
    S = sig * sig
    L = 1 / (ls * ls)
    sig_conv = 2 * sig
    ls_conv = -2 / (ls * ls * ls)

    kern = 0
    sig_derv = 0
    ls_derv = 0

    x1_len = len(bond_types_1)
    x2_len = len(bond_types_2)

    for m in range(x1_len):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]
        typ1 = bond_types_1[m]

        for n in range(x2_len):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]
            typ2 = bond_types_2[n]

            # check that bonds match
            if typ1 == typ2:
                rr = (r1-r2)*(r1-r2)
                kern += S*L*exp(-0.5*L*rr)*coord1*coord2*(1-L*rr)
                sig_derv += L*exp(-0.5*L*rr)*coord1*coord2*(1-L*rr) * sig_conv
                ls_derv += 0.5*coord1*coord2*S*exp(-L*rr/2) * \
                    (2+L*rr*(-5+L*rr))*ls_conv

    kern_grad = np.array([sig_derv, ls_derv])

    return kern, kern_grad

# -----------------------------------------------------------------------------
#           kernels acting on numpy arrays (can be jitted)
# -----------------------------------------------------------------------------


# single component n body energy/force kernel
# bond array 1: force environment
# bond array 2: local energy environment
@njit
def energy_force_jit_sc(bond_array_1, cross_bond_dists_1, combinations,
                        bond_array_2, cross_bond_dists_2, permutations,
                        d1, d2, hyps):
    sig = hyps[0]
    ls = hyps[1]
    kern = 0

    for m in range(combinations.shape[0]):
        comb = combinations[m]
        for n in range(permutations.shape[0]):
            perm = permutations[n]
            B_cp_1 = 0
            C_cp = 0

            for q, (c_ind, p_ind) in enumerate(zip(comb, perm)):
                rdiff = bond_array_1[c_ind, 0] - bond_array_2[p_ind, 0]
                coord1 = bond_array_1[c_ind, d1]

                B_cp_1 += -rdiff * coord1
                C_cp += rdiff * rdiff

                for c_ind_2, p_ind_2 in zip(comb[q+1:], perm[q+1:]):
                    cb_diff = cross_bond_dists_1[c_ind, c_ind_2] - \
                        cross_bond_dists_2[p_ind, p_ind_2]
                    C_cp += cb_diff * cb_diff

            kern += (sig*sig / (ls*ls)) * B_cp_1 * \
                exp(-C_cp / (2 * ls * ls))

    return kern


# single component n body kernel
@njit
def n_body_jit_sc(bond_array_1, cross_bond_dists_1, combinations,
                  bond_array_2, cross_bond_dists_2, permutations,
                  d1, d2, hyps):
    sig = hyps[0]
    ls = hyps[1]
    kern = 0

    for m in range(combinations.shape[0]):
        comb = combinations[m]
        for n in range(permutations.shape[0]):
            perm = permutations[n]
            A_cp = 0
            B_cp_1 = 0
            B_cp_2 = 0
            C_cp = 0

            for q, (c_ind, p_ind) in enumerate(zip(comb, perm)):
                rdiff = bond_array_1[c_ind, 0] - bond_array_2[p_ind, 0]
                coord1 = bond_array_1[c_ind, d1]
                coord2 = bond_array_2[p_ind, d2]

                A_cp += coord1 * coord2
                B_cp_1 += rdiff * coord1
                B_cp_2 += rdiff * coord2
                C_cp += rdiff * rdiff

                for c_ind_2, p_ind_2 in zip(comb[q+1:], perm[q+1:]):
                    cb_diff = cross_bond_dists_1[c_ind, c_ind_2] - \
                        cross_bond_dists_2[p_ind, p_ind_2]
                    C_cp += cb_diff * cb_diff

            B_cp = B_cp_1 * B_cp_2

            kern += (sig*sig / (ls**4)) * (A_cp * ls * ls - B_cp) * \
                exp(-C_cp / (2 * ls * ls))

    return kern


# jit function that computes three body kernel
@njit
def three_body_jit(bond_array_1, bond_types_1,
                   cross_bond_dists_1, cross_bond_types_1,
                   bond_array_2, bond_types_2,
                   cross_bond_dists_2, cross_bond_types_2,
                   d1, d2, sig, ls):
    d = sig * sig / (ls * ls * ls * ls)
    e = ls * ls
    f = 1 / (2 * ls * ls)
    kern = 0

    x1_len = len(bond_types_1)
    x2_len = len(bond_types_2)

    # loop over triplets in environment 1
    for m in range(x1_len):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]

        for n in range(m + 1, x1_len):
            ri2 = bond_array_1[n, 0]
            ci2 = bond_array_1[n, d1]
            ri3 = cross_bond_dists_1[m, n]
            t1 = cross_bond_types_1[m, n]

            # loop over triplets in environment 2
            for p in range(x2_len):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]

                for q in range(x2_len):
                    if p == q:  # consider distinct bonds
                        continue

                    # get triplet types
                    t2 = cross_bond_types_2[p, q]

                    # proceed if triplet types match
                    if t1 == t2:
                        rj2 = bond_array_2[q, 0]
                        cj2 = bond_array_2[q, d2]
                        rj3 = cross_bond_dists_2[p, q]

                        r11 = ri1 - rj1
                        r22 = ri2 - rj2
                        r33 = ri3 - rj3

                        # add to kernel
                        kern += (e * (ci1 * cj1 + ci2 * cj2) -
                                 (r11 * ci1 + r22 * ci2) *
                                 (r11 * cj1 + r22 * cj2)) * \
                            d * exp(-f * (r11 * r11 + r22 * r22 +
                                          r33 * r33))

    return kern


# jit function that computes two body kernel
@njit
def two_body_jit(bond_array_1, bond_types_1, bond_array_2,
                 bond_types_2, d1, d2, sig, ls):
    d = sig * sig / (ls * ls * ls * ls)
    e = ls * ls
    f = 1 / (2 * ls * ls)
    kern = 0

    x1_len = len(bond_types_1)
    x2_len = len(bond_types_2)

    for m in range(x1_len):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]
        typ1 = bond_types_1[m]

        for n in range(x2_len):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]
            typ2 = bond_types_2[n]

            # check that bonds match
            if typ1 == typ2:
                rr = (r1 - r2) * (r1 - r2)
                kern += d * exp(-f * rr) * coord1 * coord2 * (e - rr)

    return kern


# -----------------------------------------------------------------------------
#                               helper functions
# -----------------------------------------------------------------------------


# count combinations
def get_comb_no(N, M):
    if M > N:
        return 0
    return int(factorial(N) / (factorial(M) * factorial(N - M)))


# count permutations
def get_perm_no(N, M):
    if M > N:
        return 0
    return int(factorial(N) / factorial(N - M))


# get combination array
def get_comb_array(list_len, tuple_size):
    lst = np.arange(0, list_len)

    no_combs = get_comb_no(list_len, tuple_size)
    comb_store = np.zeros([no_combs, tuple_size])

    for count, combo in enumerate(combinations(lst, tuple_size)):
        comb_store[count, :] = combo

    # convert to ints
    comb_store = comb_store.astype(int)

    return comb_store


# get permutation array
def get_perm_array(list_len, tuple_size):
    lst = np.arange(0, list_len)

    no_perms = get_perm_no(list_len, tuple_size)
    perm_store = np.zeros([no_perms, tuple_size])

    for count, perm in enumerate(permutations(lst, tuple_size)):
        perm_store[count, :] = perm

    # convert to ints
    perm_store = perm_store.astype(int)

    return perm_store


@njit
def get_cutoff_index(bond_array, cutoff):
    for count, dist in enumerate(bond_array[:, 0]):
        if dist > cutoff:
            return count
    return count+1


@njit
def get_restricted_arrays(bond_array, cross_bond_array, cutoff):
    cutoff_ind = get_cutoff_index(bond_array, cutoff)
    restricted_bond_array = bond_array[0:cutoff_ind, :]
    restricted_cross_bond_array = cross_bond_array[0:cutoff_ind, 0:cutoff_ind]
    return restricted_bond_array, restricted_cross_bond_array


if __name__ == '__main__':
    pass

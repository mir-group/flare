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


def n_body_mc_grad(env1, env2, bodies, d1, d2, hyps, cutoffs=None):
    combs = get_comb_array(env1.bond_array.shape[0], bodies-1)
    perms = get_perm_array(env2.bond_array.shape[0], bodies-1)
    sig = hyps[0]
    ls = hyps[1]
    ICM_vec = hyps[2: len(hyps-1)]

    ICM_array = get_ICM_array_from_vector(ICM_vec, env1.structure.nos)

    return n_body_jit_mc_grad_array(env1.bond_array, env1.cross_bond_dists,
                                    combs, env1.ctyp, env1.etyps,
                                    env2.bond_array, env2.cross_bond_dists,
                                    perms, env2.ctyp, env2.etyps,
                                    d1, d2, sig, ls, ICM_array)


# get n body multi component kernel between two environments
def n_body_mc(env1, env2, bodies, d1, d2, hyps, cutoffs=None):
    combs = get_comb_array(env1.bond_array.shape[0], bodies-1)
    perms = get_perm_array(env2.bond_array.shape[0], bodies-1)
    sig = hyps[0]
    ls = hyps[1]
    ICM_vec = hyps[2: len(hyps-1)]
    ICM_array = get_ICM_array_from_vector(ICM_vec, env1.structure.nos)

    return n_body_jit_mc(env1.bond_array, env1.cross_bond_dists, combs,
                         env1.ctyp, env1.etyps,
                         env2.bond_array, env2.cross_bond_dists, perms,
                         env2.ctyp, env2.etyps,
                         d1, d2, sig, ls, ICM_array)


# get n body single component kernel between two environments
def energy_sc(env1, env2, bodies, hyps, cutoffs=None):
    combs = get_comb_array(env1.bond_array.shape[0], bodies-1)
    perms = get_perm_array(env2.bond_array.shape[0], bodies-1)

    return energy_jit_sc(env1.bond_array, env1.cross_bond_dists, combs,
                         env2.bond_array, env2.cross_bond_dists, perms,
                         hyps)


# get n body single component kernel between two environments
def energy_force_sc(env1, env2, bodies, d1, hyps, cutoffs=None):
    combs = get_comb_array(env1.bond_array.shape[0], bodies-1)
    perms = get_perm_array(env2.bond_array.shape[0], bodies-1)

    return energy_force_jit_sc(env1.bond_array, env1.cross_bond_dists, combs,
                               env2.bond_array, env2.cross_bond_dists, perms,
                               d1, hyps)


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
    return three_body_jit(env1.bond_array, env1.cross_bond_dists,
                          env2.bond_array, env2.cross_bond_dists,
                          d1, d2, sig, ls)


# get two body kernel between two environments
def two_body(env1, env2, d1, d2, sig, ls):
    return two_body_jit(env1.bond_array, env2.bond_array, d1, d2, sig, ls)


# -----------------------------------------------------------------------------
#                               kernel gradients
# -----------------------------------------------------------------------------

# multi component n body kernel
@njit
def n_body_jit_mc_grad_array(bond_array_1, cross_bond_dists_1, combinations,
                             central_species_1, environment_species_1,
                             bond_array_2, cross_bond_dists_2, permutations,
                             central_species_2, environment_species_2,
                             d1, d2, sig, ls, ICM_array):
    kern = 0
    sig_derv = 0
    ls_derv = 0
    nos = ICM_array.shape[0]
    no_ICM_param = int(nos*(nos-1)/2)
    ICM_derv = np.zeros(no_ICM_param)
    ICM_coeff_start = ICM_array[central_species_1, central_species_2]
    kern_grad = np.zeros(no_ICM_param+2)

    for m in range(combinations.shape[0]):
        comb = combinations[m]
        for n in range(permutations.shape[0]):
            perm = permutations[n]
            A_cp = 0
            B_cp_1 = 0
            B_cp_2 = 0
            C_cp = 0

            ICM_coeff = ICM_coeff_start
            ICM_counter = np.zeros((nos, nos))

            for q, (c_ind, p_ind) in enumerate(zip(comb, perm)):
                rdiff = bond_array_1[c_ind, 0] - bond_array_2[p_ind, 0]
                coord1 = bond_array_1[c_ind, d1]
                coord2 = bond_array_2[p_ind, d2]

                A_cp += coord1 * coord2
                B_cp_1 += rdiff * coord1
                B_cp_2 += rdiff * coord2
                C_cp += rdiff * rdiff

                c_typ = environment_species_1[c_ind]
                p_typ = environment_species_2[p_ind]
                ICM_coeff *= ICM_array[c_typ, p_typ]
                ICM_counter[c_typ, p_typ] += 1
                ICM_counter[p_typ, c_typ] += 1

                # account for cross bonds
                for c_ind_2, p_ind_2 in zip(comb[q+1:], perm[q+1:]):
                    cb_diff = cross_bond_dists_1[c_ind, c_ind_2] - \
                        cross_bond_dists_2[p_ind, p_ind_2]
                    C_cp += cb_diff * cb_diff

            B_cp = B_cp_1 * B_cp_2

            # update kernel
            kern += (sig*sig / (ls**4)) * (A_cp * ls * ls - B_cp) * \
                exp(-C_cp / (2 * ls * ls)) * ICM_coeff

            # update sig and ls derivatives
            sig_derv += (2*sig / (ls**4)) * (A_cp * ls * ls - B_cp) * \
                exp(-C_cp / (2 * ls * ls)) * ICM_coeff

            ls_derv += ((sig*sig)/(ls**7)) * \
                (-B_cp*C_cp+(4*B_cp+A_cp*C_cp)*ls*ls-2*A_cp*ls**4) * \
                exp(-C_cp / (2 * ls * ls)) * ICM_coeff

            # update ICM derivatives
            ICM_count = 0
            for ICM_ind_1 in range(nos):
                for ICM_ind_2 in range(ICM_ind_1+1, nos):
                    ICM_derv[ICM_count] += \
                        (sig*sig / (ls**4)) * (A_cp * ls * ls - B_cp) * \
                        exp(-C_cp / (2 * ls * ls)) * \
                        ICM_coeff * ICM_counter[ICM_ind_1, ICM_ind_2] / \
                        ICM_array[ICM_ind_1, ICM_ind_2]
                    ICM_count += 1

    kern_grad[0] = sig_derv
    kern_grad[1] = ls_derv
    kern_grad[2:] = ICM_derv
    return kern, kern_grad


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


# multi component n body kernel
@njit
def n_body_jit_mc(bond_array_1, cross_bond_dists_1, combinations,
                  central_species_1, environment_species_1,
                  bond_array_2, cross_bond_dists_2, permutations,
                  central_species_2, environment_species_2,
                  d1, d2, sig, ls, ICM_array):
    kern = 0
    ICM_coeff_start = ICM_array[central_species_1, central_species_2]

    for m in range(combinations.shape[0]):
        comb = combinations[m]
        for n in range(permutations.shape[0]):
            perm = permutations[n]
            A_cp = 0
            B_cp_1 = 0
            B_cp_2 = 0
            C_cp = 0

            ICM_coeff = ICM_coeff_start

            for q, (c_ind, p_ind) in enumerate(zip(comb, perm)):
                rdiff = bond_array_1[c_ind, 0] - bond_array_2[p_ind, 0]
                coord1 = bond_array_1[c_ind, d1]
                coord2 = bond_array_2[p_ind, d2]

                A_cp += coord1 * coord2
                B_cp_1 += rdiff * coord1
                B_cp_2 += rdiff * coord2
                C_cp += rdiff * rdiff

                c_typ = environment_species_1[c_ind]
                p_typ = environment_species_2[p_ind]
                ICM_coeff *= ICM_array[c_typ, p_typ]

                for c_ind_2, p_ind_2 in zip(comb[q+1:], perm[q+1:]):
                    cb_diff = cross_bond_dists_1[c_ind, c_ind_2] - \
                        cross_bond_dists_2[p_ind, p_ind_2]
                    C_cp += cb_diff * cb_diff

            B_cp = B_cp_1 * B_cp_2

            kern += (sig*sig / (ls**4)) * (A_cp * ls * ls - B_cp) * \
                exp(-C_cp / (2 * ls * ls)) * ICM_coeff

    return kern


# single component n body energy kernel
@njit
def energy_jit_sc(bond_array_1, cross_bond_dists_1, combinations,
                  bond_array_2, cross_bond_dists_2, permutations,
                  hyps):
    sig = hyps[0]
    ls = hyps[1]
    kern = 0

    for m in range(combinations.shape[0]):
        comb = combinations[m]
        for n in range(permutations.shape[0]):
            perm = permutations[n]
            C_cp = 0

            for q, (c_ind, p_ind) in enumerate(zip(comb, perm)):
                rdiff = bond_array_1[c_ind, 0] - bond_array_2[p_ind, 0]
                C_cp += rdiff * rdiff

                for c_ind_2, p_ind_2 in zip(comb[q+1:], perm[q+1:]):
                    cb_diff = cross_bond_dists_1[c_ind, c_ind_2] - \
                        cross_bond_dists_2[p_ind, p_ind_2]
                    C_cp += cb_diff * cb_diff

            kern += sig * sig * exp(-C_cp / (2 * ls * ls))

    return kern


# single component n body energy/force kernel
# bond array 1: force environment
# bond array 2: local energy environment
@njit
def energy_force_jit_sc(bond_array_1, cross_bond_dists_1, combinations,
                        bond_array_2, cross_bond_dists_2, permutations,
                        d1, hyps):
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
def three_body_jit(bond_array_1, cross_bond_dists_1,
                   bond_array_2, cross_bond_dists_2,
                   d1, d2, sig, ls):
    d = sig * sig / (ls * ls * ls * ls)
    e = ls * ls
    f = 1 / (2 * ls * ls)
    kern = 0

    x1_len = bond_array_1.shape[0]
    x2_len = bond_array_2.shape[0]

    # loop over triplets in environment 1
    for m in range(x1_len):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]

        for n in range(m + 1, x1_len):
            ri2 = bond_array_1[n, 0]
            ci2 = bond_array_1[n, d1]
            ri3 = cross_bond_dists_1[m, n]

            # loop over triplets in environment 2
            for p in range(x2_len):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]

                for q in range(x2_len):
                    if p == q:  # consider distinct bonds
                        continue

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
def two_body_jit(bond_array_1, bond_array_2, d1, d2, sig, ls):
    d = sig * sig / (ls * ls * ls * ls)
    e = ls * ls
    f = 1 / (2 * ls * ls)
    kern = 0

    x1_len = bond_array_1.shape[0]
    x2_len = bond_array_2.shape[0]

    for m in range(x1_len):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]

        for n in range(x2_len):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]

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


def get_ICM_array_from_vector(ICM_vector, nos):
    ICM_array = np.empty([nos, nos])

    count = 0
    for m in range(nos):
        for n in range(m, nos):
            if m == n:
                ICM_array[m, n] = 1
            else:
                ICM_array[m, n] = ICM_vector[count]
                ICM_array[n, m] = ICM_vector[count]
                count += 1
    return ICM_array

if __name__ == '__main__':
    # make environment 1
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5]))

    positions_1 = [np.array([0, 0, 0]),
                   np.array([0.1, 0.2, 0.3]),
                   np.array([0.15, 0.25, 0.35])]
    species_1 = ['A', 'B', 'C']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
    env1 = env.ChemicalEnvironment(test_structure_1, atom_1)

    # make environment 2
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5]))

    positions_1 = [np.array([0.08, 0.31, 0.41]),
                   np.array([0.05, 0.21, 0.39]),
                   np.array([0, 0, 0])]
    species_1 = ['A', 'B', 'C']
    atom_1 = 2
    test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
    env2 = env.ChemicalEnvironment(test_structure_1, atom_1)

    bodies = 3
    d1 = 1
    d2 = 1
    hyps = np.array([1, 1, 1])
    sc_grad = n_body_sc_grad(env1, env2, bodies, d1, d2, hyps, cutoffs=None)

    hyps_mc = np.array([1, 1, 1, 1, 1, 1])
    mc_grad = n_body_mc_grad(env1, env2, bodies, d1, d2, hyps_mc)
    assert(np.isclose(sc_grad[0], mc_grad[0]))
    assert(np.isclose(sc_grad[1][0], mc_grad[1][0]))
    assert(np.isclose(sc_grad[1][1], mc_grad[1][1]))

    # finite difference test
    delta = 1e-8
    hyps_mc_delta_1 = np.array([1, 1, 1+delta, 1, 1, 1])
    hyps_mc_delta_2 = np.array([1, 1, 1, 1+delta, 1, 1])
    hyps_mc_delta_3 = np.array([1, 1, 1, 1, 1+delta, 1])
    mc_grad_delta_1 = n_body_mc_grad(env1, env2, bodies, d1, d2,
                                     hyps_mc_delta_1)
    mc_grad_delta_2 = n_body_mc_grad(env1, env2, bodies, d1, d2,
                                     hyps_mc_delta_2)
    mc_grad_delta_3 = n_body_mc_grad(env1, env2, bodies, d1, d2,
                                     hyps_mc_delta_3)

    np.isclose((mc_grad_delta_1[0] - mc_grad[0])/delta, mc_grad[1][2])
    np.isclose((mc_grad_delta_2[0] - mc_grad[0])/delta, mc_grad[1][3])
    np.isclose((mc_grad_delta_3[0] - mc_grad[0])/delta, mc_grad[1][4])

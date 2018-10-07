import numpy as np
from math import exp
from math import factorial
from itertools import combinations
from itertools import permutations
from numba import njit
from struc import Structure
from env import ChemicalEnvironment
import time


# -----------------------------------------------------------------------------
#                   combinatorics helper functions
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

# -----------------------------------------------------------------------------
#                kernels acting on environment objects
# -----------------------------------------------------------------------------


# get three body kernel between two environments
def three_body(env1, env2, d1, d2, sig, ls):
    return ChemicalEnvironment.three_body_jit(env1.bond_array,
                                              env1.bond_types,
                                              env1.cross_bond_dists,
                                              env1.cross_bond_types,
                                              env2.bond_array,
                                              env2.bond_types,
                                              env2.cross_bond_dists,
                                              env2.cross_bond_types,
                                              d1, d2, sig, ls)


# get two body kernel between two environments
def two_body(env1, env2, d1, d2, sig, ls):
    return ChemicalEnvironment.two_body_jit(env1.bond_array,
                                            env1.bond_types,
                                            env2.bond_array,
                                            env2.bond_types,
                                            d1, d2, sig, ls)


# -----------------------------------------------------------------------------
#           kernels acting on numpy arrays (can be jitted)
# -----------------------------------------------------------------------------


# single component n body kernel
def n_body_jit_sc(bond_array_1, cross_bond_dists_1, combinations,
                  bond_array_2, cross_bond_dists_2, permutations,
                  d1, d2, sig, ls):

    kern = 0

    for comb in combinations:
        for perm in permutations:
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


# testing ground
if __name__ == '__main__':
    # test get_comb_no and get_perm_no
    assert(get_comb_no(3, 2) == 3)
    assert(get_perm_no(3, 2) == 6)

    # test get_comb_array
    N = 10
    M = 5
    assert(get_comb_array(N, M).shape[0] == get_comb_no(N, M))

    # test get_perm_array
    assert(get_perm_array(N, M).shape[0] == get_perm_no(N, M))

    test1 = np.array([1, 4, 5])
    test2 = np.array([9, 7, 5])
    for a, (b, c) in enumerate(zip(test1, test2)):
        print(b)

    print(test1[0+1:])
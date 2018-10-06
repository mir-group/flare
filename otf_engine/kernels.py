import numpy as np
from math import exp
from numba import njit
from struc import Structure
from env import ChemicalEnvironment
import time

# -----------------------------------------------------------------------------
#                   kernels acting on environment objects
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
#                   kernels acting on arrays
# -----------------------------------------------------------------------------

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

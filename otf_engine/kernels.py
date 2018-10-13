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
#                        likelihood and gradients
# -----------------------------------------------------------------------------


def get_likelihood_and_gradients(hyps, training_data, training_labels_np,
                                 kernel_grad, bodies):

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps-1]
    kern_hyps = hyps[0:(number_of_hyps - 1)]

    # initialize matrices
    size = len(training_data)*3
    k_mat = np.zeros([size, size])
    hyp_mat = np.zeros([size, size, number_of_hyps])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size):
        x_1 = training_data[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]

        for n_index in range(m_index, size):
            x_2 = training_data[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]

            # calculate kernel and gradient
            cov = kernel_grad(x_1, x_2, bodies, d_1, d_2, kern_hyps)

            # store kernel value
            k_mat[m_index, n_index] = cov[0]
            k_mat[n_index, m_index] = cov[0]

            # store gradients (excluding noise variance)
            for p_index in range(number_of_hyps-1):
                hyp_mat[m_index, n_index, p_index] = cov[1][p_index]
                hyp_mat[n_index, m_index, p_index] = cov[1][p_index]

    # add gradient of noise variance
    hyp_mat[:, :, number_of_hyps-1] = np.eye(size) * 2 * sigma_n

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)
    ky_mat_inv = np.linalg.inv(ky_mat)
    l_mat = np.linalg.cholesky(ky_mat)
    alpha = np.matmul(ky_mat_inv, training_labels_np)
    alpha_mat = np.matmul(alpha.reshape(alpha.shape[0], 1),
                          alpha.reshape(1, alpha.shape[0]))
    like_mat = alpha_mat - ky_mat_inv

    # calculate likelihood
    like = (-0.5*np.matmul(training_labels_np, alpha) -
            np.sum(np.log(np.diagonal(l_mat))) -
            math.log(2 * np.pi) * k_mat.shape[1] / 2)

    # calculate likelihood gradient
    like_grad = np.zeros(number_of_hyps)
    for n in range(number_of_hyps):
        like_grad[n] = 0.5 * np.trace(np.matmul(like_mat, hyp_mat[:, :, n]))

    return -like, -like_grad


def get_K_L_alpha(hyps, training_data, training_labels_np,
                  kernel, bodies):

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps-1]
    kern_hyps = hyps[0:(number_of_hyps - 1)]

    # initialize matrices
    size = len(training_data)*3
    k_mat = np.zeros([size, size])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size):
        x_1 = training_data[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]

        for n_index in range(m_index, size):
            x_2 = training_data[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]

            # calculate kernel and gradient
            cov = kernel(x_1, x_2, bodies, d_1, d_2, kern_hyps)

            # store kernel value
            k_mat[m_index, n_index] = cov
            k_mat[n_index, m_index] = cov

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)
    l_mat = np.linalg.cholesky(ky_mat)
    ky_mat_inv = np.linalg.inv(ky_mat)
    alpha = np.matmul(ky_mat_inv, training_labels_np)
    return k_mat, l_mat, alpha

# -----------------------------------------------------------------------------
#               kernels and gradients acting on environment objects
# -----------------------------------------------------------------------------


# get n body single component kernel between two environments
def n_body_sc(env1, env2, bodies, d1, d2, hyps):
    combs = get_comb_array(env1.bond_array.shape[0], bodies-1)
    perms = get_perm_array(env2.bond_array.shape[0], bodies-1)

    return n_body_jit_sc(env1.bond_array, env1.cross_bond_dists, combs,
                         env2.bond_array, env2.cross_bond_dists, perms,
                         d1, d2, hyps)


def n_body_sc_grad(env1, env2, bodies, d1, d2, hyps):
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
#           kernels acting on numpy arrays (can be jitted)
# -----------------------------------------------------------------------------


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


if __name__ == '__main__':
    pass

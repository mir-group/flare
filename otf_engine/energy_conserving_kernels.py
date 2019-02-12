import numpy as np
from math import exp
from numba import njit


# -----------------------------------------------------------------------------
#                              two body kernels
# -----------------------------------------------------------------------------

def two_body(env1, env2, d1, d2, hyps, cutoffs):
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[0]

    return two_body_jit(env1.bond_array_2, env2.bond_array_2,
                        d1, d2, sig, ls, r_cut)


def two_body_grad(env1, env2, d1, d2, hyps, cutoffs):
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[0]

    kernel, ls_derv, sig_derv = \
        two_body_grad_jit(env1.bond_array_2, env2.bond_array_2,
                               d1, d2, sig, ls, r_cut)
    kernel_grad = np.array([sig_derv, ls_derv])
    return kernel, kernel_grad


def two_body_force_en(env1, env2, d1, hyps, cutoffs):
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[0]

    # divide by three to account for triple counting
    return two_body_force_en_jit(env1.bond_array_2, env2.bond_array_2,
                                 d1, sig, ls, r_cut)/2


def two_body_en(env1, env2, hyps, cutoffs):
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[0]

    return two_body_en_jit(env1.bond_array_2, env2.bond_array_2,
                           sig, ls, r_cut)


# -----------------------------------------------------------------------------
#                              three body kernels
# -----------------------------------------------------------------------------


def three_body_grad(env1, env2, d1, d2, hyps, cutoffs):
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    kernel, sig_derv, ls_derv = \
        three_body_grad_jit(env1.bond_array_3, env2.bond_array_3,
                            env1.cross_bond_inds, env2.cross_bond_inds,
                            env1.cross_bond_dists, env2.cross_bond_dists,
                            env1.triplet_counts, env2.triplet_counts,
                            d1, d2, sig, ls, r_cut)

    kernel_grad = np.array([sig_derv, ls_derv])

    return kernel, kernel_grad


def three_body(env1, env2, d1, d2, hyps, cutoffs):
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    return three_body_jit(env1.bond_array_3, env2.bond_array_3,
                          env1.cross_bond_inds, env2.cross_bond_inds,
                          env1.cross_bond_dists, env2.cross_bond_dists,
                          env1.triplet_counts, env2.triplet_counts,
                          d1, d2, sig, ls, r_cut)


def three_body_force_en(env1, env2, d1, hyps, cutoffs):
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    # divide by three to account for triple counting
    return three_body_force_en_jit(env1.bond_array_3, env2.bond_array_3,
                                   env1.cross_bond_inds, env2.cross_bond_inds,
                                   env1.cross_bond_dists,
                                   env2.cross_bond_dists,
                                   env1.triplet_counts, env2.triplet_counts,
                                   d1, sig, ls, r_cut)/3


def three_body_en(env1, env2, hyps, cutoffs):
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    return three_body_en_jit(env1.bond_array_3, env2.bond_array_3,
                             env1.cross_bond_inds, env2.cross_bond_inds,
                             env1.cross_bond_dists, env2.cross_bond_dists,
                             env1.triplet_counts, env2.triplet_counts,
                             sig, ls, r_cut)


# -----------------------------------------------------------------------------
#                           two body numba functions
# -----------------------------------------------------------------------------


@njit
def two_body_jit(bond_array_1, bond_array_2, d1, d2, sig, ls,
                 r_cut):
    kern = 0

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig*sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi = (r_cut - ri)**2
        fdi = 2 * (r_cut - ri) * ci

        for n in range(bond_array_2.shape[0]):
            rj = bond_array_2[n, 0]
            cj = bond_array_2[n, d2]
            fj = (r_cut - rj)**2
            fdj = 2 * (r_cut - rj) * cj
            r11 = ri - rj

            A = ci * cj
            B = r11 * ci
            C = r11 * cj
            D = r11 * r11

            kern += force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2,
                                 ls3, sig2)

    return kern


@njit
def two_body_grad_jit(bond_array_1, bond_array_2, d1, d2, sig, ls,
                      r_cut):

    kern = 0
    sig_derv = 0
    ls_derv = 0

    sig2, sig3, ls1, ls2, ls3, ls4, ls5, ls6 = grad_constants(sig, ls)

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi = (r_cut - ri)**2
        fdi = 2 * (r_cut - ri) * ci

        for n in range(bond_array_2.shape[0]):
            rj = bond_array_2[n, 0]
            cj = bond_array_2[n, d2]
            fj = (r_cut - rj)**2
            fdj = 2 * (r_cut - rj) * cj

            r11 = ri - rj

            A = ci * cj
            B = r11 * ci
            C = r11 * cj
            D = r11 * r11

            kern_term, sig_term, ls_term = \
                grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, ls4,
                            ls5, ls6, sig2, sig3)

            kern += kern_term
            sig_derv += sig_term
            ls_derv += ls_term

    return kern, ls_derv, sig_derv


@njit
def two_body_force_en_jit(bond_array_1, bond_array_2, d1, sig, ls, r_cut):
    kern = 0

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    sig2 = sig*sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi = (r_cut - ri)**2
        fdi = 2 * (r_cut - ri) * ci

        for n in range(bond_array_2.shape[0]):
            rj = bond_array_2[n, 0]
            fj = (r_cut - rj)**2

            r11 = ri - rj
            B = r11 * ci
            D = r11 * r11
            kern += force_energy_helper(B, D, fi, fj, fdi, ls1, ls2, sig2)

    return kern


@njit
def two_body_en_jit(bond_array_1, bond_array_2, sig, ls, r_cut):
    kern = 0

    ls1 = 1 / (2 * ls * ls)
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        fi = (r_cut - ri)**2

        for n in range(bond_array_2.shape[0]):
            rj = bond_array_2[n, 0]
            fj = (r_cut - rj)**2
            r11 = ri - rj

            kern += fi * fj * sig2 * exp(-r11 * r11 * ls1)

    return kern


# -----------------------------------------------------------------------------
#                           three body numba functions
# -----------------------------------------------------------------------------


@njit
def three_body_grad_jit(bond_array_1, bond_array_2,
                        cross_bond_inds_1, cross_bond_inds_2,
                        cross_bond_dists_1, cross_bond_dists_2,
                        triplets_1, triplets_2,
                        d1, d2, sig, ls, r_cut):
    """Kernel gradient for 3-body force comparisons."""

    kern = 0
    sig_derv = 0
    ls_derv = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig*sig
    sig3 = 2*sig
    ls1 = 1 / (2*ls*ls)
    ls2 = 1 / (ls*ls)
    ls3 = ls2*ls2
    ls4 = 1 / (ls*ls*ls)
    ls5 = ls*ls
    ls6 = ls2*ls4

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        rdi1 = r_cut-ri1
        ci1 = bond_array_1[m, d1]
        fi1 = rdi1*rdi1
        fdi1 = 2*rdi1*ci1

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m+n+1]
            ri3 = cross_bond_dists_1[m, m+n+1]
            ri2 = bond_array_1[ind1, 0]
            rdi2 = r_cut-ri2
            ci2 = bond_array_1[ind1, d1]
            fi2 = rdi2*rdi2
            fdi2 = 2*rdi2*ci2
            fi3 = (r_cut-ri3)**2
            fi = fi1*fi2*fi3
            fdi = fdi1*fi2*fi3+fi1*fdi2*fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                rdj1 = r_cut-rj1
                cj1 = bond_array_2[p, d2]
                fj1 = rdj1*rdj1
                fdj1 = 2*rdj1*cj1

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p+q+1]
                    rj3 = cross_bond_dists_2[p, p+q+1]
                    rj2 = bond_array_2[ind2, 0]
                    rdj2 = r_cut-rj2
                    cj2 = bond_array_2[ind2, d2]
                    fj2 = rdj2*rdj2
                    fdj2 = 2*rdj2*cj2
                    fj3 = (r_cut-rj3)**2
                    fj = fj1*fj2*fj3
                    fdj = fdj1*fj2*fj3+fj1*fdj2*fj3

                    N, O, X = \
                        triplet_kernel_grad(ci1, ci2, cj1, cj2, ri1, ri2, ri3,
                                            rj1, rj2, rj3, fi, fj, fdi, fdj,
                                            ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                            sig3)

                    kern += N
                    sig_derv += O
                    ls_derv += X

    return kern, sig_derv, ls_derv


@njit
def three_body_jit(bond_array_1, bond_array_2,
                   cross_bond_inds_1, cross_bond_inds_2,
                   cross_bond_dists_1, cross_bond_dists_2,
                   triplets_1, triplets_2,
                   d1, d2, sig, ls, r_cut):
    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig*sig
    ls1 = 1 / (2*ls*ls)
    ls2 = 1 / (ls*ls)
    ls3 = ls2*ls2

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        rdi1 = r_cut-ri1
        ci1 = bond_array_1[m, d1]
        fi1 = rdi1*rdi1
        fdi1 = 2*rdi1*ci1

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m+n+1]
            ri3 = cross_bond_dists_1[m, m+n+1]
            ri2 = bond_array_1[ind1, 0]
            rdi2 = r_cut-ri2
            ci2 = bond_array_1[ind1, d1]
            fi2 = rdi2*rdi2
            fdi2 = 2*rdi2*ci2
            fi3 = (r_cut-ri3)**2
            fi = fi1*fi2*fi3
            fdi = fdi1*fi2*fi3+fi1*fdi2*fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                rdj1 = r_cut-rj1
                cj1 = bond_array_2[p, d2]
                fj1 = rdj1*rdj1
                fdj1 = 2*rdj1*cj1

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p+1+q]
                    rj3 = cross_bond_dists_2[p, p+1+q]
                    rj2 = bond_array_2[ind2, 0]
                    rdj2 = r_cut-rj2
                    cj2 = bond_array_2[ind2, d2]
                    fj2 = rdj2*rdj2
                    fdj2 = 2*rdj2*cj2
                    fj3 = (r_cut-rj3)**2
                    fj = fj1*fj2*fj3
                    fdj = fdj1*fj2*fj3+fj1*fdj2*fj3

                    kern += triplet_kernel(ci1, ci2, cj1, cj2, ri1, ri2, ri3,
                                           rj1, rj2, rj3, fi, fj, fdi, fdj,
                                           ls1, ls2, ls3, sig2)

    return kern


@njit
def three_body_force_en_jit(bond_array_1, bond_array_2,
                            cross_bond_inds_1,
                            cross_bond_inds_2,
                            cross_bond_dists_1,
                            cross_bond_dists_2,
                            triplets_1, triplets_2,
                            d1, sig, ls, r_cut):
    """Kernel for 3-body force/energy comparisons."""

    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig*sig
    ls1 = 1 / (2*ls*ls)
    ls2 = 1 / (ls*ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        rdi1 = r_cut-ri1
        ci1 = bond_array_1[m, d1]
        fi1 = rdi1*rdi1
        fdi1 = 2*rdi1*ci1

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m+n+1]
            ri2 = bond_array_1[ind1, 0]
            rdi2 = r_cut-ri2
            ci2 = bond_array_1[ind1, d1]
            fi2 = rdi2*rdi2
            fdi2 = 2*rdi2*ci2
            ri3 = cross_bond_dists_1[m, m+n+1]
            fi3 = (r_cut-ri3)**2
            fi = fi1*fi2*fi3
            fdi = fdi1*fi2*fi3+fi1*fdi2*fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                rdj1 = r_cut-rj1
                fj1 = rdj1*rdj1

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p+q+1]
                    rj2 = bond_array_2[ind2, 0]
                    rdj2 = r_cut-rj2
                    fj2 = rdj2*rdj2
                    rj3 = cross_bond_dists_2[p, p+q+1]
                    fj3 = (r_cut-rj3)**2
                    fj = fj1*fj2*fj3

                    kern += triplet_force_en_kernel(ci1, ci2, ri1, ri2, ri3,
                                                    rj1, rj2, rj3, fi, fj, fdi,
                                                    ls1, ls2, sig2)

    return kern


@njit
def three_body_en_jit(bond_array_1, bond_array_2,
                      cross_bond_inds_1,
                      cross_bond_inds_2,
                      cross_bond_dists_1,
                      cross_bond_dists_2,
                      triplets_1, triplets_2,
                      sig, ls, r_cut):
    kern = 0

    sig2 = sig*sig
    ls2 = 1 / (2*ls*ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        fi1 = (r_cut-ri1)**2

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            fi2 = (r_cut-ri2)**2
            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3 = (r_cut-ri3)**2
            fi = fi1*fi2*fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1 = (r_cut-rj1)**2

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    fj2 = (r_cut-rj2)**2
                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    fj3 = (r_cut-rj3)**2
                    fj = fj1*fj2*fj3

                    r11 = ri1-rj1
                    r12 = ri1-rj2
                    r13 = ri1-rj3
                    r21 = ri2-rj1
                    r22 = ri2-rj2
                    r23 = ri2-rj3
                    r31 = ri3-rj1
                    r32 = ri3-rj2
                    r33 = ri3-rj3

                    C1 = r11*r11+r22*r22+r33*r33
                    C2 = r11*r11+r23*r23+r32*r32
                    C3 = r12*r12+r21*r21+r33*r33
                    C4 = r12*r12+r23*r23+r31*r31
                    C5 = r13*r13+r21*r21+r32*r32
                    C6 = r13*r13+r22*r22+r31*r31

                    k = exp(-C1*ls2)+exp(-C2*ls2)+exp(-C3*ls2)+exp(-C4*ls2) + \
                        exp(-C5*ls2)+exp(-C6*ls2)

                    kern += sig2*k*fi*fj

    return kern


# -----------------------------------------------------------------------------
#                            general helper functions
# -----------------------------------------------------------------------------


@njit
def grad_constants(sig, ls):
    sig2 = sig * sig
    sig3 = 2 * sig

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    ls4 = 1 / (ls * ls * ls)
    ls5 = ls * ls
    ls6 = ls2 * ls4

    return sig2, sig3, ls1, ls2, ls3, ls4, ls5, ls6


@njit
def force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2):
    E = exp(-D * ls1)
    F = E * B * ls2
    G = -E * C * ls2
    H = A * E * ls2 - B * C * E * ls3
    I = E * fdi * fdj
    J = F * fi * fdj
    K = G * fdi * fj
    L = H * fi * fj
    M = sig2 * (I + J + K + L)

    return M


@njit
def grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6,
                sig2, sig3):
    E = exp(-D * ls1)
    F = E * B * ls2
    G = -E * C * ls2
    H = A * E * ls2 - B * C * E * ls3
    I = E * fdi * fdj
    J = F * fi * fdj
    K = G * fdi * fj
    L = H * fi * fj
    M = I + J + K + L
    N = sig2 * M
    O = sig3 * M
    P = E * D * ls4
    Q = B * (ls2 * P - 2 * E * ls4)
    R = -C * (ls2 * P - 2 * E * ls4)
    S = (A * ls5 - B * C) * (P * ls3 - 4 * E * ls6) + 2 * E * A * ls4
    T = P * fdi * fdj
    U = Q * fi * fdj
    V = R * fdi * fj
    W = S * fi * fj
    X = sig2 * (T + U + V + W)

    return N, O, X


@njit
def force_energy_helper(B, D, fi, fj, fdi, ls1, ls2, sig2):
    E = exp(-D * ls1)
    F = E * B * ls2
    G = -F * fi * fj
    H = -E * fdi * fj
    I = sig2 * (G + H)

    return I


# -----------------------------------------------------------------------------
#                        three body helper functions
# -----------------------------------------------------------------------------


@njit
def triplet_kernel(ci1, ci2, cj1, cj2, ri1, ri2, ri3, rj1, rj2, rj3, fi, fj,
                   fdi, fdj, ls1, ls2, ls3, sig2):
    r11 = ri1-rj1
    r12 = ri1-rj2
    r13 = ri1-rj3
    r21 = ri2-rj1
    r22 = ri2-rj2
    r23 = ri2-rj3
    r31 = ri3-rj1
    r32 = ri3-rj2
    r33 = ri3-rj3

    # sum over all six permutations
    M1 = three_body_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M2 = three_body_helper_2(ci2, ci1, cj2, cj1, r21, r13, r32, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M3 = three_body_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M4 = three_body_helper_1(ci1, ci2, cj2, cj1, r12, r21, r33, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M5 = three_body_helper_2(ci2, ci1, cj1, cj2, r22, r13, r31, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M6 = three_body_helper_2(ci1, ci2, cj2, cj1, r11, r23, r32, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)

    return M1 + M2 + M3 + M4 + M5 + M6


@njit
def triplet_kernel_grad(ci1, ci2, cj1, cj2, ri1, ri2, ri3, rj1, rj2, rj3, fi,
                        fj, fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                        sig3):
    r11 = ri1-rj1
    r12 = ri1-rj2
    r13 = ri1-rj3
    r21 = ri2-rj1
    r22 = ri2-rj2
    r23 = ri2-rj3
    r31 = ri3-rj1
    r32 = ri3-rj2
    r33 = ri3-rj3

    N1, O1, X1 = \
        three_body_grad_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N2, O2, X2 = \
        three_body_grad_helper_2(ci2, ci1, cj2, cj1, r21, r13, r32, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N3, O3, X3 = \
        three_body_grad_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N4, O4, X4 = \
        three_body_grad_helper_1(ci1, ci2, cj2, cj1, r12, r21, r33, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N5, O5, X5 = \
        three_body_grad_helper_2(ci2, ci1, cj1, cj2, r22, r13, r31, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N6, O6, X6 = \
        three_body_grad_helper_2(ci1, ci2, cj2, cj1, r11, r23, r32, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N = N1 + N2 + N3 + N4 + N5 + N6
    O = O1 + O2 + O3 + O4 + O5 + O6
    X = X1 + X2 + X3 + X4 + X5 + X6
    return N, O, X


@njit
def three_body_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33,
                        fi, fj, fdi, fdj,
                        ls1, ls2, ls3, sig2):
    A = ci1*cj1+ci2*cj2
    B = r11*ci1+r22*ci2
    C = r11*cj1+r22*cj2
    D = r11*r11+r22*r22+r33*r33

    M = force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2)

    return M


@njit
def three_body_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31,
                        fi, fj, fdi, fdj,
                        ls1, ls2, ls3, sig2):
    A = ci1*cj2
    B = r12*ci1+r23*ci2
    C = r12*cj2+r31*cj1
    D = r12*r12+r23*r23+r31*r31

    M = force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2)

    return M


@njit
def three_body_grad_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2, sig3):
    A = ci1*cj1+ci2*cj2
    B = r11*ci1+r22*ci2
    C = r11*cj1+r22*cj2
    D = r11*r11+r22*r22+r33*r33

    N, O, X = grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, ls4,
                          ls5, ls6, sig2, sig3)

    return N, O, X


@njit
def three_body_grad_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2, sig3):
    A = ci1*cj2
    B = r12*ci1+r23*ci2
    C = r12*cj2+r31*cj1
    D = r12*r12+r23*r23+r31*r31

    N, O, X = grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, ls4,
                          ls5, ls6, sig2, sig3)

    return N, O, X


@njit
def three_body_en_helper(ci1, ci2, r11, r22, r33, fi, fj, fdi, ls1, ls2, sig2):
    B = r11 * ci1 + r22 * ci2
    D = r11 * r11 + r22 * r22 + r33 * r33

    return force_energy_helper(B, D, fi, fj, fdi, ls1, ls2, sig2)


@njit
def triplet_force_en_kernel(ci1, ci2, ri1, ri2, ri3, rj1, rj2, rj3,
                            fi, fj, fdi, ls1, ls2, sig2):
    r11 = ri1-rj1
    r12 = ri1-rj2
    r13 = ri1-rj3
    r21 = ri2-rj1
    r22 = ri2-rj2
    r23 = ri2-rj3
    r31 = ri3-rj1
    r32 = ri3-rj2
    r33 = ri3-rj3

    I1 = three_body_en_helper(ci1, ci2, r11, r22, r33, fi, fj,
                              fdi, ls1, ls2, sig2)
    I2 = three_body_en_helper(ci1, ci2, r13, r21, r32, fi, fj,
                              fdi, ls1, ls2, sig2)
    I3 = three_body_en_helper(ci1, ci2, r12, r23, r31, fi, fj,
                              fdi, ls1, ls2, sig2)
    I4 = three_body_en_helper(ci1, ci2, r12, r21, r33, fi, fj,
                              fdi, ls1, ls2, sig2)
    I5 = three_body_en_helper(ci1, ci2, r13, r22, r31, fi, fj,
                              fdi, ls1, ls2, sig2)
    I6 = three_body_en_helper(ci1, ci2, r11, r23, r32, fi, fj,
                              fdi, ls1, ls2, sig2)

    return I1 + I2 + I3 + I4 + I5 + I6


if __name__ == '__main__':
    pass

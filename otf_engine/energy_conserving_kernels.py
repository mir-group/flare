import numpy as np
import math
from math import exp
from math import factorial
from numba import njit
from struc import Structure
from env import ChemicalEnvironment
import gp
import struc
import env
import fast_env
import time
from random import random, randint
from copy import deepcopy


def three_body_cons_quad_grad(env1, env2, bodies, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    kernel, sig_derv, ls_derv = \
        three_body_cons_quad_grad_jit(env1.bond_array, env2.bond_array,
                                      env1.cross_bond_dists,
                                      env2.cross_bond_dists,
                                      d1, d2, sig, ls, r_cut)

    kernel_grad = np.array([sig_derv, ls_derv])

    return kernel, kernel_grad


def three_body_cons_quad(env1, env2, bodies, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    return three_body_cons_quad_jit(env1.bond_array, env2.bond_array,
                                    env1.cross_bond_dists,
                                    env2.cross_bond_dists,
                                    d1, d2, sig, ls, r_cut)


def three_body_cons_quad_v2(env1, env2, bodies, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    return three_body_cons_quad_jit_v2(env1.bond_array_3, env2.bond_array_3,
                                       env1.cross_bond_inds,
                                       env2.cross_bond_inds,
                                       env1.cross_bond_dists,
                                       env2.cross_bond_dists,
                                       env1.triplet_counts,
                                       env2.triplet_counts,
                                       d1, d2, sig, ls, r_cut)


def three_body_cons_quad_force_en(env1, env2, bodies, d1, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    # divide by three to account for triple counting
    return three_body_cons_quad_force_en_jit(env1.bond_array, env2.bond_array,
                                             env1.cross_bond_dists,
                                             env2.cross_bond_dists,
                                             d1, sig, ls, r_cut)/3


def three_body_cons_quad_en(env1, env2, bodies, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    return three_body_cons_quad_en_jit(env1.bond_array, env2.bond_array,
                                       env1.cross_bond_dists,
                                       env2.cross_bond_dists,
                                       sig, ls, r_cut)


@njit
def three_body_cons_quad_grad_jit(bond_array_1, bond_array_2,
                                  cross_bond_dists_1, cross_bond_dists_2,
                                  d1, d2, sig, ls, r_cut):
    """Kernel gradient for 3-body force comparisons.

    :param bond_array_1: [description]
    :type bond_array_1: [type]
    :param bond_array_2: [description]
    :type bond_array_2: [type]
    :param cross_bond_dists_1: [description]
    :type cross_bond_dists_1: [type]
    :param cross_bond_dists_2: [description]
    :type cross_bond_dists_2: [type]
    :param d1: [description]
    :type d1: [type]
    :param d2: [description]
    :type d2: [type]
    :param sig: [description]
    :type sig: [type]
    :param ls: [description]
    :type ls: [type]
    :param r_cut: [description]
    :type r_cut: [type]
    :return: [description]
    :rtype: [type]
    """

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

        for n in range(m+1, bond_array_1.shape[0]):
            ri3 = cross_bond_dists_1[m, n]
            if ri3 > r_cut:
                continue
            ri2 = bond_array_1[n, 0]
            rdi2 = r_cut-ri2
            ci2 = bond_array_1[n, d1]
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

                for q in range(bond_array_2.shape[0]):
                    if p == q:
                        continue
                    rj3 = cross_bond_dists_2[p, q]
                    if rj3 > r_cut:
                        continue

                    rj2 = bond_array_2[q, 0]
                    rdj2 = r_cut-rj2
                    cj2 = bond_array_2[q, d2]
                    fj2 = rdj2*rdj2
                    fdj2 = 2*rdj2*cj2
                    fj3 = (r_cut-rj3)**2
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

                    # first cyclic term
                    A1 = ci1*cj1+ci2*cj2
                    B1 = r11*ci1+r22*ci2
                    C1 = r11*cj1+r22*cj2
                    D1 = r11*r11+r22*r22+r33*r33
                    E1 = exp(-D1*ls1)
                    F1 = E1*B1*ls2
                    G1 = -E1*C1*ls2
                    H1 = A1*E1*ls2-B1*C1*E1*ls3
                    I1 = E1*fdi*fdj
                    J1 = F1*fi*fdj
                    K1 = G1*fdi*fj
                    L1 = H1*fi*fj
                    M1 = I1+J1+K1+L1
                    N1 = sig2*M1
                    O1 = sig3*M1
                    P1 = E1*D1*ls4
                    Q1 = B1*(ls2*P1-2*E1*ls4)
                    R1 = -C1*(ls2*P1-2*E1*ls4)
                    S1 = (A1*ls5-B1*C1)*(P1*ls3-4*E1*ls6)+2*E1*A1*ls4
                    T1 = P1*fdi*fdj
                    U1 = Q1*fi*fdj
                    V1 = R1*fdi*fj
                    W1 = S1*fi*fj
                    X1 = sig2*(T1+U1+V1+W1)

                    # second cyclic term
                    A2 = ci2*cj1
                    B2 = r13*ci1+r21*ci2
                    C2 = r21*cj1+r32*cj2
                    D2 = r13*r13+r21*r21+r32*r32
                    E2 = exp(-D2*ls1)
                    F2 = E2*B2*ls2
                    G2 = -E2*C2*ls2
                    H2 = A2*E2*ls2-B2*C2*E2*ls3
                    I2 = E2*fdi*fdj
                    J2 = F2*fi*fdj
                    K2 = G2*fdi*fj
                    L2 = H2*fi*fj
                    M2 = I2+J2+K2+L2
                    N2 = sig2*M2
                    O2 = sig3*M2
                    P2 = E2*D2*ls4
                    Q2 = B2*(ls2*P2-2*E2*ls4)
                    R2 = -C2*(ls2*P2-2*E2*ls4)
                    S2 = (A2*ls5-B2*C2)*(P2*ls3-4*E2*ls6)+2*E2*A2*ls4
                    T2 = P2*fdi*fdj
                    U2 = Q2*fi*fdj
                    V2 = R2*fdi*fj
                    W2 = S2*fi*fj
                    X2 = sig2*(T2+U2+V2+W2)

                    # third cyclic term
                    A3 = ci1*cj2
                    B3 = r12*ci1+r23*ci2
                    C3 = r12*cj2+r31*cj1
                    D3 = r12*r12+r23*r23+r31*r31
                    E3 = exp(-D3*ls1)
                    F3 = E3*B3*ls2
                    G3 = -E3*C3*ls2
                    H3 = A3*E3*ls2-B3*C3*E3*ls3
                    I3 = E3*fdi*fdj
                    J3 = F3*fi*fdj
                    K3 = G3*fdi*fj
                    L3 = H3*fi*fj
                    M3 = I3+J3+K3+L3
                    N3 = sig2*M3
                    O3 = sig3*M3
                    P3 = E3*D3*ls4
                    Q3 = B3*(ls2*P3-2*E3*ls4)
                    R3 = -C3*(ls2*P3-2*E3*ls4)
                    S3 = (A3*ls5-B3*C3)*(P3*ls3-4*E3*ls6)+2*E3*A3*ls4
                    T3 = P3*fdi*fdj
                    U3 = Q3*fi*fdj
                    V3 = R3*fdi*fj
                    W3 = S3*fi*fj
                    X3 = sig2*(T3+U3+V3+W3)

                    kern += N1+N2+N3
                    sig_derv += O1+O2+O3
                    ls_derv += X1+X2+X3

    return kern, sig_derv, ls_derv


@njit
def three_body_cons_quad_jit(bond_array_1, bond_array_2,
                             cross_bond_dists_1, cross_bond_dists_2,
                             d1, d2, sig, ls, r_cut):
    """Kernel for 3-body force comparisons.

    :param bond_array_1: [description]
    :type bond_array_1: [type]
    :param bond_array_2: [description]
    :type bond_array_2: [type]
    :param cross_bond_dists_1: [description]
    :type cross_bond_dists_1: [type]
    :param cross_bond_dists_2: [description]
    :type cross_bond_dists_2: [type]
    :param d1: [description]
    :type d1: [type]
    :param d2: [description]
    :type d2: [type]
    :param sig: [description]
    :type sig: [type]
    :param ls: [description]
    :type ls: [type]
    :param r_cut: [description]
    :type r_cut: [type]
    :return: [description]
    :rtype: [type]
    """

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

        for n in range(m+1, bond_array_1.shape[0]):
            ri3 = cross_bond_dists_1[m, n]
            if ri3 > r_cut:
                continue
            ri2 = bond_array_1[n, 0]
            rdi2 = r_cut-ri2
            ci2 = bond_array_1[n, d1]
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

                for q in range(p+1, bond_array_2.shape[0]):
                    rj3 = cross_bond_dists_2[p, q]
                    if rj3 > r_cut:
                        continue
                    rj2 = bond_array_2[q, 0]
                    rdj2 = r_cut-rj2
                    cj2 = bond_array_2[q, d2]
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
def three_body_cons_quad_jit_v2(bond_array_1, bond_array_2,
                                cross_bond_inds_1, cross_bond_inds_2,
                                cross_bond_dists_1, cross_bond_dists_2,
                                triplets_1, triplets_2,
                                d1, d2, sig, ls, r_cut):
    """Kernel for 3-body force comparisons.

    :param bond_array_1: [description]
    :type bond_array_1: [type]
    :param bond_array_2: [description]
    :type bond_array_2: [type]
    :param cross_bond_dists_1: [description]
    :type cross_bond_dists_1: [type]
    :param cross_bond_dists_2: [description]
    :type cross_bond_dists_2: [type]
    :param d1: [description]
    :type d1: [type]
    :param d2: [description]
    :type d2: [type]
    :param sig: [description]
    :type sig: [type]
    :param ls: [description]
    :type ls: [type]
    :param r_cut: [description]
    :type r_cut: [type]
    :return: [description]
    :rtype: [type]
    """

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
            ind1 = cross_bond_inds_1[m, m+1+n]
            ri3 = cross_bond_dists_1[m, m+1+n]
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
def three_body_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33,
                        fi, fj, fdi, fdj,
                        ls1, ls2, ls3, sig2):
    A1 = ci1*cj1+ci2*cj2
    B1 = r11*ci1+r22*ci2
    C1 = r11*cj1+r22*cj2
    D1 = r11*r11+r22*r22+r33*r33
    E1 = exp(-D1*ls1)
    F1 = E1*B1*ls2
    G1 = -E1*C1*ls2
    H1 = A1*E1*ls2-B1*C1*E1*ls3
    I1 = E1*fdi*fdj
    J1 = F1*fi*fdj
    K1 = G1*fdi*fj
    L1 = H1*fi*fj
    M1 = sig2*(I1+J1+K1+L1)

    return M1


@njit
def three_body_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31,
                        fi, fj, fdi, fdj,
                        ls1, ls2, ls3, sig2):
    A3 = ci1*cj2
    B3 = r12*ci1+r23*ci2
    C3 = r12*cj2+r31*cj1
    D3 = r12*r12+r23*r23+r31*r31
    E3 = exp(-D3*ls1)
    F3 = E3*B3*ls2
    G3 = -E3*C3*ls2
    H3 = A3*E3*ls2-B3*C3*E3*ls3
    I3 = E3*fdi*fdj
    J3 = F3*fi*fdj
    K3 = G3*fdi*fj
    L3 = H3*fi*fj
    M3 = sig2*(I3+J3+K3+L3)

    return M3


@njit
def three_body_cons_quad_force_en_jit(bond_array_1, bond_array_2,
                                      cross_bond_dists_1,
                                      cross_bond_dists_2,
                                      d1, sig, ls, r_cut):
    """Kernel for 3-body force/energy comparisons.

    :param bond_array_1: [description]
    :type bond_array_1: [type]
    :param bond_array_2: [description]
    :type bond_array_2: [type]
    :param cross_bond_dists_1: [description]
    :type cross_bond_dists_1: [type]
    :param cross_bond_dists_2: [description]
    :type cross_bond_dists_2: [type]
    :param d1: [description]
    :type d1: [type]
    :param sig: [description]
    :type sig: [type]
    :param ls: [description]
    :type ls: [type]
    :param r_cut: [description]
    :type r_cut: [type]
    :return: [description]
    :rtype: [type]
    """

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

        for n in range(m+1, bond_array_1.shape[0]):
            ri2 = bond_array_1[n, 0]
            rdi2 = r_cut-ri2
            ci2 = bond_array_1[n, d1]
            fi2 = rdi2*rdi2
            fdi2 = 2*rdi2*ci2
            ri3 = cross_bond_dists_1[m, n]
            if ri3 > r_cut:
                continue
            fi3 = (r_cut-ri3)**2
            fi = fi1*fi2*fi3
            fdi = fdi1*fi2*fi3+fi1*fdi2*fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                rdj1 = r_cut-rj1
                fj1 = rdj1*rdj1

                for q in range(bond_array_2.shape[0]):
                    if p == q:
                        continue

                    rj2 = bond_array_2[q, 0]
                    rdj2 = r_cut-rj2
                    fj2 = rdj2*rdj2
                    rj3 = cross_bond_dists_2[p, q]
                    if rj3 > r_cut:
                        continue
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

                    # first cyclic term
                    B1 = r11*ci1+r22*ci2
                    D1 = r11*r11+r22*r22+r33*r33
                    E1 = exp(-D1*ls1)
                    F1 = E1*B1*ls2
                    G1 = -F1*fi*fj
                    H1 = -E1*fdi*fj
                    I1 = sig2*(G1+H1)

                    # second cyclic term
                    B2 = r13*ci1+r21*ci2
                    D2 = r13*r13+r21*r21+r32*r32
                    E2 = exp(-D2*ls1)
                    F2 = E2*B2*ls2
                    G2 = -F2*fi*fj
                    H2 = -E2*fdi*fj
                    I2 = sig2*(G2+H2)

                    # third cyclic term
                    B3 = r12*ci1+r23*ci2
                    D3 = r12*r12+r23*r23+r31*r31
                    E3 = exp(-D3*ls1)
                    F3 = E3*B3*ls2
                    G3 = -F3*fi*fj
                    H3 = -E3*fdi*fj
                    I3 = sig2*(G3+H3)

                    kern += I1+I2+I3

    return kern


@njit
def three_body_cons_quad_en_jit(bond_array_1, bond_array_2,
                                cross_bond_dists_1, cross_bond_dists_2,
                                sig, ls, r_cut):
    kern = 0

    sig2 = sig*sig
    ls2 = 1 / (2*ls*ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        fi1 = (r_cut-ri1)**2

        for n in range(m+1, bond_array_1.shape[0]):
            ri2 = bond_array_1[n, 0]
            fi2 = (r_cut-ri2)**2
            ri3 = cross_bond_dists_1[m, n]

            if ri3 > r_cut:
                continue
            fi3 = (r_cut-ri3)**2
            fi = fi1*fi2*fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1 = (r_cut-rj1)**2

                for q in range(p+1, bond_array_2.shape[0]):
                    rj2 = bond_array_2[q, 0]
                    fj2 = (r_cut-rj2)**2
                    rj3 = cross_bond_dists_2[p, q]

                    if rj3 > r_cut:
                        continue
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

if __name__ == '__main__':
    # create env 1
    delt = 1e-5
    cell = np.eye(3)
    cutoff = 1

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    species_1 = ['A', 'B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
    test_structure_2 = struc.Structure(cell, species_1, positions_2, cutoff)

    env1_1 = env.ChemicalEnvironment(test_structure_1, atom_1)
    env1_2 = env.ChemicalEnvironment(test_structure_2, atom_1)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]

    species_2 = ['A', 'A', 'B']
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1, cutoff)
    env2 = env.ChemicalEnvironment(test_structure_1, atom_2)

    sig = random()
    ls = random()
    d1 = 1
    bodies = None

    hyps = np.array([sig, ls])

    # check force kernel
    calc1 = three_body_cons_quad_en(env1_2, env2, bodies, hyps, cutoff)
    calc2 = three_body_cons_quad_en(env1_1, env2, bodies, hyps, cutoff)

    kern_finite_diff = (calc1 - calc2) / delt
    kern_analytical = three_body_cons_quad_force_en(env1_1, env2, bodies,
                                                    d1, hyps, cutoff)

    assert(np.isclose(-kern_finite_diff, kern_analytical))

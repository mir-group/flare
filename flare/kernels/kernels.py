import numpy as np
from math import exp
from numba import njit
import flare.kernels.cutoffs as cf

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
    """Helper function for computing the force/force kernel between two
    pairs or triplets of atoms of the same type.

    See Table IV of the SI of the FLARE paper for definitions of intermediate
    quantities.

    Returns:
        float: Force/force kernel between two pairs or triplets of atoms of
            the same type.
    """
    E = exp(-D * ls1)
    I = fdi * fdj
    J = B * ls2 * fi * fdj
    K = - C * ls2 * fdi * fj
    L = (A * ls2 - B * C * ls3) * fi * fj
    M = sig2 * (I + J + K + L) * E
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
    P = D * ls4
    Q = B * (ls2 * P - 2 * ls4)
    R = -C * (ls2 * P - 2 * ls4)
    S = (A * ls5 - B * C) * (P * ls3 - 4 * ls6) + 2 * A * ls4
    T = P * fdi * fdj
    U = Q * fi * fdj
    V = R * fdi * fj
    W = S * fi * fj
    X = sig2 * (T + U + V + W) * E

    return N, O, X


@njit
def force_energy_helper(B, D, fi, fj, fdi, ls1, ls2, sig2):
    E = exp(-D * ls1)
    F = B * ls2
    G = -F * fi * fj
    H = -fdi * fj
    I = sig2 * (G + H) * E

    return I


@njit
def three_body_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj, fdi, fdj,
                        ls1, ls2, ls3, sig2):
    A = ci1 * cj1 + ci2 * cj2
    B = r11 * ci1 + r22 * ci2
    C = r11 * cj1 + r22 * cj2
    D = r11 * r11 + r22 * r22 + r33 * r33

    M = force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2)

    return M


@njit
def three_body_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj, fdi, fdj,
                        ls1, ls2, ls3, sig2):
    A = ci1 * cj2
    B = r12 * ci1 + r23 * ci2
    C = r12 * cj2 + r31 * cj1
    D = r12 * r12 + r23 * r23 + r31 * r31

    M = force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2)

    return M


@njit
def three_body_sf_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj, fdi, fdj,
                    ls1, ls2, ls3, sig2, coord1, coord2, fdi1, fdi2):
    A = ci1 * cj1 * coord1 + ci2 * cj2 * coord2
    B = r11 * ci1 * coord1 + r22 * ci2 * coord2
    C = r11 * cj1 + r22 * cj2
    D = r11 * r11 + r22 * r22 + r33 * r33
    E = exp(-D * ls1)
    F = B * ls2
    G = -C * ls2
    H = A * ls2 - B * C * ls3
    I = (fdi1 * coord1 + fdi2 * coord2) * fdj
    J = F * fi * fdj
    K = G * (fdi1 * coord1 + fdi2 * coord2) * fj
    L = H * fi * fj
    M = -sig2 * (I + J + K + L) * E

    return M


@njit
def three_body_sf_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj, fdi, fdj,
                    ls1, ls2, ls3, sig2, coord1, coord2, fdi1, fdi2):
    A = ci1 * cj2 * coord1
    B = r12 * ci1 * coord1 + r23 * ci2 * coord2
    C = r12 * cj2 + r31 * cj1
    D = r12 * r12 + r23 * r23 + r31 * r31
    E = exp(-D * ls1)
    F = B * ls2
    G = -C * ls2
    H = A * ls2 - B * C * ls3
    I = (fdi1 * coord1 + fdi2 * coord2) * fdj
    J = F * fi * fdj
    K = G * (fdi1 * coord1 + fdi2 * coord2) * fj
    L = H * fi * fj
    M = -sig2 * (I + J + K + L) * E

    return M


@njit
def three_body_ss_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj, fdi, fdj,
                    ls1, ls2, ls3, sig2, coord1, coord2, coord3, coord4,
                    fdi_p1, fdi_p2, fdj_p1, fdj_p2):
    A = ci1 * cj1 * coord1 * coord3 + ci2 * cj2 * coord2 * coord4
    B = r11 * ci1 * coord1 + r22 * ci2 * coord2
    C = r11 * cj1 * coord3 + r22 * cj2 * coord4
    D = r11 * r11 + r22 * r22 + r33 * r33
    E = exp(-D * ls1)
    F = B * ls2
    G = -C * ls2
    H = A * ls2 - B * C * ls3
    I = (fdi_p1 * coord1 + fdi_p2 * coord2) * \
        (fdj_p1 * coord3 + fdj_p2 * coord4)
    J = F * fi * (fdj_p1 * coord3 + fdj_p2 * coord4)
    K = G * (fdi_p1 * coord1 + fdi_p2 * coord2) * fj
    L = H * fi * fj
    M = sig2 * (I + J + K + L) * E

    return M


@njit
def three_body_ss_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj, fdi, fdj,
                    ls1, ls2, ls3, sig2, coord1, coord2, coord3, coord4,
                    fdi_p1, fdi_p2, fdj_p1, fdj_p2):
    A = ci1 * cj2 * coord1 * coord4
    B = r12 * ci1 * coord1 + r23 * ci2 * coord2
    C = r12 * cj2 * coord4 + r31 * cj1 * coord3
    D = r12 * r12 + r23 * r23 + r31 * r31
    E = exp(-D * ls1)
    F = B * ls2
    G = -C * ls2
    H = A * ls2 - B * C * ls3
    I = (fdi_p1 * coord1 + fdi_p2 * coord2) * \
        (fdj_p1 * coord3 + fdj_p2 * coord4)
    J = F * fi * (fdj_p1 * coord3 + fdj_p2 * coord4)
    K = G * (fdi_p1 * coord1 + fdi_p2 * coord2) * fj
    L = H * fi * fj
    M = sig2 * (I + J + K + L) * E

    return M


@njit
def three_body_grad_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2, sig3):
    A = ci1 * cj1 + ci2 * cj2
    B = r11 * ci1 + r22 * ci2
    C = r11 * cj1 + r22 * cj2
    D = r11 * r11 + r22 * r22 + r33 * r33

    N, O, X = grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, ls4,
                          ls5, ls6, sig2, sig3)

    return N, O, X


@njit
def three_body_grad_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2, sig3):
    A = ci1 * cj2
    B = r12 * ci1 + r23 * ci2
    C = r12 * cj2 + r31 * cj1
    D = r12 * r12 + r23 * r23 + r31 * r31

    N, O, X = grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, ls4,
                          ls5, ls6, sig2, sig3)

    return N, O, X


@njit
def three_body_en_helper(ci1, ci2, r11, r22, r33, fi, fj, fdi, ls1, ls2, sig2):
    B = r11 * ci1 + r22 * ci2
    D = r11 * r11 + r22 * r22 + r33 * r33

    return force_energy_helper(B, D, fi, fj, fdi, ls1, ls2, sig2)


@njit
def three_body_ee_perm(r11, r12, r13, r21, r22, r23, r31, r32, r33, c1, c2,
                       ei1, ei2, ej1, ej2, fi, fj, ls2, sig2):
    kern = 0

    if (c1 == c2):
        if (ei1 == ej1) and (ei2 == ej2):
            C1 = r11 * r11 + r22 * r22 + r33 * r33
            kern += exp(-C1 * ls2)
        if (ei1 == ej2) and (ei2 == ej1):
            C3 = r12 * r12 + r21 * r21 + r33 * r33
            kern += exp(-C3 * ls2)
    if (c1 == ej1):
        if (ei1 == ej2) and (ei2 == c2):
            C5 = r13 * r13 + r21 * r21 + r32 * r32
            kern += exp(-C5 * ls2)
        if (ei1 == c2) and (ei2 == ej2):
            C2 = r11 * r11 + r23 * r23 + r32 * r32
            kern += exp(-C2 * ls2)
    if (c1 == ej2):
        if (ei1 == ej1) and (ei2 == c2):
            C6 = r13 * r13 + r22 * r22 + r31 * r31
            kern += exp(-C6 * ls2)
        if (ei1 == c2) and (ei2 == ej1):
            C4 = r12 * r12 + r23 * r23 + r31 * r31
            kern += exp(-C4 * ls2)

    return kern * sig2 * fi * fj


@njit
def three_body_fe_perm(r11, r12, r13, r21, r22, r23, r31, r32, r33, c1, c2,
                       ci1, ci2, ei1, ei2, ej1, ej2, fi, fj, fdi, ls1, ls2,
                       sig2):
    kern = 0

    if (c1 == c2):
        if (ei1 == ej1) and (ei2 == ej2):
            kern += three_body_en_helper(
                ci1, ci2, r11, r22, r33, fi, fj, fdi, ls1, ls2, sig2)
        if (ei1 == ej2) and (ei2 == ej1):
            kern += three_body_en_helper(
                ci1, ci2, r12, r21, r33, fi, fj, fdi, ls1, ls2, sig2)
    if (c1 == ej1):
        if (ei1 == ej2) and (ei2 == c2):
            kern += three_body_en_helper(
                ci1, ci2, r13, r21, r32, fi, fj, fdi, ls1, ls2, sig2)
        if (ei1 == c2) and (ei2 == ej2):
            kern += three_body_en_helper(
                ci1, ci2, r11, r23, r32, fi, fj, fdi, ls1, ls2, sig2)
    if (c1 == ej2):
        if (ei1 == ej1) and (ei2 == c2):
            kern += three_body_en_helper(
                ci1, ci2, r13, r22, r31, fi, fj, fdi, ls1, ls2, sig2)
        if (ei1 == c2) and (ei2 == ej1):
            kern += three_body_en_helper(
                ci1, ci2, r12, r23, r31, fi, fj, fdi, ls1, ls2, sig2)

    return kern


@njit
def three_body_ff_perm(r11, r12, r13, r21, r22, r23, r31, r32, r33, c1, c2,
                       ci1, ci2, cj1, cj2, ei1, ei2, ej1, ej2, fi, fj, fdi,
                       fdj, ls1, ls2, ls3, sig2):

    kern = 0

    if (c1 == c2):
        if (ei1 == ej1) and (ei2 == ej2):
            kern += \
                three_body_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi,
                                    fj, fdi, fdj, ls1, ls2, ls3, sig2)
        if (ei1 == ej2) and (ei2 == ej1):
            kern += \
                three_body_helper_1(ci1, ci2, cj2, cj1, r12, r21, r33, fi,
                                    fj, fdi, fdj, ls1, ls2, ls3, sig2)
    if (c1 == ej1):
        if (ei1 == ej2) and (ei2 == c2):
            kern += \
                three_body_helper_2(ci2, ci1, cj2, cj1, r21, r13, r32, fi,
                                    fj, fdi, fdj, ls1, ls2, ls3, sig2)
        if (ei1 == c2) and (ei2 == ej2):
            kern += \
                three_body_helper_2(ci1, ci2, cj2, cj1, r11, r23, r32, fi,
                                    fj, fdi, fdj, ls1, ls2, ls3, sig2)
    if (c1 == ej2):
        if (ei1 == ej1) and (ei2 == c2):
            kern += \
                three_body_helper_2(ci2, ci1, cj1, cj2, r22, r13, r31, fi,
                                    fj, fdi, fdj, ls1, ls2, ls3, sig2)
        if (ei1 == c2) and (ei2 == ej1):
            kern += \
                three_body_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi,
                                    fj, fdi, fdj, ls1, ls2, ls3, sig2)

    return kern


@njit
def three_body_sf_perm(r11, r12, r13, r21, r22, r23, r31, r32, r33, c1, c2,
                       ci1, ci2, cj1, cj2, ei1, ei2, ej1, ej2, fi, fj, fdi,
                       fdj, ls1, ls2, ls3, sig2, coord1, coord2, fdi1, fdi2):

    kern = 0

    if (c1 == c2):
        if (ei1 == ej1) and (ei2 == ej2):
            kern += \
                three_body_sf_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord1, coord2, fdi1, fdi2)
        if (ei1 == ej2) and (ei2 == ej1):
            kern += \
                three_body_sf_1(ci1, ci2, cj2, cj1, r12, r21, r33, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord1, coord2, fdi1, fdi2)
    if (c1 == ej1):
        if (ei1 == ej2) and (ei2 == c2):
            kern += \
                three_body_sf_2(ci2, ci1, cj2, cj1, r21, r13, r32, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord2, coord1, fdi2, fdi1)
        if (ei1 == c2) and (ei2 == ej2):
            kern += \
                three_body_sf_2(ci1, ci2, cj2, cj1, r11, r23, r32, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord1, coord2, fdi1, fdi2)
    if (c1 == ej2):
        if (ei1 == ej1) and (ei2 == c2):
            kern += \
                three_body_sf_2(ci2, ci1, cj1, cj2, r22, r13, r31, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord2, coord1, fdi2, fdi1)
        if (ei1 == c2) and (ei2 == ej1):
            kern += \
                three_body_sf_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord1, coord2, fdi1, fdi2)

    return kern


@njit
def three_body_ss_perm(r11, r12, r13, r21, r22, r23, r31, r32, r33, c1, c2,
                       ci1, ci2, cj1, cj2, ei1, ei2, ej1, ej2, fi, fj, fdi,
                       fdj, ls1, ls2, ls3, sig2, coord1, coord2, coord3,
                       coord4, fdi_p1, fdi_p2, fdj_p1, fdj_p2):

    kern = 0

    if (c1 == c2):
        if (ei1 == ej1) and (ei2 == ej2):
            kern += \
                three_body_ss_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord1, coord2, coord3, coord4, fdi_p1,
                                fdi_p2, fdj_p1, fdj_p2)
        if (ei1 == ej2) and (ei2 == ej1):
            kern += \
                three_body_ss_1(ci1, ci2, cj2, cj1, r12, r21, r33, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord1, coord2, coord4, coord3, fdi_p1,
                                fdi_p2, fdj_p2, fdj_p1)
    if (c1 == ej1):
        if (ei1 == ej2) and (ei2 == c2):
            kern += \
                three_body_ss_2(ci2, ci1, cj2, cj1, r21, r13, r32, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord2, coord1, coord4, coord3, fdi_p2,
                                fdi_p1, fdj_p2, fdj_p1)
        if (ei1 == c2) and (ei2 == ej2):
            kern += \
                three_body_ss_2(ci1, ci2, cj2, cj1, r11, r23, r32, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord1, coord2, coord4, coord3, fdi_p1,
                                fdi_p2, fdj_p2, fdj_p1)
    if (c1 == ej2):
        if (ei1 == ej1) and (ei2 == c2):
            kern += \
                three_body_ss_2(ci2, ci1, cj1, cj2, r22, r13, r31, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord2, coord1, coord3, coord4, fdi_p2,
                                fdi_p1, fdj_p1, fdj_p2)
        if (ei1 == c2) and (ei2 == ej1):
            kern += \
                three_body_ss_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi,
                                fj, fdi, fdj, ls1, ls2, ls3, sig2,
                                coord1, coord2, coord3, coord4, fdi_p1,
                                fdi_p2, fdj_p1, fdj_p2)

    return kern


@njit
def three_body_se_helper(ci1, ci2, r11, r22, r33, fi, fj, fdi, ls1, ls2, sig2,
                         coord1, coord2, fdi1, fdi2):
    p1 = r11 * r11 + r22 * r22 + r33 * r33
    p2 = exp(-p1 * ls1)
    p3 = p2 * ls2 * fi * fj
    p4 = p2 * fj
    f1 = p3 * r11 * ci1 + p4 * fdi1
    f2 = p3 * r22 * ci2 + p4 * fdi2

    s_val = sig2 * (f1 * coord1 + f2 * coord2)

    return s_val


@njit
def three_body_se_perm(r11, r12, r13, r21, r22, r23, r31, r32, r33, c1, c2,
                       ci1, ci2, ei1, ei2, ej1, ej2, fi, fj, fdi, ls1, ls2,
                       sig2, coord1, coord2, fdi1, fdi2):
    kern = 0

    if (c1 == c2):
        if (ei1 == ej1) and (ei2 == ej2):
            kern += three_body_se_helper(ci1, ci2, r11, r22, r33, fi, fj, fdi,
                                         ls1, ls2, sig2, coord1, coord2,
                                         fdi1, fdi2)
        if (ei1 == ej2) and (ei2 == ej1):
            kern += three_body_se_helper(ci1, ci2, r12, r21, r33, fi, fj, fdi,
                                         ls1, ls2, sig2, coord1, coord2,
                                         fdi1, fdi2)
    if (c1 == ej1):
        if (ei1 == ej2) and (ei2 == c2):
            kern += three_body_se_helper(ci1, ci2, r13, r21, r32, fi, fj, fdi,
                                         ls1, ls2, sig2, coord1, coord2,
                                         fdi1, fdi2)
        if (ei1 == c2) and (ei2 == ej2):
            kern += three_body_se_helper(ci1, ci2, r11, r23, r32, fi, fj, fdi,
                                         ls1, ls2, sig2, coord1, coord2,
                                         fdi1, fdi2)
    if (c1 == ej2):
        if (ei1 == ej1) and (ei2 == c2):
            kern += three_body_se_helper(ci1, ci2, r13, r22, r31, fi, fj, fdi,
                                         ls1, ls2, sig2, coord1, coord2,
                                         fdi1, fdi2)
        if (ei1 == c2) and (ei2 == ej1):
            kern += three_body_se_helper(ci1, ci2, r12, r23, r31, fi, fj, fdi,
                                         ls1, ls2, sig2, coord1, coord2,
                                         fdi1, fdi2)

    return kern


@njit
def three_body_grad_perm(r11, r12, r13, r21, r22, r23, r31, r32, r33, c1, c2,
                         ci1, ci2, cj1, cj2, ei1, ei2, ej1, ej2, fi, fj, fdi,
                         fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2, sig3):

    kern = 0
    sig_derv = 0
    ls_derv = 0

    if (c1 == c2):
        if (ei1 == ej1) and (ei2 == ej2):
            kern_term, sig_term, ls_term = \
                three_body_grad_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi,
                                         fj, fdi, fdj, ls1, ls2, ls3, ls4, ls5,
                                         ls6, sig2, sig3)
            kern += kern_term
            sig_derv += sig_term
            ls_derv += ls_term

        if (ei1 == ej2) and (ei2 == ej1):
            kern_term, sig_term, ls_term = \
                three_body_grad_helper_1(ci1, ci2, cj2, cj1, r12, r21, r33, fi,
                                         fj, fdi, fdj, ls1, ls2, ls3, ls4, ls5,
                                         ls6, sig2, sig3)
            kern += kern_term
            sig_derv += sig_term
            ls_derv += ls_term

    if (c1 == ej1):
        if (ei1 == ej2) and (ei2 == c2):
            kern_term, sig_term, ls_term = \
                three_body_grad_helper_2(ci2, ci1, cj2, cj1, r21, r13, r32, fi,
                                         fj, fdi, fdj, ls1, ls2, ls3, ls4, ls5,
                                         ls6, sig2, sig3)
            kern += kern_term
            sig_derv += sig_term
            ls_derv += ls_term

        if (ei1 == c2) and (ei2 == ej2):
            kern_term, sig_term, ls_term = \
                three_body_grad_helper_2(ci1, ci2, cj2, cj1, r11, r23, r32, fi,
                                         fj, fdi, fdj, ls1, ls2, ls3, ls4, ls5,
                                         ls6, sig2, sig3)
            kern += kern_term
            sig_derv += sig_term
            ls_derv += ls_term

    if (c1 == ej2):
        if (ei1 == ej1) and (ei2 == c2):
            kern_term, sig_term, ls_term = \
                three_body_grad_helper_2(ci2, ci1, cj1, cj2, r22, r13, r31, fi,
                                         fj, fdi, fdj, ls1, ls2, ls3, ls4, ls5,
                                         ls6, sig2, sig3)
            kern += kern_term
            sig_derv += sig_term
            ls_derv += ls_term

        if (ei1 == c2) and (ei2 == ej1):
            kern_term, sig_term, ls_term = \
                three_body_grad_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi,
                                         fj, fdi, fdj, ls1, ls2, ls3, ls4, ls5,
                                         ls6, sig2, sig3)

            kern += kern_term
            sig_derv += sig_term
            ls_derv += ls_term

    return kern, sig_derv, ls_derv


# -----------------------------------------------------------------------------
#                        many body helper functions
# -----------------------------------------------------------------------------

@njit


def k_sq_exp_double_dev(q1, q2, sig, ls):
    """Second Gradient of generic squared exponential kernel on two many body functions

    Args:
        q1 (float): the many body descriptor of the first local environment
        q2 (float): the many body descriptor of the second local environment
        sig (float): amplitude hyperparameter
        ls2 (float): squared lenghtscale hyperparameter
    Return:
        float: the value of the double derivative of the squared exponential kernel
    """

    qdiffsq = (q1 - q2) * (q1 - q2)

    ls2 = ls * ls

    ker = exp(-qdiffsq / (2 * ls2))

    ret = sig * sig * ker / ls2 * (1 - qdiffsq / ls2)

    return ret

@njit


def k_sq_exp_dev(q1, q2, sig, ls):
    """First Gradient of generic squared exponential kernel on two many body functions

    Args:
        q1 (float): the many body descriptor of the first local environment
        q2 (float): the many body descriptor of the second local environment
        sig (float): amplitude hyperparameter
        ls2 (float): squared lenghtscale hyperparameter
    Return:
        float: the value of the derivative of the squared exponential kernel
    """

    qdiff = (q1 - q2)

    ls2 = ls * ls

    ker = exp(-qdiff * qdiff / (2 * ls2))

    ret = - sig * sig * ker / ls2 * qdiff

    return ret


@njit
def coordination_number(rij, cij, r_cut, cutoff_func):
    """Pairwise contribution to many-body descriptor based on number of
        atoms in the environment

    Args:
        rij (float): distance between atoms i and j
        cij (float): Component of versor of rij along given direction
        r_cut (float): cutoff hyperparameter
        cutoff_func (callable): cutoff function
    Return:
        float: the value of the pairwise many-body contribution
        float: the value of the derivative of the pairwise many-body
        contribution w.r.t. the central atom displacement
    """

    fij, fdij = cutoff_func(r_cut, rij, cij)

    return fij, fdij


@njit
def q_value(distances, r_cut, cutoff_func, q_func=coordination_number):
    """Compute value of many-body descriptor based on distances of atoms
    in the local many-body environment.

    Args:
        distances (np.ndarray): distances between atoms i and j
        r_cut (float): cutoff hyperparameter
        cutoff_func (callable): cutoff function
        q_func (callable): many-body pairwise descrptor function

    Return:
        float: the value of the many-body descriptor
    """
    q = 0

    for d in distances:
        q_, _ = q_func(d, 0, r_cut, cutoff_func)
        q += q_

    return q

@njit
def q_value_mc(distances, r_cut, ref_species, species, cutoff_func, q_func=coordination_number):
    """Compute value of many-body many components descriptor based
    on distances of atoms in the local many-body environment.

    Args:
        distances (np.ndarray): distances between atoms i and j
        r_cut (float): cutoff hyperparameter
        ref_species (int): species to consider to compute the contribution
        species (np.ndarray): atomic species of neighbours
        cutoff_func (callable): cutoff function
        q_func (callable): many-body pairwise descrptor function

    Return:
        float: the value of the many-body descriptor
    """

    q = 0.0

    for i in range(len(distances)):
        if species[i] == ref_species:
            q_, _ = q_func(distances[i], 0, r_cut, cutoff_func)
            q += q_

    return q


@njit
def q_value_mc(distances, r_cut, ref_species, species, cutoff_func, q_func=coordination_number):
    """Compute value of many-body many components descriptor based
    on distances of atoms in the local many-body environment.

    Args:
        distances (np.ndarray): distances between atoms i and j
        r_cut (float): cutoff hyperparameter
        ref_species (int): species to consider to compute the contribution
        species (np.ndarray): atomic species of neighbours
        cutoff_func (callable): cutoff function
        q_func (callable): many-body pairwise descrptor function

    Return:
        float: the value of the many-body descriptor
    """

    q = 0

    for i in range(len(distances)):
        if species[i] == ref_species:
            q_, _ = q_func(distances[i], 0, r_cut, cutoff_func)
            q += q_

    return q


@njit
def mb_grad_helper_ls_(qdiffsq, sig, ls):
    """
    Derivative of a many body force-force kernel w.r.t. ls
    """

    ls2 = ls * ls

    prefact = exp(-(qdiffsq / (2 * ls2))) * (sig * sig) / ls ** 5

    ret = - prefact * (qdiffsq ** 2 / ls2 - 5 * qdiffsq + 2 * ls2)

    return ret

@njit
def mb_grad_helper_ls(q1, q2, qi, qj, sig, ls):
    """
    Helper function for many body gradient collecting all the derivatives
    of the force-force many body kernel w.r.t. ls
    """

    q12diffsq = ((q1 - q2) * (q1 - q2))
    qijdiffsq = ((qi - qj) * (qi - qj))
    qi2diffsq = ((qi - q2) * (qi - q2))
    q1jdiffsq = ((q1 - qj) * (q1 - qj))

    dk12 = mb_grad_helper_ls_(q12diffsq, sig, ls)
    dkij = mb_grad_helper_ls_(qijdiffsq, sig, ls)
    dki2 = mb_grad_helper_ls_(qi2diffsq, sig, ls)
    dk1j = mb_grad_helper_ls_(q1jdiffsq, sig, ls)

    return dk12 + dkij + dki2 + dk1j

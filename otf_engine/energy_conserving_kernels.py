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
import time


def three_body_cons_quad(env1, env2, bodies, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    return three_body_cons_quad_jit(env1.bond_array, env2.bond_array,
                                    env1.cross_bond_dists,
                                    env2.cross_bond_dists,
                                    d1, d2, sig, ls, r_cut)


def three_body_cons_quad_en(env1, env2, bodies, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    return three_body_cons_quad_en_jit(env1.bond_array, env2.bond_array,
                                       env1.cross_bond_dists,
                                       env2.cross_bond_dists,
                                       sig, ls, r_cut)


@njit
def three_body_cons_quad_jit(bond_array_1, bond_array_2,
                             cross_bond_dists_1, cross_bond_dists_2,
                             d1, d2, sig, ls, r_cut):
    kern = 0

    # pre-compute constants that appear inside the inner loop
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
                cj1 = bond_array_2[p, d2]
                fj1 = rdj1*rdj1
                fdj1 = 2*rdj1*cj1

                for q in range(bond_array_2.shape[0]):
                    if p == q:
                        continue

                    rj2 = bond_array_2[q, 0]
                    rdj2 = r_cut-rj2
                    cj2 = bond_array_2[q, d2]
                    fj2 = rdj2*rdj2
                    fdj2 = 2*rdj2*cj2
                    rj3 = cross_bond_dists_2[p, q]
                    if rj3 > r_cut:
                        continue
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
                    M1 = sig2*(I1+J1+K1+L1)

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
                    M2 = sig2*(I2+J2+K2+L2)

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
                    M3 = sig2*(I3+J3+K3+L3)

                    kern += M1+M2+M3

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
    pass

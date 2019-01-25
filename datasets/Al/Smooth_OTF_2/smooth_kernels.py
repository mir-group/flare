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


def three_body_quad_mc(env1, env2, bodies, d1, d2, hyps, r_cut):
    pass


def two_body_quad_mc(env1, env2, bodies, d1, d2, hyps, r_cut):
    sigs = hyps[0:-1]
    ls = hyps[-1]

    return two_body_quad_mc_jit(env1.bond_array, env1.etyps,
                                env2.bond_array, env2.etyps,
                                d1, d2, sigs, ls, r_cut)


def two_body_quad_mc_grad(env1, env2, bodies, d1, d2, hyps, r_cut):
    sigs = hyps[0:-1]
    ls = hyps[-1]

    kernel, kern_grad = \
        two_body_quad_mc_grad_jit(env1.bond_array, env1.etyps,
                                  env2.bond_array, env2.etyps,
                                  d1, d2, sigs, ls, r_cut)

    return kernel, kern_grad


def three_body_quad_grad(env1, env2, bodies, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    kernel, sig_derv, ls_derv = \
        three_body_quad_grad_jit(env1.bond_array, env2.bond_array,
                                 env1.cross_bond_dists,
                                 env2.cross_bond_dists,
                                 d1, d2, sig, ls, r_cut)

    kernel_grad = np.array([sig_derv, ls_derv])

    return kernel, kernel_grad


def three_body_quad(env1, env2, bodies, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    return three_body_quad_jit(env1.bond_array, env2.bond_array,
                               env1.cross_bond_dists, env2.cross_bond_dists,
                               d1, d2, sig, ls, r_cut)


def three_body_quad_en(env1, env2, bodies, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    return three_body_quad_en_jit(env1.bond_array, env2.bond_array,
                                  env1.cross_bond_dists, env2.cross_bond_dists,
                                  sig, ls, r_cut)


def two_body_quad(env1, env2, bodies, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    return two_body_quad_jit(env1.bond_array, env2.bond_array,
                             d1, d2, sig, ls, r_cut)


def two_body_quad_grad(env1, env2, bodies, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]

    kernel, ls_derv, sig_derv = \
        two_body_quad_grad_jit(env1.bond_array, env2.bond_array,
                               d1, d2, sig, ls, r_cut)
    kernel_grad = np.array([sig_derv, ls_derv])
    return kernel, kernel_grad


def two_body_smooth(env1, env2, bodies, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]
    d = hyps[2]

    return two_body_smooth_jit(env1.bond_array, env2.bond_array,
                               d1, d2, sig, ls, r_cut, d)


def two_body_smooth_grad(env1, env2, bodies, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]
    d = hyps[2]

    kernel, ls_derv, sig_derv, d_derv = \
        two_body_smooth_grad_jit(env1.bond_array, env2.bond_array,
                                 d1, d2, sig, ls, r_cut, d)
    kernel_grad = np.array([sig_derv, ls_derv, d_derv])
    return kernel, kernel_grad


def two_body_smooth_en(env1, env2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]
    d = hyps[2]

    kernel = two_body_smooth_en_jit(env1.bond_array, env2.bond_array, sig, ls,
                                    r_cut, d)

    return kernel


# -----------------------------------------------------------------------------
#                             njit kernels and gradients
# -----------------------------------------------------------------------------


@njit
def three_body_quad_mc_jit(bond_array_1, bond_array_2,
                           environment_species_1, environment_species_2,
                           cross_bond_dists_1, cross_bond_dists_2,
                           d1, d2, sigs, ls, r_cut):
    kern = 0

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1 = (r_cut-ri1)**2
        fdi1 = 2*(r_cut-ri1)*ci1
        ctyp1 = environment_species_1[m]

        for n in range(m+1, bond_array_1.shape[0]):
            ri2 = bond_array_1[n, 0]
            ci2 = bond_array_1[n, d1]
            fi2 = (r_cut-ri2)**2
            fdi2 = 2*(r_cut-ri2)*ci2
            ctyp2 = environment_species_1[n]
            ri3 = cross_bond_dists_1[m, n]

            fi = fi1*fi2
            fdi = fdi1*fi2+fi1*fdi2

            for p in range(bond_array_2.shape[0]):
                ptyp1 = environment_species_2[p]
                if ctyp1 != ptyp1:
                    continue
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                fj1 = (r_cut-rj1)**2
                fdj1 = 2*(r_cut-rj1)*cj1

                for q in range(bond_array_2.shape[0]):
                    if q == p:
                        continue

                    ptyp2 = environment_species_2[p]
                    if ctyp2 != ptyp2:
                        continue

                    rj2 = bond_array_2[q, 0]
                    cj2 = bond_array_2[q, d2]
                    fj2 = (r_cut-rj2)**2
                    fdj2 = 2*(r_cut-rj2)*cj2
                    rj3 = cross_bond_dists_2[p, q]

                    fj = fj1*fj2
                    fdj = fdj1*fj2+fj1*fdj2

                    r11 = ri1-rj1
                    r22 = ri2-rj2
                    r33 = ri3-rj3

                    A = ci1*cj1+ci2*cj2
                    B1 = r11*ci1+r22*ci2
                    B2 = r11*cj1+r22*cj2
                    C = r11*r11+r22*r22+r33*r33

                    k = exp(-C/(2*ls*ls))
                    dk_1 = k*B1/(ls*ls)
                    dk_2 = -k*B2/(ls*ls)
                    dk_dk = A*k/(ls*ls) - B1*B2*k/(ls**4)

                    k0 = k*fdi*fdj
                    k1 = dk_1*fi*fdj
                    k2 = dk_2*fdi*fj
                    k3 = dk_dk*fi*fj

                    sig = sigs[ctyp1, ctyp2]
                    kern += sig*sig*(k0+k1+k2+k3)

    return kern


@njit
def two_body_quad_mc_grad_jit(bond_array_1, environment_species_1,
                              bond_array_2, environment_species_2,
                              d1, d2, sigs, ls, r_cut):

    kern = 0
    nos = sigs.shape[0]
    sig_derv = np.zeros(nos)
    ls_derv = 0
    kern_grad = np.zeros(nos+1)

    ls2 = ls*ls
    ls3 = ls*ls2
    ls4 = ls2*ls2
    ls5 = ls2*ls3

    for m in range(bond_array_1.shape[0]):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]
        ctyp = environment_species_1[m]

        # get cutoff factors
        fcut1 = (r_cut - r1)**2
        fderv1 = 2 * (r_cut - r1) * coord1

        for n in range(bond_array_2.shape[0]):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]
            ptyp = environment_species_2[n]

            if ctyp == ptyp:
                # get cutoff factors
                sig = sigs[ctyp]
                fcut2 = (r_cut - r2)**2
                fderv2 = 2 * (r_cut - r2) * coord2

                rdiff = r1 - r2

                A = coord1 * coord2
                B_1 = rdiff * coord1
                B_2 = rdiff * coord2
                C = rdiff * rdiff

                # kernel terms
                k = exp(-C / (2*ls2))
                dk_1 = k*B_1/ls2
                dk_2 = -k*B_2/ls2
                dk_dk = (A*ls2-B_1*B_2)*k/ls4

                k0 = k*fderv1*fderv2
                k1 = dk_1*fcut1*fderv2
                k2 = dk_2*fderv1*fcut2
                k3 = dk_dk*fcut1*fcut2

                ksum = k0+k1+k2+k3
                kern += sig*sig*ksum
                sig_derv[ctyp] += 2*sig*ksum

                # l terms
                dk_dl = k*C/ls3
                dk_dl_xi = B_1*(dk_dl/ls2 - 2*k/ls3)
                dk_dl_chi = -B_2*(dk_dl/ls2 - 2*k/ls3)
                dk_dl_trip = \
                    (A*ls2-B_1*B_2)*(dk_dl/ls4 - 4*k/ls5) + \
                    2*k*A/ls3

                l0 = dk_dl*fderv1*fderv2
                l1 = dk_dl_xi*fcut1*fderv2
                l2 = dk_dl_chi*fderv1*fcut2
                l3 = dk_dl_trip*fcut1*fcut2

                ls_derv += sig*sig*(l0+l1+l2+l3)

    kern_grad[0:-1] = sig_derv
    kern_grad[-1] = ls_derv
    return kern, kern_grad


@njit
def two_body_quad_mc_jit(bond_array_1, environment_species_1,
                         bond_array_2, environment_species_2,
                         d1, d2, sigs, ls, r_cut):
    kern = 0
    ls2 = ls*ls
    ls4 = ls2*ls2

    for m in range(bond_array_1.shape[0]):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]

        # get cutoff factors
        fcut1 = (r_cut - r1)**2
        fderv1 = 2 * (r_cut - r1) * coord1

        # get species
        ctyp = environment_species_1[m]

        for n in range(bond_array_2.shape[0]):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]

            # get cutoff factors
            fcut2 = (r_cut - r2)**2
            fderv2 = 2 * (r_cut - r2) * coord2

            # get species
            ptyp = environment_species_2[n]

            # add to kernel only if species are the same
            if ctyp == ptyp:
                sig = sigs[ctyp]

                rdiff = r1 - r2

                A = coord1 * coord2
                B_1 = rdiff * coord1
                B_2 = rdiff * coord2
                C = rdiff * rdiff

                # kernel terms
                k = exp(-C / (2*ls2))
                dk_1 = k*B_1/ls2
                dk_2 = -k*B_2/ls2
                dk_dk = (A*ls2-B_1*B_2)*k/ls4

                k0 = k*fderv1*fderv2
                k1 = dk_1*fcut1*fderv2
                k2 = dk_2*fderv1*fcut2
                k3 = dk_dk*fcut1*fcut2

                ksum = k0+k1+k2+k3
                kern += sig*sig*ksum
    return kern


@njit
def three_body_quad_grad_jit(bond_array_1, bond_array_2,
                             cross_bond_dists_1, cross_bond_dists_2,
                             d1, d2, sig, ls, r_cut):
    kern = 0
    sig_derv = 0
    ls_derv = 0

    sig2 = sig*sig
    ls2 = ls*ls
    ls3 = ls*ls*ls
    ls4 = ls2*ls2
    ls5 = ls2*ls3

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1 = (r_cut-ri1)**2
        fdi1 = 2*(r_cut-ri1)*ci1

        for n in range(m+1, bond_array_1.shape[0]):
            ri2 = bond_array_1[n, 0]
            ci2 = bond_array_1[n, d1]
            fi2 = (r_cut-ri2)**2
            fdi2 = 2*(r_cut-ri2)*ci2
            ri3 = cross_bond_dists_1[m, n]

            fi = fi1*fi2
            fdi = fdi1*fi2+fi1*fdi2

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                fj1 = (r_cut-rj1)**2
                fdj1 = 2*(r_cut-rj1)*cj1

                for q in range(bond_array_2.shape[0]):
                    if q == p:
                        continue

                    rj2 = bond_array_2[q, 0]
                    cj2 = bond_array_2[q, d2]
                    fj2 = (r_cut-rj2)**2
                    fdj2 = 2*(r_cut-rj2)*cj2
                    rj3 = cross_bond_dists_2[p, q]

                    fj = fj1*fj2
                    fdj = fdj1*fj2+fj1*fdj2

                    r11 = ri1-rj1
                    r22 = ri2-rj2
                    r33 = ri3-rj3

                    A = ci1*cj1+ci2*cj2
                    B1 = r11*ci1+r22*ci2
                    B2 = r11*cj1+r22*cj2
                    C = r11*r11+r22*r22+r33*r33

                    k = exp(-C/(2*ls2))
                    dk_1 = k*B1/(ls2)
                    dk_2 = -k*B2/(ls2)
                    dk_dk = A*k/(ls2) - B1*B2*k/ls4

                    k0 = k*fdi*fdj
                    k1 = dk_1*fi*fdj
                    k2 = dk_2*fdi*fj
                    k3 = dk_dk*fi*fj

                    k_sum = k0+k1+k2+k3
                    kern += sig2*k_sum
                    sig_derv += 2*sig*k_sum

                    dk_dl = k*C/ls3
                    dk_dl_xi = B1*(dk_dl/ls2 - 2*k/ls3)
                    dk_dl_chi = -B2*(dk_dl/ls2 - 2*k/ls3)
                    dk_dl_trip = \
                        (A*ls2-B1*B2)*(dk_dl/ls4 - 4*k/ls5) + \
                        2*k*A/ls3

                    l0 = dk_dl*fdi*fdj
                    l1 = dk_dl_xi*fi*fdj
                    l2 = dk_dl_chi*fdi*fj
                    l3 = dk_dl_trip*fi*fj

                    ls_derv += sig2*(l0+l1+l2+l3)

    return kern, sig_derv, ls_derv


@njit
def three_body_quad_jit(bond_array_1, bond_array_2,
                        cross_bond_dists_1, cross_bond_dists_2,
                        d1, d2, sig, ls, r_cut):
    kern = 0

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1 = (r_cut-ri1)**2
        fdi1 = 2*(r_cut-ri1)*ci1

        for n in range(m+1, bond_array_1.shape[0]):
            ri2 = bond_array_1[n, 0]
            ci2 = bond_array_1[n, d1]
            fi2 = (r_cut-ri2)**2
            fdi2 = 2*(r_cut-ri2)*ci2
            ri3 = cross_bond_dists_1[m, n]

            fi = fi1*fi2
            fdi = fdi1*fi2+fi1*fdi2

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                fj1 = (r_cut-rj1)**2
                fdj1 = 2*(r_cut-rj1)*cj1

                for q in range(bond_array_2.shape[0]):
                    if q == p:
                        continue

                    rj2 = bond_array_2[q, 0]
                    cj2 = bond_array_2[q, d2]
                    fj2 = (r_cut-rj2)**2
                    fdj2 = 2*(r_cut-rj2)*cj2
                    rj3 = cross_bond_dists_2[p, q]

                    fj = fj1*fj2
                    fdj = fdj1*fj2+fj1*fdj2

                    r11 = ri1-rj1
                    r22 = ri2-rj2
                    r33 = ri3-rj3

                    A = ci1*cj1+ci2*cj2
                    B1 = r11*ci1+r22*ci2
                    B2 = r11*cj1+r22*cj2
                    C = r11*r11+r22*r22+r33*r33

                    k = exp(-C/(2*ls*ls))
                    dk_1 = k*B1/(ls*ls)
                    dk_2 = -k*B2/(ls*ls)
                    dk_dk = A*k/(ls*ls) - B1*B2*k/(ls**4)

                    k0 = k*fdi*fdj
                    k1 = dk_1*fi*fdj
                    k2 = dk_2*fdi*fj
                    k3 = dk_dk*fi*fj

                    kern += sig*sig*(k0+k1+k2+k3)

    return kern


@njit
def three_body_quad_en_jit(bond_array_1, bond_array_2,
                           cross_bond_dists_1, cross_bond_dists_2,
                           sig, ls, r_cut):
    kern = 0

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        fi1 = (r_cut-ri1)**2

        for n in range(m+1, bond_array_1.shape[0]):
            ri2 = bond_array_1[n, 0]
            fi2 = (r_cut-ri2)**2
            ri3 = cross_bond_dists_1[m, n]

            fi = fi1*fi2

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1 = (r_cut-rj1)**2

                for q in range(bond_array_2.shape[0]):
                    if q == p:
                        continue

                    rj2 = bond_array_2[q, 0]
                    fj2 = (r_cut-rj2)**2
                    rj3 = cross_bond_dists_2[p, q]

                    fj = fj1*fj2

                    r11 = rj1-ri1
                    r22 = rj2-ri2
                    r33 = rj3-ri3

                    C = r11*r11+r22*r22+r33*r33
                    k = exp(-C/(2*ls*ls))

                    k0 = k*fi*fj

                    kern += sig*sig*k0

    return kern


@njit
def two_body_quad_jit(bond_array_1, bond_array_2, d1, d2, sig, ls,
                      r_cut):
    kern = 0

    ls2 = ls*ls
    ls4 = ls2*ls2
    sig2 = sig*sig

    for m in range(bond_array_1.shape[0]):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]

        # get cutoff factors
        fcut1 = (r_cut - r1)**2
        fderv1 = 2 * (r_cut - r1) * coord1

        for n in range(bond_array_2.shape[0]):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]

            # get cutoff factors
            fcut2 = (r_cut - r2)**2
            fderv2 = 2 * (r_cut - r2) * coord2

            rdiff = r1 - r2

            A = coord1 * coord2
            B_1 = rdiff * coord1
            B_2 = rdiff * coord2
            C = rdiff * rdiff

            # kernel terms
            k = exp(-C / (2*ls2))
            dk_1 = k*B_1/ls2
            dk_2 = -k*B_2/ls2
            dk_dk = (A*ls2-B_1*B_2)*k/ls4

            k0 = k*fderv1*fderv2
            k1 = dk_1*fcut1*fderv2
            k2 = dk_2*fderv1*fcut2
            k3 = dk_dk*fcut1*fcut2

            ksum = k0+k1+k2+k3
            kern += sig2*ksum
    return kern


@njit
def two_body_quad_grad_jit(bond_array_1, bond_array_2, d1, d2, sig, ls,
                           r_cut):

    kern = 0
    sig_derv = 0
    ls_derv = 0

    ls2 = ls*ls
    ls3 = ls*ls2
    ls4 = ls2*ls2
    ls5 = ls2*ls3

    sig2 = sig*sig

    for m in range(bond_array_1.shape[0]):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]

        # get cutoff factors
        fcut1 = (r_cut - r1)**2
        fderv1 = 2 * (r_cut - r1) * coord1

        for n in range(bond_array_2.shape[0]):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]

            # get cutoff factors
            fcut2 = (r_cut - r2)**2
            fderv2 = 2 * (r_cut - r2) * coord2

            rdiff = r1 - r2

            A = coord1 * coord2
            B_1 = rdiff * coord1
            B_2 = rdiff * coord2
            C = rdiff * rdiff

            # kernel terms
            k = exp(-C / (2*ls2))
            dk_1 = k*B_1/ls2
            dk_2 = -k*B_2/ls2
            dk_dk = (A*ls2-B_1*B_2)*k/ls4

            k0 = k*fderv1*fderv2
            k1 = dk_1*fcut1*fderv2
            k2 = dk_2*fderv1*fcut2
            k3 = dk_dk*fcut1*fcut2

            ksum = k0+k1+k2+k3
            kern += sig2*ksum
            sig_derv += 2*sig*ksum

            # l terms
            dk_dl = k*C/ls3
            dk_dl_xi = B_1*(dk_dl/ls2 - 2*k/ls3)
            dk_dl_chi = -B_2*(dk_dl/ls2 - 2*k/ls3)
            dk_dl_trip = \
                (A*ls2-B_1*B_2)*(dk_dl/ls4 - 4*k/ls5) + \
                2*k*A/ls3

            l0 = dk_dl*fderv1*fderv2
            l1 = dk_dl_xi*fcut1*fderv2
            l2 = dk_dl_chi*fderv1*fcut2
            l3 = dk_dl_trip*fcut1*fcut2

            ls_derv += sig2*(l0+l1+l2+l3)

    return kern, ls_derv, sig_derv


@njit
def two_body_smooth_grad_jit(bond_array_1, bond_array_2, d1, d2, sig, ls,
                             r_cut, d):

    kern = 0
    sig_derv = 0
    ls_derv = 0
    d_derv = 0

    for m in range(bond_array_1.shape[0]):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]

        # check if r1 is in the smoothed region
        if r1 > r_cut - d:
            fcut1 = (1/2) * (np.cos(np.pi * (r1-r_cut+d) / d) + 1)
            fderv1 = (np.pi/(2*d)) * np.sin(np.pi*(r_cut-r1) / d) * coord1
            dterm11 = (np.pi*(r1-r_cut)/(2*d*d))*np.sin(np.pi*(r_cut-r1) / d)
            dterm12 = (np.pi*coord1/(2*d**3)) * \
                (np.pi*(r1-r_cut) *
                 np.cos(np.pi * (r1-r_cut) / d) +
                 d*np.sin(np.pi*(r1-r_cut) / d))
        else:
            fcut1 = 1
            fderv1 = 0
            dterm11 = 0
            dterm12 = 0

        for n in range(bond_array_2.shape[0]):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]

            # check if r2 is in the smoothed region
            if r2 > r_cut - d:
                fcut2 = (1/2) * (np.cos(np.pi * (r2-r_cut+d) / d) + 1)
                fderv2 = (np.pi/(2*d)) * np.sin(np.pi*(r_cut-r2) / d) * coord2
                dterm21 = (np.pi*(r2-r_cut)/(2*d*d)) * \
                    np.sin(np.pi*(r_cut-r2) / d)
                dterm22 = (np.pi*coord2/(2*d**3)) * \
                    (np.pi*(r2-r_cut) *
                     np.cos(np.pi * (r2-r_cut) / d) +
                     d*np.sin(np.pi*(r2-r_cut) / d))
            else:
                fcut2 = 1
                fderv2 = 0
                dterm21 = 0
                dterm22 = 0

            rdiff = r1 - r2

            A = coord1 * coord2
            B_1 = rdiff * coord1
            B_2 = -rdiff * coord2
            B_cp = rdiff * coord1 * rdiff * coord2
            C_cp = rdiff * rdiff

            # kernel terms
            kern0 = sig*sig*exp(-C_cp / (2*ls*ls))
            kern1 = kern0 * B_1 / (ls*ls)
            kern2 = kern0 * B_2 / (ls*ls)
            kern3 = kern0 * (A*ls*ls-B_cp) / (ls**4)

            # ls derv terms
            ls0 = kern0*rdiff*rdiff/(ls*ls*ls)
            ls1 = kern0*coord1*(-2*ls*ls+rdiff*rdiff)*rdiff/(ls**5)
            ls2 = -kern0*coord2*(-2*ls*ls+rdiff*rdiff)*rdiff/(ls**5)
            ls3 = kern0*coord1*coord2 * \
                (-2*ls**4+5*ls*ls*rdiff*rdiff-rdiff**4)/(ls**7)

            kern_val = kern3*fcut1*fcut2 + kern2*fderv1*fcut2 +\
                kern1*fderv2*fcut1 + kern0*fderv1*fderv2

            kern += kern_val
            ls_derv += ls3*fcut1*fcut2 + ls2*fderv1*fcut2 +\
                ls1*fderv2*fcut1 + ls0*fderv1*fderv2
            sig_derv += 2*kern_val/sig
            d_derv += kern3*(dterm11*fcut2+fcut1*dterm21) + \
                kern2*(dterm12*fcut2+fderv1*dterm21) + \
                kern1*(dterm11*fderv2+fcut1*dterm22) + \
                kern0*(dterm12*fderv2+fderv1*dterm22)

    return kern, ls_derv, sig_derv, d_derv


@njit
def two_body_smooth_jit(bond_array_1, bond_array_2, d1, d2, sig, ls,
                        r_cut, d):
    kern = 0

    for m in range(bond_array_1.shape[0]):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]

        # check if r1 is in the smoothed region
        if r1 > r_cut - d:
            fcut1 = (1/2) * (np.cos(np.pi * (r1-r_cut+d) / d) + 1)
            fderv1 = (np.pi/(2*d)) * np.sin(np.pi*(r_cut-r1) / d) * coord1
        else:
            fcut1 = 1
            fderv1 = 0

        for n in range(bond_array_2.shape[0]):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]

            # check if r2 is in the smoothed region
            if r2 > r_cut - d:
                fcut2 = (1/2) * (np.cos(np.pi * (r2-r_cut+d) / d) + 1)
                fderv2 = (np.pi/(2*d)) * np.sin(np.pi*(r_cut-r2) / d) * coord2
            else:
                fcut2 = 1
                fderv2 = 0

            rdiff = r1 - r2

            A = coord1 * coord2
            B_1 = rdiff * coord1
            B_2 = -rdiff * coord2
            B_cp = rdiff * coord1 * rdiff * coord2
            C_cp = rdiff * rdiff

            kern0 = sig*sig*exp(-C_cp / (2*ls*ls))
            kern1 = kern0 * B_1 / (ls*ls)
            kern2 = kern0 * B_2 / (ls*ls)
            kern3 = kern0 * (A*ls*ls-B_cp) / (ls**4)

            kern += kern3*fcut1*fcut2 + kern2*fderv1*fcut2 +\
                kern1*fderv2*fcut1 + kern0*fderv1*fderv2
    return kern


@njit
def two_body_smooth_en_jit(bond_array_1, bond_array_2, sig, ls,
                           r_cut, d):
    kern = 0

    for m in range(bond_array_1.shape[0]):
        r1 = bond_array_1[m, 0]

        # check if r1 is in the smoothed region
        if r1 > r_cut - d:
            fcut1 = (1/2) * (np.cos(np.pi * (r1-r_cut+d) / d) + 1)
        else:
            fcut1 = 1

        for n in range(bond_array_2.shape[0]):
            r2 = bond_array_2[n, 0]

            # check if r2 is in the smoothed region
            if r2 > r_cut - d:
                fcut2 = (1/2) * (np.cos(np.pi * (r2-r_cut+d) / d) + 1)
            else:
                fcut2 = 1

            rdiff = r1 - r2
            C_cp = rdiff * rdiff
            kern0 = sig*sig*exp(-C_cp / (2*ls*ls))
            kern += kern0*fcut1*fcut2

    return kern


if __name__ == '__main__':
    pass

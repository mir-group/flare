import numpy as np
# from numba import njit
from math import exp
import flare.cutoffs as cf
from flare.kernels.kernels import force_helper, grad_constants, grad_helper, \
    force_energy_helper, three_body_en_helper, three_body_helper_1, \
    three_body_helper_2, three_body_grad_helper_1, three_body_grad_helper_2

# @njit
def three_body_mc_sepcut_jit(bond_array_1, c1, etypes1,
                             bond_array_2, c2, etypes2,
                             cross_bond_inds_1, cross_bond_inds_2,
                             cross_bond_dists_1, cross_bond_dists_2,
                             triplets_1, triplets_2,
                             d1, d2, sig, ls, r_cut, cutoff_func,
                             nspec, spec_mask, bond_mask, triplet_mask, cut3b_mask):
    '''
    bond_mask is for multiple cutoffs
    '''

    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2

    bci = spec_mask[ci]
    bcin = bci * nspec
    bcj = spec_mask[cj]
    bcjn = bcj * nspec

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        btype_ei1 = bond_mask[bei1 + bcin]
        cut_ei1 = r_cut[btype_ei1]
        bei1n = nspec * bei1
        fi1, fdi1 = cutoff_func(cut_ei1, ri1, ci1)

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]
            ei2 = etypes1[ind1]

            # skip if species does not match
            tr_spec = [c1, ei1, ei2]
            c2_ind = tr_spec
            if c2 not in tr_spec:
                continue
            else:
                tr_spec.remove(c2)

            bei2 = spec_mask[ei2]
            btype_ei2 = bond_mask[bei2 + bcin]
            cut_ei2 = r_cut[btype_ei2]
            fi2, fdi2 = cutoff_func(cut_ei2, ri2, ci2)

            ttypei = triplet_mask[bcin + bei1n + bei2]
            tls1 = ls1[ttypei]
            tls2 = ls2[ttypei]
            tls3 = ls3[ttypei]
            tsig2 = sig2[ttypei]

            btype_ei3 = bond_mask[bei1n + bei2]
            cut_ei3 = r_cut[btype_ei3]
            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(cut_ei3, ri3, 0)

            fi = fi1 * fi2 * fi3
            fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                ej1 = etypes2[p]

                if ej1 not in tr_spec:
                    continue
                else:
                    tr_spec.remove(ej1)

                bej1 = spec_mask[ej1]
                btype_ej1 = bond_mask[bej1 + bcjn]
                cut_ej1 = r_cut[btype_ej1]
                bej1n = nspec * bej1

                fj1, fdj1 = cutoff_func(cut_ej1, rj1, cj1)

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + 1 + q]
                    rj2 = bond_array_2[ind2, 0]
                    cj2 = bond_array_2[ind2, d2]
                    ej2 = etypes2[ind2]

                    if ej2 != tr_spec[0]:
                        continue

                    bej2 = spec_mask[ej2]
                    btype_ej2 = bond_mask[bej2 + bcjn]
                    cut_ej2 = r_cut[btype_ej2]

                    fj2, fdj2 = cutoff_func(cut_ej2, rj2, cj2)

                    btype_ej3 = bond_mask[bej1n + bej2]
                    cut_ej3 = r_cut[btype_ej3]
                    rj3 = cross_bond_dists_2[p, p + 1 + q]
                    fj3, _ = cutoff_func(cut_ej3, rj3, 0)

                    fj = fj1 * fj2 * fj3
                    fdj = fdj1 * fj2 * fj3 + fj1 * fdj2 * fj3

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    if (c1 == c2):
                        if (ei1 == ej1) and (ei2 == ej2):
                            kern += \
                                three_body_helper_1(ci1, ci2, cj1, cj2, r11,
                                                    r22, r33, fi, fj, fdi, fdj,
                                                    tls1, tls2, tls3,
                                                    tsig2)
                        if (ei1 == ej2) and (ei2 == ej1):
                            kern += \
                                three_body_helper_1(ci1, ci2, cj2, cj1, r12,
                                                    r21, r33, fi, fj, fdi, fdj,
                                                    tls1, tls2, tls3,
                                                    tsig2)
                    if (c1 == ej1):
                        if (ei1 == ej2) and (ei2 == c2):
                            kern += \
                                three_body_helper_2(ci2, ci1, cj2, cj1, r21,
                                                    r13, r32, fi, fj, fdi,
                                                    fdj,
                                                    tls1, tls2, tls3,
                                                    tsig2)
                        if (ei1 == c2) and (ei2 == ej2):
                            kern += \
                                three_body_helper_2(ci1, ci2, cj2, cj1, r11,
                                                    r23, r32, fi, fj, fdi,
                                                    fdj,
                                                    tls1, tls2, tls3,
                                                    tsig2)
                    if (c1 == ej2):
                        if (ei1 == ej1) and (ei2 == c2):
                            kern += \
                                three_body_helper_2(ci2, ci1, cj1, cj2, r22,
                                                    r13, r31, fi, fj, fdi,
                                                    fdj,
                                                    tls1, tls2, tls3,
                                                    tsig2)
                        if (ei1 == c2) and (ei2 == ej1):
                            kern += \
                                three_body_helper_2(ci1, ci2, cj1, cj2, r12,
                                                    r23, r31, fi, fj, fdi,
                                                    fdj,
                                                    tls1, tls2, tls3,
                                                    tsig2)

    return kern

# @njit
def three_body_mc_grad_sepcut_jit(bond_array_1, c1, etypes1,
                                  bond_array_2, c2, etypes2,
                                  cross_bond_inds_1, cross_bond_inds_2,
                                  cross_bond_dists_1, cross_bond_dists_2,
                                  triplets_1, triplets_2,
                                  d1, d2, sig, ls, r_cut, cutoff_func,
                                  nspec, spec_mask, ntriplet, triplet_mask):
    """Kernel gradient for 3-body force comparisons."""

    kern = 0
    sig_derv = np.zeros(ntriplet)
    ls_derv = np.zeros(ntriplet)
    kern_grad = np.zeros(2)

    # pre-compute constants that appear in the inner loop
    sig2, sig3, ls1, ls2, ls3, ls4, ls5, ls6 = grad_constants(sig, ls)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec * nspec

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        bei1n = bei1 * nspec

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri3 = cross_bond_dists_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]
            ei2 = etypes1[ind1]

            bei2 = spec_mask[ei2]

            ttypei = triplet_mask[bc1n + bei1n + bei2]

            tls1 = ls1[ttypei]
            tls2 = ls2[ttypei]
            tls3 = ls3[ttypei]
            tls4 = ls4[ttypei]
            tls5 = ls5[ttypei]
            tls6 = ls6[ttypei]
            tsig2 = sig2[ttypei]
            tsig3 = sig3[ttypei]
            tr_cut = r_cut[ttypei]

            fi1, fdi1 = cutoff_func(tr_cut, ri1, ci1)
            fi2, fdi2 = cutoff_func(tr_cut, ri2, ci2)
            fi3, _ = cutoff_func(tr_cut, ri3, 0)

            fi = fi1 * fi2 * fi3
            fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                ej1 = etypes2[p]
                fj1, fdj1 = cutoff_func(tr_cut, rj1, cj1)

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    cj2 = bond_array_2[ind2, d2]
                    ej2 = etypes2[ind2]

                    fj2, fdj2 = cutoff_func(tr_cut, rj2, cj2)
                    fj3, _ = cutoff_func(tr_cut, rj3, 0)

                    fj = fj1 * fj2 * fj3
                    fdj = fdj1 * fj2 * fj3 + fj1 * fdj2 * fj3

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    if (c1 == c2):
                        if (ei1 == ej1) and (ei2 == ej2):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_1(ci1, ci2, cj1, cj2,
                                                         r11, r22, r33, fi, fj,
                                                         fdi, fdj, tls1, tls2,
                                                         tls3, tls4, tls5,
                                                         tls6,
                                                         tsig2, tsig3)
                            kern += kern_term
                            sig_derv[ttypei] += sig_term
                            ls_derv[ttypei] += ls_term

                        if (ei1 == ej2) and (ei2 == ej1):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_1(ci1, ci2, cj2, cj1,
                                                         r12, r21, r33, fi, fj,
                                                         fdi, fdj, tls1, tls2,
                                                         tls3, tls4, tls5,
                                                         tls6,
                                                         tsig2, tsig3)
                            kern += kern_term
                            sig_derv[ttypei] += sig_term
                            ls_derv[ttypei] += ls_term

                    if (c1 == ej1):
                        if (ei1 == ej2) and (ei2 == c2):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_2(ci2, ci1, cj2, cj1,
                                                         r21, r13, r32, fi, fj,
                                                         fdi, fdj, tls1, tls2,
                                                         tls3, tls4, tls5,
                                                         tls6,
                                                         tsig2, tsig3)
                            kern += kern_term
                            sig_derv[ttypei] += sig_term
                            ls_derv[ttypei] += ls_term

                        if (ei1 == c2) and (ei2 == ej2):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_2(ci1, ci2, cj2, cj1,
                                                         r11, r23, r32, fi, fj,
                                                         fdi, fdj, tls1, tls2,
                                                         tls3, tls4, tls5,
                                                         tls6,
                                                         tsig2, tsig3)
                            kern += kern_term
                            sig_derv[ttypei] += sig_term
                            ls_derv[ttypei] += ls_term

                    if (c1 == ej2):
                        if (ei1 == ej1) and (ei2 == c2):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_2(ci2, ci1, cj1, cj2,
                                                         r22, r13, r31, fi, fj,
                                                         fdi, fdj, tls1, tls2,
                                                         tls3, tls4, tls5,
                                                         tls6,
                                                         tsig2, tsig3)
                            kern += kern_term
                            sig_derv[ttypei] += sig_term
                            ls_derv[ttypei] += ls_term

                        if (ei1 == c2) and (ei2 == ej1):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_2(ci1, ci2, cj1, cj2,
                                                         r12, r23, r31, fi, fj,
                                                         fdi, fdj, tls1, tls2,
                                                         tls3, tls4, tls5,
                                                         tls6,
                                                         tsig2, tsig3)

                            kern += kern_term
                            sig_derv[ttypei] += sig_term
                            ls_derv[ttypei] += ls_term

    kern_grad = np.hstack([sig_derv, ls_derv])

    return kern, kern_grad

# @njit
def three_body_mc_force_en_sepcut_jit(bond_array_1, c1, etypes1,
                                      bond_array_2, c2, etypes2,
                                      cross_bond_inds_1, cross_bond_inds_2,
                                      cross_bond_dists_1, cross_bond_dists_2,
                                      triplets_1, triplets_2,
                                      d1, sig, ls, r_cut, cutoff_func,
                                      nspec, spec_mask, triplet_mask):
    """Kernel for 3-body force/energy comparisons."""

    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)

    bc1 = spec_mask[c1]
    bc1n = nspec * nspec * bc1

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        bei1n = nspec * bei1

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]
            ei2 = etypes1[ind1]

            bei2 = spec_mask[ei2]

            ttypei = triplet_mask[bc1n + bei1n + bei2]

            tls1 = ls1[ttypei]
            tls2 = ls2[ttypei]
            tsig2 = sig2[ttypei]
            tr_cut = r_cut[ttypei]

            fi2, fdi2 = cutoff_func(tr_cut, ri2, ci2)
            fi1, fdi1 = cutoff_func(tr_cut, ri1, ci1)
            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(tr_cut, ri3, 0)

            fi = fi1 * fi2 * fi3
            fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(tr_cut, rj1, 0)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    fj2, _ = cutoff_func(tr_cut, rj2, 0)
                    ej2 = etypes2[ind2]
                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    fj3, _ = cutoff_func(tr_cut, rj3, 0)
                    fj = fj1 * fj2 * fj3

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    if (c1 == c2):
                        if (ei1 == ej1) and (ei2 == ej2):
                            kern += three_body_en_helper(ci1, ci2, r11, r22,
                                                         r33, fi, fj, fdi,
                                                         tls1,
                                                         tls2, tsig2)
                        if (ei1 == ej2) and (ei2 == ej1):
                            kern += three_body_en_helper(ci1, ci2, r12, r21,
                                                         r33, fi, fj, fdi,
                                                         tls1,
                                                         tls2, tsig2)
                    if (c1 == ej1):
                        if (ei1 == ej2) and (ei2 == c2):
                            kern += three_body_en_helper(ci1, ci2, r13, r21,
                                                         r32, fi, fj, fdi,
                                                         tls1,
                                                         tls2, tsig2)
                        if (ei1 == c2) and (ei2 == ej2):
                            kern += three_body_en_helper(ci1, ci2, r11, r23,
                                                         r32, fi, fj, fdi,
                                                         tls1,
                                                         tls2, tsig2)
                    if (c1 == ej2):
                        if (ei1 == ej1) and (ei2 == c2):
                            kern += three_body_en_helper(ci1, ci2, r13, r22,
                                                         r31, fi, fj, fdi,
                                                         tls1,
                                                         tls2, tsig2)
                        if (ei1 == c2) and (ei2 == ej1):
                            kern += three_body_en_helper(ci1, ci2, r12, r23,
                                                         r31, fi, fj, fdi,
                                                         tls1,
                                                         tls2, tsig2)

    return kern

# @njit
def three_body_mc_en_sepcut_jit(bond_array_1, c1, etypes1,
                                bond_array_2, c2, etypes2,
                                cross_bond_inds_1, cross_bond_inds_2,
                                cross_bond_dists_1, cross_bond_dists_2,
                                triplets_1, triplets_2,
                                sig, ls, r_cut, cutoff_func,
                                nspec, spec_mask, triplet_mask):
    kern = 0

    sig2 = sig * sig
    ls2 = 1 / (2 * ls * ls)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec * nspec

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        bei1n = nspec * bei1

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ei2 = etypes1[ind1]

            bei2 = spec_mask[ei2]

            ttypei = triplet_mask[bc1n + bei1n + bei2]

            tls2 = ls2[ttypei]
            tsig2 = sig2[ttypei]
            tr_cut = r_cut[ttypei]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi1, _ = cutoff_func(tr_cut, ri1, 0)
            fi2, _ = cutoff_func(tr_cut, ri2, 0)
            fi3, _ = cutoff_func(tr_cut, ri3, 0)
            fi = fi1 * fi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(tr_cut, rj1, 0)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    fj2, _ = cutoff_func(tr_cut, rj2, 0)
                    ej2 = etypes2[ind2]

                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    fj3, _ = cutoff_func(tr_cut, rj3, 0)
                    fj = fj1 * fj2 * fj3

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    if (c1 == c2):
                        if (ei1 == ej1) and (ei2 == ej2):
                            C1 = r11 * r11 + r22 * r22 + r33 * r33
                            kern += tsig2 * exp(-C1 * tls2) * fi * fj
                        if (ei1 == ej2) and (ei2 == ej1):
                            C3 = r12 * r12 + r21 * r21 + r33 * r33
                            kern += tsig2 * exp(-C3 * tls2) * fi * fj
                    if (c1 == ej1):
                        if (ei1 == ej2) and (ei2 == c2):
                            C5 = r13 * r13 + r21 * r21 + r32 * r32
                            kern += tsig2 * exp(-C5 * tls2) * fi * fj
                        if (ei1 == c2) and (ei2 == ej2):
                            C2 = r11 * r11 + r23 * r23 + r32 * r32
                            kern += tsig2 * exp(-C2 * tls2) * fi * fj
                    if (c1 == ej2):
                        if (ei1 == ej1) and (ei2 == c2):
                            C6 = r13 * r13 + r22 * r22 + r31 * r31
                            kern += tsig2 * exp(-C6 * tls2) * fi * fj
                        if (ei1 == c2) and (ei2 == ej1):
                            C4 = r12 * r12 + r23 * r23 + r31 * r31
                            kern += tsig2 * exp(-C4 * tls2) * fi * fj

    return kern

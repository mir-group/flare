import numpy as np
from numba import njit
from math import exp
import flare.cutoffs as cf
from flare.kernels.kernels import force_helper, grad_constants, grad_helper, \
    force_energy_helper, three_body_en_helper, three_body_helper_1, \
    three_body_helper_2, three_body_grad_helper_1, three_body_grad_helper_2

"""
Multicomponent kernels that restrict all signal variance hyperparameters to
a single value.
Masking hyperparameters allows you to have different sets of hyperparameters
for different elements, and different groupings of elements.
hyps_mask is a dictionary with the following keys and values:
spec_mask: 118-long integer array descirbing which elements belong to
    like groups for determining which bond hyperparameters to use. For
    instance, [0,1,1,0 ...] assigns H to group 0, He and Li to group 1,
    and Be to group 0.
nspec: Integer, number of different species groups (equal to number of
    unique values in spec_mask).
nbond: Integer, number of different hyperparameter sets to associate with
    different 2-body pairings of atoms in groups defined in spec_mask.
bond_mask: Array of length nspec^2, which describes the hyperparameter sets to
    associate with different pairings of species types. For example, if there
    are atoms of type 0 and 1, then bond_mask defines which hyperparameters
    to use for parings [0-0, 0-1, 1-0, 1-1]: if we wanted hyperparameter set 0 for
    0-0 parings and set 1 for 0-1 and 1-1 pairings, then we would make
    bond_mask [0, 1, 1, 1].
ntriplet = Integer, number of different hyperparameter sets to associate
    with different 3-body pariings of atoms in groups defined in spec_mask.
triplet_mask: Similar to bond mask: Triplet pairings of type 0 and 1 atoms
    would go {0-0-0, 0-0-1, 0-1-0, 0-1-1, 1-0-0, 1-0-1, 1-1-0, 1-1-1},
    and if we wanted hyp. set 0 for triplets with only atoms of type 0
    and hyp. set 1 for all the rest, then the triplet_mask array would
    read [0,1,1,1,1,1,1,1]. The user should make sure that the mask has
    a permutational symmetry.
For selective optimization. one can define 'map', 'train_noise' and 'original'
to identify which element to be optimized. All three have to be defined.
train_noise = Bool (True/False), whether the noise parameter can be optimized
original: np.array. Full set of initial values for hyperparmeters
map: np.array, array to map the hyper parameter back to the full set.
map[i]=j means the i-th element in hyps should be the j-th element in
hyps_mask['original']


For example, the full set of hyper parmeters
may include [ls21, ls22, sig21, sig22, ls3
sg3, noise] but suppose you wanted only the set 21 optimized.
The full set of hyperparameters is defined in 'original'; include all those
you want to leave static, and set initial guesses for those you want to vary.
Have the 'map' list contain the indices of the hyperparameters in 'original'
that correspond to the hyperparameters you want to vary.
Have a hyps list which contain those which you want to vary. Below,
ls21, ls22 etc... represent floating-point variables which correspond
to the initial guesses / static values.
You would then pass in:

hyps = [ls21, sig21]
hyps_mask = { ..., 'train_noise': False, 'map':[0, 2],
                   'original': [ls21, ls22, sig21, sig22, ls3, sg3, noise]}
the hyps argument should only contain the values that need to be optimized.
If you want noise to be trained as well include noise as the
final hyperparameter value in hyps.

"""


# -----------------------------------------------------------------------------
#                        two plus three body kernels
# -----------------------------------------------------------------------------

def from_mask_to_hyps(hyps, hyps_mask: dict = {}):
    """
    :param hyps:
    :param hyps_mask:
    :return:
    """

    n2b = hyps_mask.get('nbond', 0)
    n3b = hyps_mask.get('ntriplet', 0)
    if ('map' in hyps_mask.keys()):
        orig_hyps = hyps_mask['original']
        hm = hyps_mask['map']
        for i, h in enumerate(hyps):
            orig_hyps[hm[i]] = h
    else:
        orig_hyps = hyps

    if (n2b != 0) and (n3b != 0):
        sig2 = orig_hyps[:n2b]
        ls2 = orig_hyps[n2b:n2b * 2]
        sig3 = orig_hyps[n2b * 2:n2b * 2 + n3b]
        ls3 = orig_hyps[n2b * 2 + n3b:n2b * 2 + n3b * 2]
        return n2b, n3b, sig2, ls2, sig3, ls3

    elif (n2b == 0) and (n3b != 0):
        sig = orig_hyps[:n3b]
        ls = orig_hyps[n3b:n3b * 2]
        return 0, n3b, None, None, sig, ls

    elif (n2b != 0) and (n3b == 0):
        sig = orig_hyps[:n2b]
        ls = orig_hyps[n2b:n2b * 2]
        return n2b, 0, sig, ls, None, None

    elif (n2b == 0) and (n3b == 0):
        raise NameError("Hyperparameter mask missing nbond and/or"
                        "ntriplet key")


def from_grad_to_mask(grad, hyps_mask):
    """
    Return gradient which only includes hyperparameters
    which are meant to vary
    :param grad:
    :param hyps_mask:
    :return:
    """
    if 'map' not in hyps_mask.keys():
        return grad

    # if the last element is not sigma_noise
    if (hyps_mask['map'][-1] == len(grad)):
        hm = hyps_mask['map'][:-1]
    else:
        hm = hyps_mask['map']

    newgrad = np.zeros(len(hm))
    for i, mapid in enumerate(hm):
        newgrad[i] = grad[mapid]
    return newgrad


def two_plus_three_body_mc(env1, env2, d1, d2, hyps, cutoffs,
                           cutoff_func=cf.quadratic_cutoff,
                           hyps_mask=None):
    n2b, n3b, sig2, ls2, sig3, ls3 = \
        from_mask_to_hyps(hyps, hyps_mask)

    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]

    two_term = two_body_mc_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                               env2.bond_array_2, env2.ctype, env2.etypes,
                               d1, d2, sig2, ls2, r_cut_2, cutoff_func,
                               hyps_mask['nspec'], hyps_mask['spec_mask'],
                               hyps_mask['bond_mask'])

    three_term = \
        three_body_mc_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                          env2.bond_array_3, env2.ctype, env2.etypes,
                          env1.cross_bond_inds, env2.cross_bond_inds,
                          env1.cross_bond_dists, env2.cross_bond_dists,
                          env1.triplet_counts, env2.triplet_counts,
                          d1, d2, sig3, ls3, r_cut_3, cutoff_func,
                          hyps_mask['nspec'], hyps_mask['spec_mask'],
                          hyps_mask['triplet_mask'])

    return two_term + three_term


def two_plus_three_body_mc_grad(env1, env2, d1, d2, hyps, cutoffs,
                                cutoff_func=cf.quadratic_cutoff,
                                hyps_mask=None):
    n2b, n3b, sig2, ls2, sig3, ls3 = \
        from_mask_to_hyps(hyps, hyps_mask)

    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]

    kern2, grad2 = \
        two_body_mc_grad_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                             env2.bond_array_2, env2.ctype, env2.etypes,
                             d1, d2, sig2, ls2, r_cut_2, cutoff_func,
                             hyps_mask['nspec'], hyps_mask['spec_mask'],
                             hyps_mask['nbond'], hyps_mask['bond_mask'])

    kern3, grad3 = \
        three_body_mc_grad_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                               env2.bond_array_3, env2.ctype, env2.etypes,
                               env1.cross_bond_inds, env2.cross_bond_inds,
                               env1.cross_bond_dists, env2.cross_bond_dists,
                               env1.triplet_counts, env2.triplet_counts,
                               d1, d2, sig3, ls3, r_cut_3,
                               cutoff_func,
                               hyps_mask['nspec'], hyps_mask['spec_mask'],
                               hyps_mask['ntriplet'],
                               hyps_mask['triplet_mask'])

    g = from_grad_to_mask(np.hstack([grad2, grad3]), hyps_mask)

    return kern2 + kern3, g


def two_plus_three_mc_force_en(env1, env2, d1, hyps, cutoffs,
                               cutoff_func=cf.quadratic_cutoff,
                               hyps_mask=None):
    n2b, n3b, sig2, ls2, sig3, ls3 = \
        from_mask_to_hyps(hyps, hyps_mask)
    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]

    two_term = \
        two_body_mc_force_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                 env2.bond_array_2, env2.ctype, env2.etypes,
                                 d1, sig2, ls2, r_cut_2, cutoff_func,
                                 hyps_mask['nspec'], hyps_mask['spec_mask'],
                                 hyps_mask['bond_mask']) / 2

    three_term = \
        three_body_mc_force_en_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                                   env2.bond_array_3, env2.ctype, env2.etypes,
                                   env1.cross_bond_inds, env2.cross_bond_inds,
                                   env1.cross_bond_dists,
                                   env2.cross_bond_dists,
                                   env1.triplet_counts, env2.triplet_counts,
                                   d1, sig3, ls3, r_cut_3, cutoff_func,
                                   hyps_mask['nspec'],
                                   hyps_mask['spec_mask'],
                                   hyps_mask['triplet_mask']) / 3

    return two_term + three_term


def two_plus_three_mc_en(env1, env2, hyps, cutoffs,
                         cutoff_func=cf.quadratic_cutoff, hyps_mask=None):
    n2b, n3b, sig2, ls2, sig3, ls3 = \
        from_mask_to_hyps(hyps, hyps_mask)
    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]

    two_term = two_body_mc_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                  env2.bond_array_2, env2.ctype, env2.etypes,
                                  sig2, ls2, r_cut_2, cutoff_func,
                                  hyps_mask['nspec'],
                                  hyps_mask['spec_mask'],
                                  hyps_mask['bond_mask'])

    three_term = \
        three_body_mc_en_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                             env2.bond_array_3, env2.ctype, env2.etypes,
                             env1.cross_bond_inds, env2.cross_bond_inds,
                             env1.cross_bond_dists, env2.cross_bond_dists,
                             env1.triplet_counts, env2.triplet_counts,
                             sig3, ls3, r_cut_3, cutoff_func,
                             hyps_mask['nspec'], hyps_mask['spec_mask'],
                             hyps_mask['triplet_mask'])

    return two_term + three_term


# -----------------------------------------------------------------------------
#                      three body multicomponent kernel
# -----------------------------------------------------------------------------


def three_body_mc(env1, env2, d1, d2, hyps, cutoffs,
                  cutoff_func=cf.quadratic_cutoff,
                  hyps_mask=None):
    n2b, n3b, sig2, ls2, sig, ls = \
        from_mask_to_hyps(hyps, hyps_mask)

    r_cut = cutoffs[1]

    return three_body_mc_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                             env2.bond_array_3, env2.ctype, env2.etypes,
                             env1.cross_bond_inds, env2.cross_bond_inds,
                             env1.cross_bond_dists, env2.cross_bond_dists,
                             env1.triplet_counts, env2.triplet_counts,
                             d1, d2, sig, ls, r_cut, cutoff_func,
                             hyps_mask['nspec'], hyps_mask['spec_mask'],
                             hyps_mask['triplet_mask'])


def three_body_mc_grad(env1, env2, d1, d2, hyps, cutoffs,
                       cutoff_func=cf.quadratic_cutoff, hyps_mask=None):
    n2b, n3b, sig2, ls2, sig, ls = \
        from_mask_to_hyps(hyps, hyps_mask)
    r_cut = cutoffs[0]

    k, grad = three_body_mc_grad_jit(env1.bond_array_3, env1.ctype,
                                     env1.etypes,
                                     env2.bond_array_3, env2.ctype,
                                     env2.etypes,
                                     env1.cross_bond_inds,
                                     env2.cross_bond_inds,
                                     env1.cross_bond_dists,
                                     env2.cross_bond_dists,
                                     env1.triplet_counts, env2.triplet_counts,
                                     d1, d2, sig, ls, r_cut, cutoff_func,
                                     hyps_mask['nspec'],
                                     hyps_mask['spec_mask'],
                                     hyps_mask['ntriplet'],
                                     hyps_mask['triplet_mask'])
    return k, from_grad_to_mask(grad, hyps_mask)


def three_body_mc_force_en(env1, env2, d1, hyps, cutoffs,
                           cutoff_func=cf.quadratic_cutoff,
                           hyps_mask=None):
    n2b, n3b, sig2, ls2, sig, ls = \
        from_mask_to_hyps(hyps, hyps_mask)
    r_cut = cutoffs[1]

    return three_body_mc_force_en_jit(env1.bond_array_3, env1.ctype,
                                      env1.etypes,
                                      env2.bond_array_3, env2.ctype,
                                      env2.etypes,
                                      env1.cross_bond_inds,
                                      env2.cross_bond_inds,
                                      env1.cross_bond_dists,
                                      env2.cross_bond_dists,
                                      env1.triplet_counts,
                                      env2.triplet_counts,
                                      d1, sig, ls, r_cut,
                                      cutoff_func,
                                      hyps_mask['nspec'],
                                      hyps_mask['spec_mask'],
                                      hyps_mask['triplet_mask']) / 3


def three_body_mc_en(env1, env2, hyps, cutoffs,
                     cutoff_func=cf.quadratic_cutoff,
                     hyps_mask=None):
    n2b, n3b, sig2, ls2, sig, ls = \
        from_mask_to_hyps(hyps, hyps_mask)
    r_cut = cutoffs[1]

    return three_body_mc_en_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                                env2.bond_array_3, env2.ctype, env2.etypes,
                                env1.cross_bond_inds, env2.cross_bond_inds,
                                env1.cross_bond_dists, env2.cross_bond_dists,
                                env1.triplet_counts, env2.triplet_counts,
                                sig, ls, r_cut, cutoff_func,
                                hyps_mask['nspec'], hyps_mask['spec_mask'],
                                hyps_mask['triplet_mask'])


# -----------------------------------------------------------------------------
#                       two body multicomponent kernel
# -----------------------------------------------------------------------------


def two_body_mc(env1, env2, d1, d2, hyps, cutoffs,
                cutoff_func=cf.quadratic_cutoff, hyps_mask=None):
    n2b, n3b, sig, ls, sig3, ls3 = \
        from_mask_to_hyps(hyps, hyps_mask)
    r_cut = cutoffs[0]

    return two_body_mc_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                           env2.bond_array_2, env2.ctype, env2.etypes,
                           d1, d2, sig, ls, r_cut, cutoff_func,
                           hyps_mask['nspec'], hyps_mask['spec_mask'],
                           hyps_mask['bond_mask'])


def two_body_mc_grad(env1, env2, d1, d2, hyps, cutoffs,
                     cutoff_func=cf.quadratic_cutoff,
                     hyps_mask=None):
    n2b, n3b, sig, ls, sig3, ls3 = \
        from_mask_to_hyps(hyps, hyps_mask)
    r_cut = cutoffs[0]

    k, grad = two_body_mc_grad_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                   env2.bond_array_2, env2.ctype, env2.etypes,
                                   d1, d2, sig, ls, r_cut, cutoff_func,
                                   hyps_mask['nspec'], hyps_mask['spec_mask'],
                                   hyps_mask['nbond'], hyps_mask['bond_mask'])
    return k, from_grad_to_mask(grad, hyps_mask)


def two_body_mc_force_en(env1, env2, d1, hyps, cutoffs,
                         cutoff_func=cf.quadratic_cutoff,
                         hyps_mask=None):
    n2b, n3b, sig, ls, sig3, ls3 = \
        from_mask_to_hyps(hyps, hyps_mask)
    r_cut = cutoffs[0]

    return two_body_mc_force_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                    env2.bond_array_2, env2.ctype, env2.etypes,
                                    d1, sig, ls, r_cut, cutoff_func,
                                    hyps_mask['nspec'],
                                    hyps_mask['spec_mask'],
                                    hyps_mask['bond_mask']) / 2


def two_body_mc_en(env1, env2, hyps, cutoffs,
                   cutoff_func=cf.quadratic_cutoff,
                   hyps_mask=None):
    n2b, n3b, sig, ls, sig3, ls3 = \
        from_mask_to_hyps(hyps, hyps_mask)

    r_cut = cutoffs[0]

    return two_body_mc_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                              env2.bond_array_2, env2.ctype, env2.etypes,
                              sig, ls, r_cut, cutoff_func,
                              hyps_mask['nspec'], hyps_mask['spec_mask'],
                              hyps_mask['bond_mask'])


# -----------------------------------------------------------------------------
#                 three body multicomponent kernel (numba)
# -----------------------------------------------------------------------------

@njit
def three_body_mc_jit(bond_array_1, c1, etypes1,
                      bond_array_2, c2, etypes2,
                      cross_bond_inds_1, cross_bond_inds_2,
                      cross_bond_dists_1, cross_bond_dists_2,
                      triplets_1, triplets_2,
                      d1, d2, sig, ls, r_cut, cutoff_func,
                      nspec, spec_mask, triplet_mask):
    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec * nspec

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        bei1n = nspec * bei1

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]
            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
            ei2 = etypes1[ind1]

            bei2 = spec_mask[ei2]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            fi = fi1 * fi2 * fi3
            fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

            ttypei = triplet_mask[bc1n + bei1n + bei2]

            tls1 = ls1[ttypei]
            tls2 = ls2[ttypei]
            tls3 = ls3[ttypei]
            tsig2 = sig2[ttypei]

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + 1 + q]
                    rj2 = bond_array_2[ind2, 0]
                    cj2 = bond_array_2[ind2, d2]
                    fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                    ej2 = etypes2[ind2]

                    rj3 = cross_bond_dists_2[p, p + 1 + q]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)

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


@njit
def three_body_mc_grad_jit(bond_array_1, c1, etypes1,
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
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
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

            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            fi = fi1 * fi2 * fi3
            fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    cj2 = bond_array_2[ind2, d2]
                    ej2 = etypes2[ind2]

                    fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                    fj3, _ = cutoff_func(r_cut, rj3, 0)

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

    kern_grad = np.zeros(2 * ntriplet)
    for i in range(ntriplet):
        kern_grad[i] = sig_derv[i]
    for i in range(ntriplet):
        kern_grad[i + ntriplet] = ls_derv[i]

    return kern, kern_grad


@njit
def three_body_mc_force_en_jit(bond_array_1, c1, etypes1,
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
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        bei1n = nspec * bei1

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]
            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
            ei2 = etypes1[ind1]

            bei2 = spec_mask[ei2]

            ttypei = triplet_mask[bc1n + bei1n + bei2]

            tls1 = ls1[ttypei]
            tls2 = ls2[ttypei]
            tsig2 = sig2[ttypei]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            fi = fi1 * fi2 * fi3
            fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(r_cut, rj1, 0)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    fj2, _ = cutoff_func(r_cut, rj2, 0)
                    ej2 = etypes2[ind2]
                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
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


@njit
def three_body_mc_en_jit(bond_array_1, c1, etypes1,
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
        fi1, _ = cutoff_func(r_cut, ri1, 0)
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        bei1n = nspec * bei1

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            fi2, _ = cutoff_func(r_cut, ri2, 0)
            ei2 = etypes1[ind1]

            bei2 = spec_mask[ei2]

            ttypei = triplet_mask[bc1n + bei1n + bei2]

            tls2 = ls2[ttypei]
            tsig2 = sig2[ttypei]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)
            fi = fi1 * fi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(r_cut, rj1, 0)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    fj2, _ = cutoff_func(r_cut, rj2, 0)
                    ej2 = etypes2[ind2]

                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
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


# -----------------------------------------------------------------------------
#                 two body multicomponent kernel (numba)
# -----------------------------------------------------------------------------


@njit
def two_body_mc_jit(bond_array_1, c1, etypes1,
                    bond_array_2, c2, etypes2,
                    d1, d2, sig, ls, r_cut, cutoff_func,
                    nspec, spec_mask, bond_mask):
    """Multicomponent two-body force/force kernel accelerated with Numba's
    njit decorator.
    Loops over bonds in two environments and adds to the kernel if bonds are
    of the same type.
    """

    kern = 0

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig * sig

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi, fdi = cutoff_func(r_cut, ri, ci)
        e1 = etypes1[m]

        be1 = spec_mask[e1]
        btype = bond_mask[bc1n + be1]

        tls1 = ls1[btype]
        tls2 = ls2[btype]
        tls3 = ls3[btype]
        tsig2 = sig2[btype]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                cj = bond_array_2[n, d2]
                fj, fdj = cutoff_func(r_cut, rj, cj)
                r11 = ri - rj

                A = ci * cj
                B = r11 * ci
                C = r11 * cj
                D = r11 * r11

                kern += force_helper(A, B, C, D, fi, fj, fdi, fdj,
                                     tls1, tls2, tls3, tsig2)

    return kern


@njit
def two_body_mc_grad_jit(bond_array_1, c1, etypes1,
                         bond_array_2, c2, etypes2,
                         d1, d2, sig, ls, r_cut, cutoff_func,
                         nspec, spec_mask, nbond, bond_mask):
    """Multicomponent two-body force/force kernel gradient accelerated with
    Numba's njit decorator."""

    kern = 0
    sig_derv = np.zeros(nbond)
    ls_derv = np.zeros(nbond)

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    ls4 = 1 / (ls * ls * ls)
    ls5 = ls * ls
    ls6 = ls2 * ls4

    sig2 = sig * sig
    sig3 = 2 * sig

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi, fdi = cutoff_func(r_cut, ri, ci)
        e1 = etypes1[m]

        be1 = spec_mask[e1]
        btype = bond_mask[bc1n + be1]

        tls1 = ls1[btype]
        tls2 = ls2[btype]
        tls3 = ls3[btype]
        tls4 = ls4[btype]
        tls5 = ls5[btype]
        tls6 = ls6[btype]
        tsig2 = sig2[btype]
        tsig3 = sig3[btype]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                cj = bond_array_2[n, d2]
                fj, fdj = cutoff_func(r_cut, rj, cj)

                r11 = ri - rj

                A = ci * cj
                B = r11 * ci
                C = r11 * cj
                D = r11 * r11

                kern_term, sig_term, ls_term = \
                    grad_helper(A, B, C, D, fi, fj, fdi, fdj,
                                tls1, tls2, tls3,
                                tls4, tls5, tls6,
                                tsig2, tsig3)

                kern += kern_term
                sig_derv[btype] += sig_term
                ls_derv[btype] += ls_term

    kern_grad = np.zeros(2 * nbond)
    for i in range(nbond):
        kern_grad[i] = sig_derv[i]
    for i in range(nbond):
        kern_grad[i + nbond] = ls_derv[i]

    return kern, kern_grad


@njit
def two_body_mc_force_en_jit(bond_array_1, c1, etypes1,
                             bond_array_2, c2, etypes2,
                             d1, sig, ls, r_cut, cutoff_func,
                             nspec, spec_mask, bond_mask):
    """Multicomponent two-body force/energy kernel accelerated with
    Numba's njit decorator."""

    kern = 0

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    sig2 = sig * sig

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi, fdi = cutoff_func(r_cut, ri, ci)
        e1 = etypes1[m]

        be1 = spec_mask[e1]
        btype = bond_mask[bc1n + be1]

        tls1 = ls1[btype]
        tls2 = ls2[btype]
        tsig2 = sig2[btype]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                fj, _ = cutoff_func(r_cut, rj, 0)

                r11 = ri - rj
                B = r11 * ci
                D = r11 * r11
                kern += force_energy_helper(B, D, fi, fj, fdi,
                                            tls1, tls2,
                                            tsig2)

    return kern


@njit
def two_body_mc_en_jit(bond_array_1, c1, etypes1,
                       bond_array_2, c2, etypes2,
                       sig, ls, r_cut, cutoff_func,
                       nspec, spec_mask, bond_mask):
    """Multicomponent two-body energy/energy kernel accelerated with
    Numba's njit decorator."""

    kern = 0

    ls1 = 1 / (2 * ls * ls)
    sig2 = sig * sig

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        fi, _ = cutoff_func(r_cut, ri, 0)
        e1 = etypes1[m]

        be1 = spec_mask[e1]
        btype = bond_mask[bc1n + be1]

        tls1 = ls1[btype]
        tsig2 = sig2[btype]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                fj, _ = cutoff_func(r_cut, rj, 0)
                r11 = ri - rj
                kern += fi * fj * tsig2 * exp(-r11 * r11 * tls1)

    return kern


_str_to_kernel = {'two_body_mc': two_body_mc,
                  'two_body_mc_grad': two_body_mc_grad,
                  'two_body_mc_en': two_body_mc_en,
                  'two_body_mc_force_en': two_body_mc_force_en,
                  'three_body_mc': three_body_mc,
                  'three_body_mc_grad': three_body_mc_grad,
                  'three_body_mc_en': three_body_mc_en,
                  'three_body_mc_force_en': three_body_mc_force_en,
                  'two_plus_three_body_mc': two_plus_three_body_mc,
                  'two_plus_three_body_mc_grad': two_plus_three_body_mc_grad,
                  'two_plus_three_body_mc_en': two_plus_three_mc_en,
                  'two_plus_three_body_mc_force_en': two_plus_three_mc_force_en,
                  'two_plus_three_mc': two_plus_three_body_mc,
                  'two_plus_three_mc_grad': two_plus_three_body_mc_grad,
                  'two_plus_three_mc_en': two_plus_three_mc_en,
                  'two_plus_three_mc_force_en': two_plus_three_mc_force_en,
                  'two_body_mc_sh': two_body_mc,
                  'two_body_mc_sh_grad': two_body_mc_grad,
                  'two_body_mc_sh_en': two_body_mc_en,
                  'two_body_mc_sh_force_en': two_body_mc_force_en,
                  'three_body_mc_sh': three_body_mc,
                  'three_body_mc_sh_grad': three_body_mc_grad,
                  'three_body_mc_sh_en': three_body_mc_en,
                  'three_body_mc_sh_force_en': three_body_mc_force_en,
                  'two_plus_three_body_mc_sh': two_plus_three_body_mc,
                  'two_plus_three_body_mc_sh_grad': two_plus_three_body_mc_grad,
                  'two_plus_three_body_mc_sh_en': two_plus_three_mc_en,
                  'two_plus_three_body_mc_sh_force_en': two_plus_three_mc_force_en,
                  'two_plus_three_mc_sh': two_plus_three_body_mc,
                  'two_plus_three_mc_sh_grad': two_plus_three_body_mc_grad,
                  'two_plus_three_mc_sh_en': two_plus_three_mc_en,
                  'two_plus_three_mc_sh_force_en': two_plus_three_mc_force_en
                  }


def str_to_mc_kernel(string: str, include_grad: bool = False):
    if string not in _str_to_kernel.keys():
        raise ValueError("Kernel {} not found in list of available "
                         "kernels{}:".format(string, _str_to_kernel.keys()))

    if not include_grad:
        return _str_to_kernel[string]
    else:
        if 'two' in string and 'three' in string:
            return _str_to_kernel[string], two_plus_three_body_mc_grad
        elif 'two' in string and 'three' not in string:
            return _str_to_kernel[string], two_body_mc_grad
        elif 'two' not in string and 'three' in string:
            return _str_to_kernel[string], three_body_mc_grad
        else:
            raise ValueError("Gradient callable for {} not found".format(
                string))

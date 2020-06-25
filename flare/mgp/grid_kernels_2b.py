import numpy as np
from numba import njit
from math import exp, floor
from typing import Callable

from flare.kernels.cutoffs import quadratic_cutoff


def self_kernel_sephyps(map_force, grids, c2, etypes2, 
                      cutoff_2b, cutoff_3b, cutoff_mb,
                      nspec, spec_mask,
                      nbond, bond_mask,
                      ntriplet, triplet_mask,
                      ncut3b, cut3b_mask,
                      nmb, mb_mask,
                      sig2, ls2, sig3, ls3, sigm, lsm,
                      cutoff_func=quadratic_cutoff):
    '''
    Args:
        data: a single env of a list of envs
    '''

    if map_force:
        raise NotImplementedError

    bc1 = spec_mask[c2]
    bc2 = spec_mask[etypes2[0]]
    btype = bond_mask[nspec * bc1 + bc2]
    ls = ls2[btype]
    sig = sig2[btype]
    cutoffs = [cutoff_2b[btype]]
    hyps = [sig, ls]

    return self_kernel(map_force, grids, c2, etypes2, 
                       hyps, cutoffs, cutoff_func)


def self_kernel(map_force, grids, c2, etypes2, hyps, cutoffs, 
              cutoff_func: Callable = quadratic_cutoff):

    if map_force:
        raise NotImplementedError

    # pre-compute constants
    r_cut = cutoffs[0]
    sig = hyps[0]
    ls = hyps[1]
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    
    fj, _ = cutoff_func(r_cut, grids, 0)

    kern = (sig2 / 4) * fj ** 2 # (n_grids,)

    return kern
 

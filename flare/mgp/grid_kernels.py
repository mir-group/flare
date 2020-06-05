from numpy import array
from numba import njit
from math import exp, floor
from typing import Callable

from flare.env import AtomicEnvironment
from flare.kernels.cutoffs import quadratic_cutoff


def get_3b_args(env1):
    return [env1.bond_array_3, 
            env1.ctype, env1.etypes,
            env1.cross_bond_inds, 
            env1.cross_bond_dists, 
            env1.triplet_counts]


def grid_kernel_3b(kern_type,
                   env1: AtomicEnvironment, grids, fj, 
                   c2, etypes2, perm_list,
                   hyps: 'ndarray', r_cut: float,
                   cutoff_func: Callable = quadratic_cutoff):

    sig = hyps[0]
    ls = hyps[1]

    bond_array_1, c1, etypes1, cross_bond_inds_1, cross_bond_dists_1, triplets_1\
         = get_3b_args(env1)

    kern = np.zeros((3, grids.shape[0]), dtype=np.float64)

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)


    # -------- 1. collect all the triplets in this training env --------
    triplet_coord_list = get_triplets_for_kern(bond_array_1, c1, etypes1,
        cross_bond_inds_1, cross_bond_dists_1, triplets_1,
        c2, etypes2, perm_list)

    if len(triplet_coord_list) == 0: # no triplets
        return kern

    triplet_coord_list = np.array(triplet_coord_list)
    triplet_list = triplet_coord_list[:, :3] # (n_triplets, 3)
    coord_list = triplet_coord_list[:, 3:] # ((n_triplets, 9)


    # ---------------- 2. calculate cutoff of the triplets ----------------
    fi, fdi = triplet_cutoff_grad(triplet_list, r_cut, coord_list) # (n_triplets, 1)
    fifj = fi @ fj.T # (n_triplets, n_grids)


    # -------- 3. calculate distance difference & exponential part --------
    D = 0
    for r in range(3):
        rj, ri = np.meshgrid(grids[:, r], triplet_list[:, r])
        rij = ri - rj
        D += rij * rij # (n_triplets, n_grids)
    kern_exp = sig2 * np.exp(- D * ls1)


    # ---------------- 4. calculate the derivative part ----------------
    if kern_type == 'energy_energy':
        kern_exp = np.sum(kern_exp * fifj, axis=0) / 9 # (n_grids,)
        for d in range(3):
            kern[d, :] = kern_exp    

    elif kern_type == 'energy_force':
        for d in range(3):
            B = 0
            fdij = fdi[d] @ fj.T
            for r in range(3):
                rj, ri = np.meshgrid(grids[:, r], triplet_list[:, r])
                rij = ri - rj
                # column-wise multiplication
                # coord_list[:, [r]].shape = (n_triplets, 1)
                B += rij * coord_list[:, [3*d+r]] # (n_triplets, n_grids)
   
            kern[d,:] = - np.sum(kern_exp * (B * ls2 * fifj + fdij), axis=0) / 3 # (n_grids,)

    return kern




def grid_kernel_3b_sephyps(kern_type,
                  env1, grids, fj, c2, etypes2, perm_list,
                  cutoff_2b, cutoff_3b, nspec, spec_mask,
                  nbond, bond_mask, ntriplet, triplet_mask,
                  ncut3b, cut3b_mask,
                  sig2, ls2, sig3, ls3,
                  cutoff_func=quadratic_cutoff):

    bc1 = spec_mask[c2]
    bc2 = spec_mask[etypes2[0]]
    bc3 = spec_mask[etypes2[1]]
    ttype = triplet_mask[nspec * nspec * bc1 + nspec*bc2 + bc3]
    ls = ls3[ttype]
    sig = sig3[ttype]
    r_cut = cutoff_3b

    args = get_3b_args(env1)

    hyps = [sig, ls]
    return grid_kernel_3b(kern_type,
                   env1, grids, fj, 
                   c2, etypes2, perm_list,
                   hyps, r_cut,
                   cutoff_func)


@njit
def triplet_cutoff(triplets, r_cut, cutoff_func=quadratic_cutoff):
    f0, _ = cutoff_func(r_cut, triplets, 0) # (n_grid, 3)
    fj = f0[:, 0] * f0[:, 1] * f0[:, 2] # (n_grid,)
    return np.expand_dims(fj, axis=1) # (n_grid, 1)

@njit
def triplet_cutoff_grad(triplets, r_cut, coords, cutoff_func=quadratic_cutoff):

    dfj_list = []
    for d in range(3):
        s = 3 * d
        e = 3 * (d + 1)
        f0, df0 = cutoff_func(r_cut, triplets, coords[:, s:e]) # (n_grid, 3)
        dfj = df0[:, 0] *  f0[:, 1] *  f0[:, 2] + \
               f0[:, 0] * df0[:, 1] *  f0[:, 2] + \
               f0[:, 0] *  f0[:, 1] * df0[:, 2]
        dfj = np.expand_dims(dfj, axis=1)
        dfj_list.append(dfj_list)

    fj = f0[:, 0] * f0[:, 1] * f0[:, 2] # (n_grid,)
    fj = np.expand_dims(fj, axis=1)

    return fj, dfj

@njit
def get_triplets_for_kern(bond_array_1, c1, etypes1,
                          cross_bond_inds_1, cross_bond_dists_1,
                          triplets_1,
                          c2, etypes2, perm_list):

    #triplet_list = np.empty((0, 6), dtype=np.float64)
    triplet_list = []

    ej1 = etypes2[0]
    ej2 = etypes2[1]

    all_spec = [c2, ej1, ej2]
    if c1 in all_spec:
        c1_ind = all_spec.index(c1)
        ind_list = [0, 1, 2]
        ind_list.remove(c1_ind)
        all_spec.remove(c1)
    
        for m in range(bond_array_1.shape[0]):
            two_inds = ind_list.copy()
    
            ri1 = bond_array_1[m, 0]
            ci1 = bond_array_1[m, 1:]
            ei1 = etypes1[m]
    
            two_spec = [all_spec[0], all_spec[1]]
            if (ei1 in two_spec):
    
                ei1_ind = ind_list[0] if ei1 == two_spec[0] else ind_list[1]
                two_spec.remove(ei1)
                two_inds.remove(ei1_ind)
                one_spec = two_spec[0]
                ei2_ind = two_inds[0]
    
                for n in range(triplets_1[m]):
                    ind1 = cross_bond_inds_1[m, m + n + 1]
                    ei2 = etypes1[ind1]
                    if (ei2 == one_spec):
    
                        order = [c1_ind, ei1_ind, ei2_ind]
                        ri2 = bond_array_1[ind1, 0]
                        ci2 = bond_array_1[ind1, 1:]
    
                        ri3 = cross_bond_dists_1[m, m + n + 1]
                        ci3 = np.zeros(3)
    
                        # align this triplet to the same species order as r1, r2, r12
                        tri = np.take(np.array([ri1, ri2, ri3]), order)
                        crd = np.take(np.array([ci1, ci2, ci3]), order, axis=0)
    
                        # append permutations
                        for perm in perm_list:
                            tricrd = np.take(tri, perm)
                            for d in range(3):
                                tricrd = np.hstack((tricrd, np.take(crd[:, d], perm)))
                            triplet_list.append(tricrd)

    return triplet_list

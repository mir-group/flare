import numpy as np
from numba import njit
from math import exp, floor
from typing import Callable

from flare.kernels.cutoffs import quadratic_cutoff

from time import time

def grid_kernel_sephyps(kern_type,
                  data, grids, fj, fdj,
                  c2, etypes2, perm_list,
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

    bc1 = spec_mask[c2]
    bc2 = spec_mask[etypes2[0]]
    bc3 = spec_mask[etypes2[1]]
    ttype = triplet_mask[nspec * nspec * bc1 + nspec*bc2 + bc3]
    ls = ls3[ttype]
    sig = sig3[ttype]
    cutoffs = [cutoff_2b, cutoff_3b]

    hyps = [sig, ls]
    return grid_kernel(kern_type,
                   data, grids, fj, fdj, 
                   c2, etypes2, perm_list,
                   hyps, cutoffs, cutoff_func)


def grid_kernel(kern_type, 
                struc, grids, fj, fdj, 
                c2, etypes2, perm_list,
                hyps: 'ndarray', cutoffs,
                cutoff_func: Callable = quadratic_cutoff):

    r_cut = cutoffs[1]

    if not isinstance(struc, list):
        struc = [struc]

    kern = 0
    for env in struc:
        kern += grid_kernel_env(kern_type, 
                    env, grids, fj, fdj, 
                    c2, etypes2, perm_list,
                    hyps, r_cut, cutoff_func)

    return kern


def grid_kernel_env(kern_type, 
                env1, grids, fj, fdj, 
                c2, etypes2, perm_list,
                hyps: 'ndarray', r_cut: float,
                cutoff_func: Callable = quadratic_cutoff):

    # pre-compute constants that appear in the inner loop
    sig = hyps[0]
    ls = hyps[1]
    derivative = derv_dict[kern_type] 


    # collect all the triplets in this training env
    triplet_coord_list = get_triplets_for_kern(env1.bond_array_3, env1.ctype, env1.etypes,
        env1.cross_bond_inds, env1.cross_bond_dists, env1.triplet_counts,
        c2, etypes2, perm_list)

    if len(triplet_coord_list) == 0: # no triplets
        if derivative:
            return np.zeros((3, grids.shape[0]), dtype=np.float64)
        else:
            return np.zeros(grids.shape[0], dtype=np.float64)

    triplet_coord_list = np.array(triplet_coord_list)
    triplet_list = triplet_coord_list[:, :3] # (n_triplets, 3)
    coord_list = triplet_coord_list[:, 3:] # ((n_triplets, 9)

    # calculate distance difference & exponential part
    ls1 = 1 / (2 * ls * ls)
    D = 0
    rij_list = []
    for r in range(3):
        rj, ri = np.meshgrid(grids[:, r], triplet_list[:, r])
        rij = ri - rj
        D += rij * rij # (n_triplets, n_grids)
        rij_list.append(rij)
    kern_exp = (sig * sig) * np.exp(- D * ls1)

    # calculate cutoff of the triplets
    fi, fdi = triplet_cutoff(triplet_list, r_cut, coord_list, derivative, 
        cutoff_func) # (n_triplets, 1)

    # calculate the derivative part
    kern_func = kern_dict[kern_type]
    kern = kern_func(kern_exp, fi, fj, fdi, fdj, 
             rij_list, coord_list, ls)

    return kern


@njit
def en_en(kern_exp, fi, fj, *args):
    '''energy map + energy block'''
    fifj = fi @ fj.T # (n_triplets, n_grids)
    kern = np.sum(kern_exp * fifj, axis=0) / 9 # (n_grids,)
    return kern


#@njit
def en_force(kern_exp, fi, fj, fdi, fdj, 
             rij_list, coord_list, ls):
    '''energy map + force block'''
    fifj = fi @ fj.T # (n_triplets, n_grids)
    ls2 = 1 / (ls * ls)
    n_trplt, n_grids = kern_exp.shape
    kern = np.zeros((3, n_grids), dtype=np.float64)
    for d in range(3):
        B = 0
        fdij = fdi[:, [d]] @ fj.T
        for r in range(3):
            # one day when numba supports np.meshgrid, we can replace the block below
            rij = rij_list[r]
            # column-wise multiplication
            # coord_list[:, [r]].shape = (n_triplets, 1)
            B += rij * coord_list[:, [3*d+r]] # (n_triplets, n_grids)
   
        kern[d, :] = - np.sum(kern_exp * (B * ls2 * fifj + fdij), axis=0) / 3 # (n_grids,)
    return kern


@njit
def force_en(kern_exp, fi, fj, fdi, fdj, 
             grids, triplet_list, coord_list, ls):
    '''force map + energy block'''
    fifj = fi @ fj.T # (n_triplets, n_grids)
    fdji = fi @ fdj.T
    # only r = 0 is non zero, since the grid coords are all (1, 0, 0)
    rj, ri = np.meshgrid(grids[:, 0], triplet_list[:, 0])
    rji = rj - ri
    B = rji # (n_triplets, n_grids)
    kern = - np.sum(kern_exp * (B * ls2 * fifj + fdji), axis=0) / 3 # (n_grids,)
    return kern


@njit
def force_force(kern_exp, fi, fj, fdi, fdj, 
             grids, triplet_list, coord_list, ls):
    '''force map + force block'''
    kern = np.zeros((3, grids.shape[0]), dtype=np.float64)

    fifj = fi @ fj.T # (n_triplets, n_grids)
    fdji = fi @ fdj.T
    # only r = 0 is non zero, since the grid coords are all (1, 0, 0)
    rj, ri = np.meshgrid(grids[:, 0], triplet_list[:, 0])
    rji = rj - ri
    B = rji # (n_triplets, n_grids)
    kern = - np.sum(kern_exp * (B * ls2 * fifj + fdji), axis=0) / 3 # (n_grids,)

    for d in range(3):
        pass

    return kern



def triplet_cutoff(triplets, r_cut, coords, derivative=False, cutoff_func=quadratic_cutoff):

    dfj_list = np.zeros((len(triplets), 3), dtype=np.float64) 

    if derivative:
        for d in range(3):
            s = 3 * d
            e = 3 * (d + 1)
            f0, df0 = cutoff_func(r_cut, triplets, coords[:, s:e]) 
            dfj = df0[:, 0] *  f0[:, 1] *  f0[:, 2] + \
                   f0[:, 0] * df0[:, 1] *  f0[:, 2] + \
                   f0[:, 0] *  f0[:, 1] * df0[:, 2]
#            dfj = np.expand_dims(dfj, axis=1)
            dfj_list[:, d] = dfj 
    else:
        f0, _ = cutoff_func(r_cut, triplets, 0) # (n_grid, 3)

    fj = f0[:, 0] * f0[:, 1] * f0[:, 2] # (n_grid,)
    fj = np.expand_dims(fj, axis=1)

    return fj, dfj_list


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
                        crd1 = np.take(np.array([ci1[0], ci2[0], ci3[0]]), order)
                        crd2 = np.take(np.array([ci1[1], ci2[1], ci3[1]]), order)
                        crd3 = np.take(np.array([ci1[2], ci2[2], ci3[2]]), order)
    
                        # append permutations
                        for perm in perm_list:
                            tricrd = np.take(tri, perm)
                            crd1_p = np.take(crd1, perm)
                            crd2_p = np.take(crd2, perm)
                            crd3_p = np.take(crd3, perm)
                            tricrd = np.hstack((tricrd, crd1_p, crd2_p, crd3_p))
                            triplet_list.append(tricrd)

    return triplet_list



kern_dict = {'energy_energy': en_en,
             'energy_force': en_force,
             'force_energy': force_en,
             'force_force': force_force}

derv_dict = {'energy_energy': False,
             'energy_force': True,
             'force_energy': False,
             'force_force': True}


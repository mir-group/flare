import numpy as np
from numba import njit
from math import exp, floor
from typing import Callable

from flare.kernels.cutoffs import quadratic_cutoff


def grid_kernel_sephyps(kern_type,
                  data, grids, fj, fdj,
                  c2, etypes2, 
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
    btype = bond_mask[nspec * bc1 + bc2]
    ls = ls2[btype]
    sig = sig2[btype]
    cutoffs = [cutoff_2b[btype]]
    hyps = [sig, ls]

    return grid_kernel(kern_type,
                   data, grids, fj, fdj,
                   c2, etypes2, 
                   hyps, cutoffs, cutoff_func)


def grid_kernel(kern_type, struc, grids, fj, fdj, c2, etypes2, 
                hyps: 'ndarray', cutoffs,
                cutoff_func: Callable = quadratic_cutoff):

    r_cut = cutoffs[0]

    if not isinstance(struc, list):
        struc = [struc]

    kern = 0
    for env in struc:
        kern += grid_kernel_env(kern_type, env, grids, fj, fdj,
                    c2, etypes2, hyps, r_cut, cutoff_func)

    return kern


def grid_kernel_env(kern_type, env1, grids, fj, fdj, c2, etypes2, 
                hyps: 'ndarray', r_cut: float,
                cutoff_func: Callable = quadratic_cutoff):

    # pre-compute constants that appear in the inner loop
    sig = hyps[0]
    ls = hyps[1]
    derivative = derv_dict[kern_type]

    # collect all bonds
    bond_coord_list = get_bonds_for_kern(env1.bond_array_2, env1.ctype, 
        env1.etypes, c2, etypes2)

    if len(bond_coord_list) == 0:
        if derivative:
            return np.zeros((3, grids.shape[0]), dtype=np.float64)
        else:
            return np.zeros(grids.shape[0], dtype=np.float64) 

    bond_coord_list = np.array(bond_coord_list)
    bond_list = bond_coord_list[:, :1]
    coord_list = bond_coord_list[:, 1:]
    del bond_coord_list

    # calculate distance difference & exponential part
    ls1 = 1 / (2 * ls * ls)
    rj, ri = np.meshgrid(grids, bond_list)
    rij = ri - rj
    D = rij * rij # (n_bonds, n_grids)
    rij_list = [rij]

    kern_exp = (sig * sig) * np.exp(- D * ls1)
    del D

    # calculate cutoff of the triplets
    fi, fdi = cutoff_func(r_cut, bond_list, coord_list)
    del bond_list

    # calculate the derivative part
    kern_func = kern_dict[kern_type]
    kern = kern_func(kern_exp, fi, fj, fdi, fdj,
             rij_list, coord_list, ls)

    return kern


def en_en(kern_exp, fi, fj, *args):
    '''energy map + energy block'''
    fifj = fi @ fj.T # (n_triplets, n_grids)
    kern = np.sum(kern_exp * fifj, axis=0) / 4 # (n_grids,)
    return kern


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
        # column-wise multiplication
        # coord_list[:, [r]].shape = (n_triplets, 1)
        B += rij_list[0] * coord_list[:, [d]] # (n_triplets, n_grids)

        kern[d, :] = - np.sum(kern_exp * (B * ls2 * fifj + fdij), axis=0) / 2 # (n_grids,)
    return kern


def force_en(kern_exp, fi, fj, fdi, fdj, 
             rij_list, coord_list, ls):
    '''force map + energy block'''
    ls2 = 1 / (ls * ls)
    fifj = fi @ fj.T # (n_triplets, n_grids)
    fdji = fi @ fdj.T
    # only r = 0 is non zero, since the grid coords are all (1, 0, 0)
    B = rij_list[0] # (n_triplets, n_grids)
    kern = np.sum(kern_exp * (B * ls2 * fifj - fdji), axis=0) / 2 # (n_grids,)
    return kern


def force_force(kern_exp, fi, fj, fdi, fdj, 
                rij_list, coord_list, ls):
    '''force map + force block'''
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2 
    
    n_trplt, n_grids = kern_exp.shape
    kern = np.zeros((3, n_grids), dtype=np.float64)

    fifj = fi @ fj.T # (n_triplets, n_grids)
    fdji = (fi * ls2) @ fdj.T

    B = rij_list[0] * ls3 # B and C both have opposite signs with that in the three_body_helper_1
    
    for d in range(3):
        fdij = (fdi[:, [d]] * ls2) @ fj.T
        I = fdi[:, [d]] @ fdj.T
        J = rij_list[0] * fdij # (n_triplets, n_grids)

        A = np.repeat(ls2 * coord_list[:, [d]], n_grids, axis=1)
        C = 0
        # column-wise multiplication
        # coord_list[:, [r]].shape = (n_triplets, 1)
        C += rij_list[0] * coord_list[:, [d]] # (n_triplets, n_grids)

        IJKL = I - J + C * fdji + (A - B * C) * fifj
        kern[d, :] = np.sum(kern_exp * IJKL, axis=0)

    return kern


def bond_cutoff(bonds, r_cut, coords, derivative=False, cutoff_func=quadratic_cutoff):
    fj, dfj = cutoff_func(r_cut, bonds, coords)
    return fj, dfj


@njit
def get_bonds_for_kern(bond_array_1, c1, etypes1, c2, etypes2):

    e2 = etypes2[0]

    bond_list = []
    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, 1:]
        e1 = etypes1[m]

        if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
            bond_list.append([ri, ci[0], ci[1], ci[2]])

    return bond_list



def self_kernel_sephyps(map_force, grids, fj, fdj, c2, etypes2, 
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
    btype = bond_mask[nspec * bc1 + bc2]
    ls = ls2[btype]
    sig = sig2[btype]
    cutoffs = [cutoff_2b[btype]]
    hyps = [sig, ls]

    return self_kernel(map_force, grids, fj, fdj, c2, etypes2, 
                       hyps, cutoffs, cutoff_func)


def self_kernel(map_force, grids, fj, fdj, c2, etypes2, hyps, cutoffs, 
              cutoff_func: Callable = quadratic_cutoff):

    # pre-compute constants
    r_cut = cutoffs[0]
    sig = hyps[0]
    ls = hyps[1]
    sig2 = sig * sig
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
   
    if map_force:
        I = fdj ** 2
        L = ls2 * fj ** 2
        kern = sig2 * (I + L)
        return np.sum(kern, axis=1)
    else:
        kern = (sig2 / 4) * fj ** 2 # (n_grids,)
        return np.sum(kern, axis=1)
 

kern_dict = {'energy_energy': en_en,
             'energy_force': en_force,
             'force_energy': force_en,
             'force_force': force_force}

derv_dict = {'energy_energy': False,
             'energy_force': True,
             'force_energy': False,
             'force_force': True}


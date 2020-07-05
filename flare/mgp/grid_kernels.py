import numpy as np
from numba import njit
from math import exp, floor
from typing import Callable

from flare.kernels.cutoffs import quadratic_cutoff

from time import time


def grid_kernel(struc, *args):

    if not isinstance(struc, list):
        struc = [struc]

    kern = 0
    for env in struc:
        kern += grid_kernel_env(env, *args)

    return kern


def grid_kernel_env(
    env1,
    bodies,
    kern_type,
    get_bonds_func,
    bonds_cutoff_func,
    c2,
    etypes2,
    hyps,
    r_cut,
    grids,
    fj,
    fdj,
    cutoff_func: Callable = quadratic_cutoff,
):

    # pre-compute constants that appear in the inner loop
    sig = hyps[0]
    ls = hyps[1]
    derivative = derv_dict[kern_type]
    grid_dim = grids.shape[1]

    # collect all the triplets in this training env
    bond_coord_list = get_bonds_func(env1, c2, etypes2)

    if len(bond_coord_list) == 0:  # no triplets
        if derivative:
            return np.zeros((3, grids.shape[0]), dtype=np.float64)
        else:
            return np.zeros(grids.shape[0], dtype=np.float64)

    bond_coord_list = np.array(bond_coord_list)
    bond_list = bond_coord_list[:, :grid_dim]
    coord_list = bond_coord_list[:, grid_dim:]
    del bond_coord_list

    # calculate distance difference & exponential part
    ls1 = 1 / (2 * ls * ls)
    D = 0
    rij_list = []
    for r in range(grid_dim):
        rj, ri = np.meshgrid(grids[:, r], bond_list[:, r])
        rij = ri - rj
        D += rij * rij  # (n_triplets, n_grids)
        rij_list.append(rij)

    kern_exp = (sig * sig) * np.exp(-D * ls1)
    del D

    # calculate cutoff of the triplets
    fi, fdi = bonds_cutoff_func(
        bond_list, r_cut, coord_list, derivative, cutoff_func
    )  # (n_triplets, 1)
    del bond_list

    # calculate the derivative part
    kern_func = kern_dict[kern_type]
    kern = kern_func(
        bodies, grid_dim, kern_exp, fi, fj, fdi, fdj, rij_list, coord_list, ls
    )

    return kern


def en_en(bodies, grid_dim, kern_exp, fi, fj, *args):
    """energy map + energy block"""
    fifj = fi @ fj.T  # (n_triplets, n_grids)
    kern = np.sum(kern_exp * fifj, axis=0) / bodies ** 2  # (n_grids,)
    return kern


def en_force(bodies, grid_dim, kern_exp, fi, fj, fdi, fdj, rij_list, coord_list, ls):
    """energy map + force block"""
    fifj = fi @ fj.T  # (n_triplets, n_grids)
    ls2 = 1 / (ls * ls)
    n_trplt, n_grids = kern_exp.shape
    kern = np.zeros((3, n_grids), dtype=np.float64)
    for d in range(3):
        B = 0
        fdij = fdi[:, [d]] @ fj.T
        for r in range(grid_dim):
            rij = rij_list[r]
            # column-wise multiplication
            # coord_list[:, [r]].shape = (n_triplets, 1)
            B += rij * coord_list[:, [3 * r + d]]  # (n_triplets, n_grids)

        kern[d, :] = (
            -np.sum(kern_exp * (B * ls2 * fifj + fdij), axis=0) / bodies
        )  # (n_grids,)
    return kern


def self_kernel(
    bodies,
    get_permutations,
    c2,
    etypes2,
    hyps,
    r_cut,
    grids,
    fj,
    fdj,
    cutoff_func: Callable = quadratic_cutoff,
):

    kern = 0

    # pre-compute constants
    sig = hyps[0]
    ls = hyps[1]
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2

    perm_list = get_permutations(c2, etypes2, c2, etypes2)

    for perm in perm_list:
        perm_grids = np.take(grids, perm, axis=1)
        rij = grids - perm_grids
        D = np.sum(rij * rij, axis=1)  # (n_grids, ) adding up three bonds
        kern_exp = np.exp(-D * ls1) * sig2
        fjfj = fj ** 2
        kern += kern_exp * np.sum(fjfj, axis=1) / bodies ** 2  # (n_grids,)

    return kern


kern_dict = {
    "energy_energy": en_en,
    "energy_force": en_force,
}

derv_dict = {
    "energy_energy": False,
    "energy_force": True,
}

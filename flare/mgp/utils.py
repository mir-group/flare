import warnings
import numpy as np

from numpy import array
from numba import njit
from math import exp, floor
from typing import Callable

from flare.env import AtomicEnvironment
from flare.kernels.cutoffs import quadratic_cutoff
from flare.kernels.utils import str_to_kernel_set
from flare.parameters import Parameters

from flare.mgp.grid_kernels_3b import grid_kernel, grid_kernel_sephyps


def str_to_mapped_kernel(name: str, component: str = "mc",
                         hyps_mask: dict = None):
    """
    Return kernels and kernel gradient function based on a string.
    If it contains 'sc', it will use the kernel in sc module;
    otherwise, it uses the kernel in mc_simple;
    if sc is not included and multihyps is True,
    it will use the kernel in mc_sephyps module.
    Otherwise, it will use the kernel in the sc module.

    Args:

    name (str): name for kernels. example: "2+3mc"
    multihyps (bool, optional): True for using multiple hyperparameter groups

    :return: mapped kernel function, kernel gradient, energy kernel,
             energy_and_force kernel

    """

    multihyps = True
    if hyps_mask is None:
        multihyps = False
    elif hyps_mask['nspecie'] == 1:
        multihyps = False

    # b2 = Two body in use, b3 = Three body in use
    b2 = False
    many = False
    b3 = False
    for s in ['3', 'three']:
        if s in name.lower() or s == name.lower():
            b3 = True

    if b3:
         if multihyps:
             return grid_kernel_sephyps, None, None, None
         else:
             return grid_kernel, None, None, None
    else:
        warnings.Warn(NotImplemented("mapped kernel for two-body and manybody kernels "
                                  "are not implemented"))
        return None

def get_kernel_term(kernel_name, component, hyps_mask, hyps, grid_kernel=False):
    """
    Args
        term (str): 'twobody' or 'threebody'
    """
    if grid_kernel:
        stks = str_to_mapped_kernel
        kernel_name_list = kernel_name
    else:
        stks = str_to_kernel_set
        kernel_name_list = [kernel_name] 

    kernel, _, ek, efk = stks(kernel_name_list, component, hyps_mask)

    # hyps_mask is modified here
    hyps, cutoffs, hyps_mask = Parameters.get_component_mask(hyps_mask, kernel_name, hyps=hyps)

    return (kernel, ek, efk, cutoffs, hyps, hyps_mask)



@njit
def get_bonds(ctype, etypes, bond_array):
    exist_species = []
    bond_lengths = []
    bond_dirs = []
    for i in range(len(bond_array)):
        bond = bond_array[i]
        if ctype <= etypes[i]:
            spc = [ctype, etypes[i]]
            b_dir = bond[1:]
        else:
            spc = [etypes[i], ctype]
            b_dir = bond[1:]

        if spc in exist_species:
            ind = exist_species.index(spc)
            bond_lengths[ind].append([bond[0]])
            bond_dirs[ind].append(b_dir)
        else:
            exist_species.append(spc)
            bond_lengths.append([[bond[0]]])
            bond_dirs.append([b_dir])
    return exist_species, bond_lengths, bond_dirs


@njit
def get_triplets(ctype, etypes, bond_array, cross_bond_inds,
                 cross_bond_dists, triplets):
    exist_species = []
    tris = []
    tri_dir = []

    for m in range(bond_array.shape[0]):
        r1 = bond_array[m, 0]
        c1 = bond_array[m, 1:]
        spc1 = etypes[m]

        for n in range(triplets[m]):
            ind1 = cross_bond_inds[m, m+n+1]
            r2 = bond_array[ind1, 0]
            c2 = bond_array[ind1, 1:]
            spc2 = etypes[ind1]

            c12 = np.sum(c1*c2)
            r12 = np.sqrt(r1**2 + r2**2 - 2*r1*r2*c12)

            spcs_list = [[ctype, spc1, spc2], [ctype, spc2, spc1]]
            for i in range(2):
                spcs = spcs_list[i]
                triplet = array([r2, r1, r12]) if i else array([r1, r2, r12])
                coord = c2 if i else c1 
                if spcs not in exist_species:
                    exist_species.append(spcs)
                    tris.append([triplet])
                    tri_dir.append([coord])
                else:
                    k = exist_species.index(spcs)
                    tris[k].append(triplet)
                    tri_dir[k].append(coord)

    return exist_species, tris, tri_dir


@njit
def self_two_body_mc_en_jit(c2, etypes2,
                         grids, # (n_grids, 1)
                         sig, ls, r_cut, cutoff_func=quadratic_cutoff):

    kern = np.zeros(len(grids), dtype=np.float64)

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    # ls2 = 1 / (2 * ls * ls)

    fj, _ = cutoff_func(r_cut, grids[:, 0], 0)

    # C = 0
    kern += sig2 * fj ** 2 # (n_grids,)

    return kern
 

@njit
def self_three_body_mc_en_jit(c2, etypes2,
                         grids, # (n_grids, 3)
                         sig, ls, r_cut, cutoff_func=quadratic_cutoff):

    kern = np.zeros(len(grids), dtype=np.float64)

    ej1 = etypes2[0]
    ej2 = etypes2[1]

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls2 = 1 / (2 * ls * ls)

    f1, _ = cutoff_func(r_cut, grids[:, 0], 0)
    f2, _ = cutoff_func(r_cut, grids[:, 1], 0)
    f3, _ = cutoff_func(r_cut, grids[:, 2], 0)
    fj = f1 * f2 * f3 # (n_grids, )

    perm_list = get_permutations(c2, ej1, ej2)
    C = 0
    for perm in perm_list:
        perm_grids = np.take(grids, perm, axis=1)
        rij = grids - perm_grids
        C += np.sum(rij * rij, axis=1) # (n_grids, ) adding up three bonds

    kern += sig2 * np.exp(-C * ls2) * fj ** 2 # (n_grids,)

    return kern
       


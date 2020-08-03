import numpy as np
from numba import njit

from typing import List

from flare.struc import Structure
from flare.utils.element_coder import Z_to_element

from flare.mgp.mapxb import MapXbody, SingleMapXbody
from flare.mgp.grid_kernels import grid_kernel, self_kernel

from flare.kernels.utils import from_mask_to_args


class Map2body(MapXbody):
    def __init__(
        self, **kwargs,
    ):
        """
        args: the same arguments as MapXbody, to guarantee they have the same
            input parameters
        """

        self.kernel_name = "twobody"
        self.singlexbody = SingleMap2body
        self.bodies = 2
        self.pred_perm = [[0]]
        self.spc_perm = [[0, 1]]
        self.num_lmp_maps = 0 
        super().__init__(**kwargs)

    def build_bond_struc(self, species_list):
        """
        build a bond structure, used in grid generating
        """

        # 2 body (2 atoms (1 bond) config)
        self.spc = []
        self.num_lmp_maps = 0
        for spc1_ind, spc1 in enumerate(species_list):
            for spc2 in species_list[spc1_ind:]:
                species = [spc1, spc2]
                self.spc.append(sorted(species))
                self.num_lmp_maps += 1

    def get_arrays(self, atom_env):
        return get_bonds(atom_env.ctype, atom_env.etypes, atom_env.bond_array_2)

    def find_map_index(self, spc):
        # use set because of permutational symmetry
        return self.spc.index(sorted(spc))


class SingleMap2body(SingleMapXbody):
    def __init__(self, **kwargs):
        """
        Build 2-body MGP

        bond_struc: Mock structure used to sample 2-body forces on 2 atoms
        """

        self.bodies = 2
        self.grid_dim = 1
        self.kernel_name = "twobody"
        self.pred_perm = [[0]]

        super().__init__(**kwargs)

        # initialize bounds
        self.set_bounds(None, None)

        spc = self.species
        self.species_code = Z_to_element(spc[0]) + "_" + Z_to_element(spc[1])

    def set_bounds(self, lower_bound, upper_bound):
        """
        lower_bound: scalar or array
        upper_bound: scalar or array
        """
        if self.auto_lower:
            if isinstance(lower_bound, float):
                self.bounds[0] = [lower_bound]
            else:
                self.bounds[0] = lower_bound
        if self.auto_upper:
            if isinstance(upper_bound, float):
                self.bounds[1] = [upper_bound]
            else:
                self.bounds[1] = upper_bound

    def construct_grids(self):
        nop = self.grid_num[0]
        bond_lengths = np.linspace(self.bounds[0][0], self.bounds[1][0], nop)
        bond_lengths = np.expand_dims(bond_lengths, axis=1)
        return bond_lengths

    def grid_cutoff(self, bonds, r_cut, coords, derivative, cutoff_func):
        return bonds_cutoff(bonds, r_cut, coords, derivative, cutoff_func)

    def get_grid_kernel(self, kern_type, data, kernel_info, *grid_arrays):
        c2 = self.species[0]
        etypes2 = np.array(self.species[1:])

        _, cutoffs, hyps, hyps_mask = kernel_info
        hyps, r_cut = get_hyps_for_kern(hyps, cutoffs, hyps_mask, c2, etypes2)
        return grid_kernel(
            data,
            self.bodies,
            kern_type,
            get_bonds_for_kern,
            bonds_cutoff,
            c2,
            etypes2,
            hyps,
            r_cut,
            *grid_arrays,
        )

    def get_self_kernel(self, kernel_info, *grid_arrays):
        c2 = self.species[0]
        etypes2 = np.array(self.species[1:])

        _, cutoffs, hyps, hyps_mask = kernel_info
        hyps, r_cut = get_hyps_for_kern(hyps, cutoffs, hyps_mask, c2, etypes2)
        return self_kernel(
            self.bodies, get_permutations, c2, etypes2, hyps, r_cut, *grid_arrays
        )


# -----------------------------------------------------------------------------
#                               Functions
# -----------------------------------------------------------------------------


def bonds_cutoff(bonds, r_cut, coords, derivative, cutoff_func):
    fj, dfj = cutoff_func(r_cut, bonds, coords)
    return fj, dfj


def get_hyps_for_kern(hyps, cutoffs, hyps_mask, c2, etypes2):
    """
    Args:
        data: a single env of a list of envs
    """

    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

    if len(args) == 2:
        hyps, cutoffs = args
        r_cut = cutoffs[0]

    else:
        (
            cutoff_2b,
            cutoff_3b,
            cutoff_mb,
            nspec,
            spec_mask,
            nbond,
            bond_mask,
            ntriplet,
            triplet_mask,
            ncut3b,
            cut3b_mask,
            nmb,
            mb_mask,
            sig2,
            ls2,
            sig3,
            ls3,
            sigm,
            lsm,
        ) = args

        bc1 = spec_mask[c2]
        bc2 = spec_mask[etypes2[0]]
        btype = bond_mask[nspec * bc1 + bc2]
        ls = ls2[btype]
        sig = sig2[btype]
        r_cut = cutoff_2b[btype]
        hyps = [sig, ls]

    return hyps, r_cut


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
            bond_dirs[ind].append([b_dir])
        else:
            exist_species.append(spc)
            bond_lengths.append([[bond[0]]])
            bond_dirs.append([[b_dir]])
    return exist_species, bond_lengths, bond_dirs


@njit
def get_permutations(c1, etypes1, c2, etypes2):
    return [[0]]


def get_bonds_for_kern(env, c2, etypes2):
    return get_bonds_for_kern_jit(env.bond_array_2, env.ctype, env.etypes, c2, etypes2)


@njit
def get_bonds_for_kern_jit(bond_array_1, c1, etypes1, c2, etypes2):

    e2 = etypes2[0]

    bond_list = []
    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, 1:]
        e1 = etypes1[m]

        if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
            bond_list.append([ri, ci[0], ci[1], ci[2]])

    return bond_list

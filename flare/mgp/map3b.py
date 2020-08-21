import numpy as np
from numba import njit
from math import floor, ceil

from typing import List

from flare.struc import Structure
from flare.utils.element_coder import Z_to_element

from flare.mgp.mapxb import MapXbody, SingleMapXbody
from flare.mgp.grid_kernels import grid_kernel, self_kernel

from flare.kernels.utils import from_mask_to_args


class Map3body(MapXbody):
    def __init__(self, **kwargs):

        self.kernel_name = "threebody"
        self.singlexbody = SingleMap3body
        self.bodies = 3
        self.pred_perm = [[0, 1, 2], [1, 0, 2]]
        self.spc_perm = [[0, 1, 2], [0, 2, 1]]
        self.num_lmp_maps = 0
        super().__init__(**kwargs)

    def build_bond_struc(self, species_list):
        """
        build a bond structure, used in grid generating
        """

        # 2 body (2 atoms (1 bond) config)
        self.spc = []
        N_spc = len(species_list)
        self.num_lmp_maps = N_spc ** 3
        for spc1 in species_list:
            for spc2 in species_list:
                for spc3 in species_list:
                    species = [spc1, spc2, spc3]
                    self.spc.append(species)

    def get_arrays(self, atom_env):

        spcs, comp_r, comp_xyz = get_triplets(
            atom_env.ctype,
            atom_env.etypes,
            atom_env.bond_array_3,
            atom_env.cross_bond_inds,
            atom_env.cross_bond_dists,
            atom_env.triplet_counts,
        )

        return spcs, comp_r, comp_xyz

    def find_map_index(self, spc):
        return self.spc.index(spc)



class SingleMap3body(SingleMapXbody):
    def __init__(self, **kwargs):
        """
        Build 3-body MGP

        """

        self.bodies = 3
        self.grid_dim = 3
        self.kernel_name = "threebody"
        self.pred_perm = [[0, 1, 2], [1, 0, 2]]

        super().__init__(**kwargs)

        # initialize bounds
        self.set_bounds(None, None)

        spc = self.species
        self.species_code = "_".join([Z_to_element(spc) for spc in self.species])
        self.kv3name = f"kv3_{self.species_code}"

    def set_bounds(self, lower_bound, upper_bound):
        if self.auto_lower:
            if isinstance(lower_bound, float):
                self.bounds[0] = np.ones(3) * lower_bound
            else:
                self.bounds[0] = lower_bound
        if self.auto_upper:
            if isinstance(upper_bound, float):
                self.bounds[1] = np.ones(3) * upper_bound
            else:
                self.bounds[1] = upper_bound

    def construct_grids(self):
        """
        Return:
            An array of shape (n_grid, 3)
        """
        # build grids in each dimension
        triplets = []
        for d in range(3):
            bonds = np.linspace(
                self.bounds[0][d], self.bounds[1][d], self.grid_num[d], dtype=np.float64
            )
            triplets.append(bonds)

        # concatenate into one array: n_grid x 3
        mesh = np.meshgrid(*triplets, indexing="ij")
        del triplets

        mesh_list = []
        n_grid = np.prod(self.grid_num)
        for d in range(3):
            mesh_list.append(np.reshape(mesh[d], n_grid))

        mesh_list = np.array(mesh_list).T

        return mesh_list

    def grid_cutoff(self, triplets, r_cut, coords, derivative, cutoff_func):
        return bonds_cutoff(triplets, r_cut, coords, derivative, cutoff_func)

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


def bonds_cutoff(triplets, r_cut, coords, derivative, cutoff_func):
    dfj_list = np.zeros((len(triplets), 3), dtype=np.float64)

    if derivative:
        for d in range(3):
            inds = np.arange(3) * 3 + d
            f0, df0 = cutoff_func(r_cut, triplets, coords[:, inds])
            dfj = (
                df0[:, 0] * f0[:, 1] * f0[:, 2]
                + f0[:, 0] * df0[:, 1] * f0[:, 2]
                + f0[:, 0] * f0[:, 1] * df0[:, 2]
            )
            dfj_list[:, d] = dfj
    else:
        f0, _ = cutoff_func(r_cut, triplets, 0)  # (n_grid, 3)

    fj = f0[:, 0] * f0[:, 1] * f0[:, 2]  # (n_grid,)
    fj = np.expand_dims(fj, axis=1)

    return fj, dfj_list


# TODO: move this func to Parameters class
def get_hyps_for_kern(hyps, cutoffs, hyps_mask, c2, etypes2):
    """
    Args:
        data: a single env of a list of envs
    """

    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

    if len(args) == 2:
        hyps, cutoffs = args
        r_cut = cutoffs[1]

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
        bc3 = spec_mask[etypes2[1]]
        ttype = triplet_mask[nspec * nspec * bc1 + nspec * bc2 + bc3]
        ls = ls3[ttype]
        sig = sig3[ttype]
        r_cut = cutoff_3b
        hyps = [sig, ls]

    return hyps, r_cut


@njit
def get_triplets(
    ctype, etypes, bond_array, cross_bond_inds, cross_bond_dists, triplets
):
    exist_species = []
    tris = []
    tri_dir = []

    for m in range(bond_array.shape[0]):
        r1 = bond_array[m, 0]
        c1 = bond_array[m, 1:]
        spc1 = etypes[m]

        for n in range(triplets[m]):
            ind1 = cross_bond_inds[m, m + n + 1]
            r2 = bond_array[ind1, 0]
            c2 = bond_array[ind1, 1:]
            spc2 = etypes[ind1]

            c12 = np.sum(c1 * c2)
            r12 = np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * c12)

            if spc1 <= spc2:
                spcs = [ctype, spc1, spc2]
                triplet = np.array([r1, r2, r12])
                coord = [c1, c2, np.zeros(3)]
            else:
                spcs = [ctype, spc2, spc1]
                triplet = np.array([r2, r1, r12])
                coord = [c2, c1, np.zeros(3)]

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
def get_permutations(c1, etypes1, c2, etypes2):
    ei1 = etypes1[0]
    ei2 = etypes1[1]
    ej1 = etypes2[0]
    ej2 = etypes2[1]

    perms = []
    if c1 == c2:
        if (ei1 == ej1) and (ei2 == ej2):
            perms.append([0, 1, 2])
        if (ei1 == ej2) and (ei2 == ej1):
            perms.append([1, 0, 2])
    if c1 == ej1:
        if (ei1 == ej2) and (ei2 == c2):
            perms.append([1, 2, 0])
        if (ei1 == c2) and (ei2 == ej2):
            perms.append([0, 2, 1])
    if c1 == ej2:
        if (ei1 == ej1) and (ei2 == c2):
            perms.append([2, 1, 0])
        if (ei1 == c2) and (ei2 == ej1):
            perms.append([2, 0, 1])
    return perms


def get_bonds_for_kern(env, c2, etypes2):
    return get_triplets_for_kern_jit(
        env.bond_array_3,
        env.ctype,
        env.etypes,
        env.cross_bond_inds,
        env.cross_bond_dists,
        env.triplet_counts,
        c2,
        etypes2,
    )


@njit
def get_triplets_for_kern_jit(
    bond_array_1,
    c1,
    etypes1,
    cross_bond_inds_1,
    cross_bond_dists_1,
    triplets_1,
    c2,
    etypes2,
):

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
            two_inds = [ind_list[0], ind_list[1]]

            ri1 = bond_array_1[m, 0]
            ci1 = bond_array_1[m, 1:]
            ei1 = etypes1[m]

            two_spec = [all_spec[0], all_spec[1]]
            if ei1 in two_spec:

                ei1_ind = ind_list[0] if ei1 == two_spec[0] else ind_list[1]
                two_spec.remove(ei1)
                two_inds.remove(ei1_ind)
                one_spec = two_spec[0]
                ei2_ind = two_inds[0]

                for n in range(triplets_1[m]):
                    ind1 = cross_bond_inds_1[m, m + n + 1]
                    ei2 = etypes1[ind1]
                    if ei2 == one_spec:

                        ri2 = bond_array_1[ind1, 0]
                        ci2 = bond_array_1[ind1, 1:]

                        ri3 = cross_bond_dists_1[m, m + n + 1]
                        ci3 = np.zeros(3)

                        perms = get_permutations(c1, np.array([ei1, ei2]), c2, etypes2)

                        tri = np.array([ri1, ri2, ri3])
                        crd1 = np.array([ci1[0], ci2[0], ci3[0]])
                        crd2 = np.array([ci1[1], ci2[1], ci3[1]])
                        crd3 = np.array([ci1[2], ci2[2], ci3[2]])

                        # append permutations
                        nperm = len(perms)
                        for iperm in range(nperm):
                            perm = perms[iperm]
                            tricrd = np.take(tri, perm)
                            crd1_p = np.take(crd1, perm)
                            crd2_p = np.take(crd2, perm)
                            crd3_p = np.take(crd3, perm)
                            crd_p = np.vstack((crd1_p, crd2_p, crd3_p))
                            tricrd = np.hstack(
                                (tricrd, crd_p[:, 0], crd_p[:, 1], crd_p[:, 2])
                            )
                            triplet_list.append(tricrd)
                            #tricrd = np.expand_dims(tricrd, axis=0)
                            #triplet_list = np.vstack((triplet_list, tricrd))

    return triplet_list

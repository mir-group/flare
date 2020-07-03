import numpy as np
from math import floor, ceil

from typing import List

from flare.struc import Structure
from flare.utils.element_coder import Z_to_element
from flare.gp_algebra import _global_training_data, _global_training_structures
from flare.kernels.utils import from_mask_to_args

from flare.mgp.mapxb import MapXbody, SingleMapXbody
from flare.mgp.utils import get_triplets, get_kernel_term
from flare.mgp.grid_kernels_3b import triplet_cutoff


class Map3body(MapXbody):
    def __init__(self, **kwargs):

        self.kernel_name = "threebody"
        self.singlexbody = SingleMap3body
        self.bodies = 3
        self.pred_perm = [[0, 1, 2], [1, 0, 2]]
        self.spc_perm = [[0, 1, 2], [0, 2, 1]]
        super().__init__(**kwargs)

    def build_bond_struc(self, species_list):
        """
        build a bond structure, used in grid generating
        """

        # 2 body (2 atoms (1 bond) config)
        self.spc = []
        N_spc = len(species_list)
        for spc1 in species_list:
            for spc2_ind in range(N_spc):
                spc2 = species_list[spc2_ind]
                for spc3 in species_list[spc2_ind:]:
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
        self.species_code = (
            Z_to_element(spc[0])
            + "_"
            + Z_to_element(spc[1])
            + "_"
            + Z_to_element(spc[2])
        )
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

    def grid_cutoff(self, bonds, r_cut, coords, derivative, cutoff_func):
        return triplet_cutoff(bonds, r_cut, coords, derivative, cutoff_func)

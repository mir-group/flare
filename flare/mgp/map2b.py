import numpy as np

from typing import List

from flare.struc import Structure
from flare.utils.element_coder import Z_to_element

from flare.mgp.mapxb import MapXbody, SingleMapXbody
from flare.mgp.utils import get_bonds
from flare.mgp.grid_kernels_2b import bond_cutoff

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
        super().__init__(**kwargs)

    def build_bond_struc(self, species_list):
        """
        build a bond structure, used in grid generating
        """

        # 2 body (2 atoms (1 bond) config)
        self.spc = []
        for spc1_ind, spc1 in enumerate(species_list):
            for spc2 in species_list[spc1_ind:]:
                species = [spc1, spc2]
                self.spc.append(sorted(species))

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

    def set_env(self, grid_env, grid_pt):
        grid_env.bond_array_2 = np.array([[grid_pt[0], 1.0, 0.0, 0.0]])
        return grid_env

    def skip_grid(self, r):
        return False

    def grid_cutoff(self, bonds, r_cut, coords, derivative, cutoff_func):
        return bond_cutoff(bonds, r_cut, coords, derivative, cutoff_func)

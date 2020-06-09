import numpy as np

from typing import List

from flare.struc import Structure
from flare.utils.element_coder import Z_to_element

from flare.mgp.mapxb import MapXbody, SingleMapXbody
from flare.mgp.utils import get_bonds


class Map2body(MapXbody):
    def __init__(self, args):
        '''
        args: the same arguments as MapXbody, to guarantee they have the same 
            input parameters
        '''

        self.kernel_name = "twobody"
        self.singlexbody = SingleMap2body
        self.bodies = 2
        super().__init__(*args)

    def build_bond_struc(self, species_list):
        '''
        build a bond structure, used in grid generating
        '''

        # initialize bounds
        self.bounds = np.ones((2, 1)) * self.lower_bound

        # 2 body (2 atoms (1 bond) config)
        self.spc = []
        self.spc_set = []
        for spc1_ind, spc1 in enumerate(species_list):
            for spc2 in species_list[spc1_ind:]:
                species = [spc1, spc2]
                self.spc.append(species)
                self.spc_set.append(set(species))


    def get_arrays(self, atom_env):

        return get_bonds(atom_env.ctype, atom_env.etypes, atom_env.bond_array_2)




class SingleMap2body(SingleMapXbody):
    def __init__(self, args):
        '''
        Build 2-body MGP

        bond_struc: Mock structure used to sample 2-body forces on 2 atoms
        '''

        self.bodies = 2
        self.kernel_name = 'twobody'

        super().__init__(*args)

        spc = self.species
        self.species_code = Z_to_element(spc[0]) + '_' + Z_to_element(spc[1])


    def construct_grids(self):
        nop = self.grid_num[0]
        bond_lengths = np.linspace(self.bounds[0][0], self.bounds[1][0], nop)
        return bond_lengths


    def set_env(self, grid_env, r):
        grid_env.bond_array_2 = np.array([[r, 1, 0, 0]]) 
        return grid_env

    def skip_grid(self, r):
        return False


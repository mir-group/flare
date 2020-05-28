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

    def build_bond_struc(self, struc_params):
        '''
        build a bond structure, used in grid generating
        '''

        cutoff = 0.1
        cell = struc_params['cube_lat']
        species_list = struc_params['species']
        N_spc = len(species_list)

        # initialize bounds
        self.bounds = np.ones((2, 1)) * self.lower_bound

        # 2 body (2 atoms (1 bond) config)
        self.bond_struc = []
        self.spc = []
        self.spc_set = []
        for spc1_ind, spc1 in enumerate(species_list):
            for spc2 in species_list[spc1_ind:]:
                species = [spc1, spc2]
                self.spc.append(species)
                self.spc_set.append(set(species))
                positions = [[(i+1)/(self.bodies+1)*cutoff, 0, 0]
                             for i in range(self.bodies)]
                spc_struc = Structure(cell, species, positions)
                spc_struc.coded_species = np.array(species)
                self.bond_struc.append(spc_struc)


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

        spc = self.bond_struc.coded_species
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


    def write(self, f, spc):
        '''
        Write LAMMPS coefficient file
        '''
        a = self.bounds[0][0]
        b = self.bounds[1][0]
        order = self.grid_num

        coefs_2 = self.mean.__coeffs__

        elem1 = Z_to_element(spc[0])
        elem2 = Z_to_element(spc[1])
        header_2 = f'{elem1} {elem2} {a} {b} {order}\n'
        f.write(header_2)

        for c, coef in enumerate(coefs_2):
            f.write('{:.10e} '.format(coef))
            if c % 5 == 4 and c != len(coefs_2)-1:
                f.write('\n')

        f.write('\n')

import numpy as np
from math import floor

from typing import List

from flare.struc import Structure
from flare.utils.element_coder import Z_to_element
from flare.gp_algebra import _global_training_data, _global_training_structures
from flare.kernels.utils import from_mask_to_args

from flare.mgp.mapxb import MapXbody, SingleMapXbody
from flare.mgp.utils import get_triplets, get_triplets_en, get_kernel_term,\
    get_permutations
from flare.mgp.grid_kernels import triplet_cutoff


class Map3body(MapXbody):
    def __init__(self, args):

        self.kernel_name = "threebody"
        self.singlexbody = SingleMap3body
        self.bodies = 3
        super().__init__(*args)


    def build_bond_struc(self, species_list):
        '''
        build a bond structure, used in grid generating
        '''

        # initialize bounds
        self.bounds = np.ones((2, 3)) * self.lower_bound
        if self.map_force:
            self.bounds[0][2] = -1
            self.bounds[1][2] = 1

        # 2 body (2 atoms (1 bond) config)
        self.spc = []
        self.spc_set = []
        N_spc = len(species_list)
        for spc1_ind in range(N_spc):
            spc1 = species_list[spc1_ind]
            for spc2_ind in range(N_spc):  # (spc1_ind, N_spc):
                spc2 = species_list[spc2_ind]
                for spc3_ind in range(N_spc):  # (spc2_ind, N_spc):
                    spc3 = species_list[spc3_ind]
                    species = [spc1, spc2, spc3]
                    self.spc.append(species)
                    self.spc_set.append(set(species))

    def get_arrays(self, atom_env):

        if self.map_force:
            get_triplets_func = get_triplets
        else:
            get_triplets_func = get_triplets_en

        spcs, comp_r, comp_xyz = \
            get_triplets_func(atom_env.ctype, atom_env.etypes,
                    atom_env.bond_array_3, atom_env.cross_bond_inds,
                    atom_env.cross_bond_dists, atom_env.triplet_counts)

        return spcs, comp_r, comp_xyz




class SingleMap3body(SingleMapXbody):
    def __init__(self, args):
        '''
        Build 3-body MGP

        '''

        self.bodies = 3
        self.kernel_name = 'threebody'

        super().__init__(*args)

        self.grid_interval = np.min((self.bounds[1]-self.bounds[0])/self.grid_num)

        if self.map_force: # the force mapping use cos angle in the 3rd dim
            self.bounds[1][2] = 1
            self.bounds[0][2] = -1

        spc = self.species
        self.species_code = Z_to_element(spc[0]) + '_' + \
            Z_to_element(spc[1]) + '_' + Z_to_element(spc[2])
        self.kv3name = f'kv3_{self.species_code}'


    def construct_grids(self):
        '''
        Return:
            An array of shape (n_grid, 3)
        '''
        # build grids in each dimension
        bonds_list = []
        for d in range(3):
            bonds = np.linspace(self.bounds[0][d], self.bounds[1][d],
                self.grid_num[d], dtype=np.float64)
            bonds_list.append(bonds)

        # concatenate into one array: n_grid x 3
        mesh = np.meshgrid(*bonds_list)
        mesh_list = []
        n_grid = np.prod(self.grid_num)
        for d in range(3):
            mesh_list.append(np.reshape(mesh[d], n_grid))

        del bonds_list
        return np.array(mesh_list).T


    def set_env(self, grid_env, grid_pt):
        r1, r2, r12 = grid_pt

        if self.map_force:
            cos_angle12 = r12
            x2 = r2 * cos_angle12
            y2 = r2 * np.sqrt(1-cos_angle12**2)
            dist12 = np.linalg.norm(np.array([x2-r1, y2, 0]))
        else:
            dist12 = r12

        grid_env.bond_array_3 = np.array([[r1, 1, 0, 0],
                                       [r2, 0, 0, 0]])
        grid_env.cross_bond_dists = np.array([[0, dist12], [dist12, 0]])

        return grid_env


    def skip_grid(self, grid_pt):
        r1, r2, r12 = grid_pt

        if not self.map_force:
            relaxation = 1/2 * np.max(self.grid_num) * self.grid_interval
            if r1 + r2 < r12 - relaxation:
                return True
            if r1 + r12 < r2 - relaxation:
                return True
            if r12 + r2 < r1 - relaxation:
                return True

        return False


    def _gengrid_numba(self, name, force_block, s, e, env12, kernel_info):
        """
        Loop over different parts of the training set. from element s to element e

        Args:
            name: name of the gp instance
            s: start index of the training data parition
            e: end index of the training data parition
            env12: AtomicEnvironment container of the triplet
            kernel_info: return value of the get_3b_kernel
        """

        kernel, en_kernel, en_force_kernel, cutoffs, hyps, hyps_mask = \
            kernel_info

        if self.map_force:
            prefix = 'force'
            raise NotImplementedError
        else:
            prefix = 'energy'

        args = from_mask_to_args(hyps, cutoffs, hyps_mask)
        r_cut = cutoffs['threebody']

        grids = self.construct_grids()
        coords = np.zeros((grids.shape[0], 9), dtype=np.float64) # padding 0
        coords[:, 0] = np.ones_like(coords[:, 0])
        fj, fdj = triplet_cutoff(grids, r_cut, coords, derivative=True) # TODO: add cutoff func 
        fdj = fdj[0]

        perm_list = get_permutations(env12.ctype, env12.etypes[0], env12.etypes[1])

        if force_block:
            training_data = _global_training_data[name]
            kern_type = f'{prefix}_force'
        else:
            training_data = _global_training_structures[name]
            kern_type = f'{prefix}_energy'

        k_v = [] 
        for m_index in range(s, e):
            data = training_data[m_index]
            kern_vec = en_kernel(kern_type, data, grids, fj, fdj,
                                 env12.ctype, env12.etypes, perm_list,
                                 *args)
            k_v.append(kern_vec)

        k_v = np.vstack(k_v).T
        return k_v

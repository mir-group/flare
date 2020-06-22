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
        super().__init__(**kwargs)


    def build_bond_struc(self, species_list):
        '''
        build a bond structure, used in grid generating
        '''

        # 2 body (2 atoms (1 bond) config)
        self.spc = []
        N_spc = len(species_list)
        for spc1_ind in range(N_spc):
            spc1 = species_list[spc1_ind]
            for spc2_ind in range(N_spc):  # (spc1_ind, N_spc):
                spc2 = species_list[spc2_ind]
                for spc3_ind in range(N_spc):  # (spc2_ind, N_spc):
                    spc3 = species_list[spc3_ind]
                    species = [spc1, spc2, spc3]
                    self.spc.append(species)


    def get_arrays(self, atom_env):

        spcs, comp_r, comp_xyz = \
            get_triplets(atom_env.ctype, atom_env.etypes,
                    atom_env.bond_array_3, atom_env.cross_bond_inds,
                    atom_env.cross_bond_dists, atom_env.triplet_counts)

        return spcs, comp_r, comp_xyz

    def find_map_index(self, spc):
        return self.spc.index(spc)


class SingleMap3body(SingleMapXbody):

    def __init__(self, **kwargs):
        '''
        Build 3-body MGP

        '''

        self.bodies = 3
        self.kernel_name = 'threebody'

        super().__init__(**kwargs)

        # initialize bounds
        self.set_bounds(0, np.ones(3))

        self.grid_interval = np.min((self.bounds[1]-self.bounds[0])/self.grid_num)

        spc = self.species
        self.species_code = Z_to_element(spc[0]) + '_' + \
            Z_to_element(spc[1]) + '_' + Z_to_element(spc[2])
        self.kv3name = f'kv3_{self.species_code}'


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
        '''
        Return:
            An array of shape (n_grid, 3)
        '''
        # build grids in each dimension
        triplets = []
        for d in range(3):
            bonds = np.linspace(self.bounds[0][d], self.bounds[1][d],
                self.grid_num[d], dtype=np.float64)
            triplets.append(bonds)

#        r1 = np.tile(bonds1, (nb12, nb2, 1))
#        r1 = np.moveaxis(r1, -1, 0)
#        r2 = np.tile(bonds2, (nb1, nb12, 1))
#        r2 = np.moveaxis(r2, -1, 1)
#        r12 = np.tile(bonds12, (nb1, nb2, 1))

        # concatenate into one array: n_grid x 3
        mesh = np.meshgrid(*triplets, indexing='ij')
        del triplets

        mesh_list = []
        n_grid = np.prod(self.grid_num)
        for d in range(3):
            mesh_list.append(np.reshape(mesh[d], n_grid))

        mesh_list = np.array(mesh_list).T

        return mesh_list


    def set_env(self, grid_env, grid_pt):
        r1, r2, r12 = grid_pt
        dist12 = r12
        grid_env.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])
        grid_env.cross_bond_dists = np.array([[0, dist12], [dist12, 0]])

        return grid_env


    def skip_grid(self, grid_pt):
        r1, r2, r12 = grid_pt

        return False

        if not self.map_force:
            relaxation = 1/2 * np.max(self.grid_num) * self.grid_interval
            if r1 + r2 < r12 - relaxation:
                return True
            if r1 + r12 < r2 - relaxation:
                return True
            if r12 + r2 < r1 - relaxation:
                return True

        return False


    def _gengrid_numba(self, name, env12, kernel_info, force_block, s, e):
        """
        Loop over different parts of the training set. from element s to element e

        Args:
            name: name of the gp instance
            s: start index of the training data parition
            e: end index of the training data parition
            env12: AtomicEnvironment container of the triplet
            kernel_info: return value of the get_3b_kernel
        """

        grid_kernel, _, _, cutoffs, hyps, hyps_mask = kernel_info

        args = from_mask_to_args(hyps, cutoffs, hyps_mask)
        r_cut = cutoffs['threebody']

        grids = self.construct_grids()
        coords = np.zeros((grids.shape[0], 9), dtype=np.float64) # padding 0
        coords[:, 0] = np.ones_like(coords[:, 0])

        fj, fdj = triplet_cutoff(grids, r_cut, coords, derivative=True) # TODO: add cutoff func
        fdj = fdj[:, [0]]

        if self.map_force:
            prefix = 'force'
        else:
            prefix = 'energy'

        if force_block:
            training_data = _global_training_data[name]
            kern_type = f'{prefix}_force'
        else:
            training_data = _global_training_structures[name]
            kern_type = f'{prefix}_energy'

        k_v = []
        chunk_size = 32 ** 3
        n_grids = grids.shape[0]
        if n_grids > chunk_size:
            n_chunk = ceil(n_grids / chunk_size)
        else:
            n_chunk = 1

        for m_index in range(s, e):
            data = training_data[m_index]
            kern_vec = []
            for g in range(n_chunk):
                gs = chunk_size * g
                ge = np.min((chunk_size * (g + 1), n_grids))
                grid_chunk = grids[gs:ge, :]
                fj_chunk = fj[gs:ge, :]
                fdj_chunk = fdj[gs:ge, :]
                kv_chunk = grid_kernel(kern_type, data, grid_chunk, fj_chunk, fdj_chunk,
                                       env12.ctype, env12.etypes, *args)
                kern_vec.append(kv_chunk)
            kern_vec = np.hstack(kern_vec)
            k_v.append(kern_vec)

        if len(k_v) > 0:
            k_v = np.vstack(k_v).T
        else:
            k_v = np.zeros((grids.shape[0], 0))

        return k_v


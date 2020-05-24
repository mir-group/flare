import time, os, math, inspect, subprocess, json, warnings, pickle
import numpy as np
import multiprocessing as mp

from copy import deepcopy
from math import ceil, floor
from scipy.linalg import solve_triangular
from typing import List

from flare.struc import Structure
from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.gp_algebra import partition_vector, energy_force_vector_unit, \
    force_energy_vector_unit, energy_energy_vector_unit, force_force_vector_unit, \
    _global_training_data, _global_training_structures, \
    get_kernel_vector, en_kern_vec
from flare.parameters import Parameters
from flare.kernels.utils import from_mask_to_args, str_to_kernel_set, str_to_mapped_kernel
from flare.kernels.cutoffs import quadratic_cutoff
from flare.utils.element_coder import Z_to_element, NumpyEncoder


from flare.mgp.utils import get_bonds, get_triplets, get_triplets_en, \
    get_kernel_term
from flare.mgp.splines_methods import PCASplines, CubicSpline


class Map3body_MC:
    def __init__(self,
                 grid_num: List,
                 lower_bound: List,
                 svd_rank: int=0,
                 struc_params: dict,
                 map_force: bool=False,
                 GP: GaussianProcess=None,
                 mean_only: bool=False,
                 container_only: bool=True,
                 lmp_file_name: str='lmp.mgp',
                 n_cpus: int=None,
                 n_sample: int=100):

        # load all arguments as attributes
        self.grid_num = grid_num
        self.lower_bound = lower_bound
        self.svd_rank = svd_rank
        self.struc_params = struc_params
        self.map_force = map_force
        self.mean_only = mean_only
        self.lmp_file_name = lmp_file_name
        self.n_cpus = n_cpus
        self.n_sample = n_sample

        self.hyps_mask = None
        self.cutoffs = None
        self.kernel_name = "threebody"

        # initialize the bounds
        self.bounds = np.ones((2, 3)) * lower_bound
        if self.map_force:
            self.bounds[0][2] = -1
            self.bounds[1][2] = 1

        # if GP exists, the GP setup overrides the grid_params setup
        if GP is not None:

            self.cutoffs = deepcopy(GP.cutoffs)
            self.hyps_mask = deepcopy(GP.hyps_mask)

        self.build_bond_struc(struc_params)
        self.build_map_container(GP)
        self.mean_only = mean_only

        if not container_only and (GP is not None) and \
                (len(GP.training_data) > 0):
            self.build_map(GP)


    def build_bond_struc(self, struc_params):
        '''
        build a bond structure, used in grid generating
        '''

        cutoff = 0.1
        cell = struc_params['cube_lat']
        species_list = struc_params['species']
        N_spc = len(species_list)

        # 2 body (2 atoms (1 bond) config)
        self.bond_struc = []
        self.spc = []
        self.spc_set = []
        bodies = 3
        for spc1_ind in range(N_spc):
            spc1 = species_list[spc1_ind]
            for spc2_ind in range(N_spc):  # (spc1_ind, N_spc):
                spc2 = species_list[spc2_ind]
                for spc3_ind in range(N_spc):  # (spc2_ind, N_spc):
                    spc3 = species_list[spc3_ind]
                    species = [spc1, spc2, spc3]
                    self.spc.append(species)
                    self.spc_set.append(set(species))
                    positions = [[(i+1)/(bodies+1)*cutoff, 0, 0]
                                 for i in range(bodies)]
                    spc_struc = Structure(cell, species, positions)
                    spc_struc.coded_species = np.array(species)
                    self.bond_struc.append(spc_struc)


    def build_map_container(self, GP=None):
        '''
        construct an empty spline container without coefficients.
        '''

        if (GP is not None):
            self.cutoffs = deepcopy(GP.cutoffs)
            self.hyps_mask = deepcopy(GP.hyps_mask)
            if self.kernel_name not in self.hyps_mask['kernels']:
                raise Exception #TODO: deal with this

        self.maps = []

        for b_struc in self.bond_struc:
            if (GP is not None):
                self.bounds[1] = Parameters.get_cutoff(self.kernel_name,
                                 b_struc.coded_species, self.hyps_mask)

            map_3 = Map3body(self.grid_num, self.bounds,
                             b_struc, self.map_force, self.svd_rank,
                             self.mean_only, None, None, #TODO: load_grid & update
                             self.n_cpus, self.n_sample)
            self.maps.append(map_3)


    def build_map(self, GP):
        '''
        generate/load grids and get spline coefficients
        '''

        # double check the container and the GP is the consistent
        if not Parameters.compare_dict(GP.hyps_mask, self.hyps_mask):
            self.build_map_container(GP)

        self.kernel_info = get_kernel_term(GP, self.kernel_name)

        for m in self.maps:
            m.build_map(GP)

        # write to lammps pair style coefficient file
        # TODO
        # self.write_lmp_file(self.lmp_file_name)


    def predict(self, atom_env, mean_only, rank):
        
        if self.mean_only:  # if not build mapping for var
            mean_only = True

        if rank is None:
            rank = self.maps[0].svd_rank

        force_kernel, en_kernel, _, cutoffs, hyps, hyps_mask = self.kernel_info

        args = from_mask_to_args(hyps, cutoffs, hyps_mask)

        if not mean_only:
            if self.map_force:
                kern = np.zeros(3)
                for d in range(3):
                    kern[d] = force_kernel(atom_env, atom_env, d+1, d+1, *args)
            else:
                kern = en_kernel(atom_env, atom_env, *args)

        if self.map_force:
            get_triplets_func = get_triplets
        else:
            get_triplets_func = get_triplets_en

        spcs, comp_r, comp_xyz = \
            get_triplets_func(atom_env.ctype, atom_env.etypes,
                    atom_env.bond_array_3, atom_env.cross_bond_inds,
                    atom_env.cross_bond_dists, atom_env.triplet_counts)

        # predict for each species
        f_spcs = 0
        vir_spcs = 0
        v_spcs = 0
        e_spcs = 0
        for i, spc in enumerate(self.spc_set):
            lengths = np.array(comp_r[i])
            xyzs = np.array(comp_xyz[i])
            map_ind = self.spc.index(spc)
            f, vir, v, e = self.predict_component(lengths, xyzs,
                    mappings[map_ind], mean_only, rank)
            f_spcs += f
            vir_spcs += vir
            v_spcs += v
            e_spcs += e

        return f_spcs, vir_spcs, kern, v_spcs, e_spcs


    def predict_component(self, lengths, xyzs, mapping, mean_only, rank):
        '''
        predict force and variance contribution of one component
        '''
        lengths = np.array(lengths)
        xyzs = np.array(xyzs)

        # predict mean
        if self.map_force: # force mapping
            e = 0
            f_0 = mapping.mean(lengths)
            f_d = np.diag(f_0) @ xyzs
            f = np.sum(f_d, axis=0)

            # predict stress from force components
            vir = np.zeros(6)
            vir_order = ((0,0), (1,1), (2,2), (0,1), (0,2), (1,2))
            for i in range(6):
                vir_i = f_d[:,vir_order[i][0]]\
                        * xyzs[:,vir_order[i][1]] * lengths[:,0]
                vir[i] = np.sum(vir_i)
            vir *= 0.5

            # predict var
            v = np.zeros(3)
            if not mean_only:
                v_0 = mapping.var(lengths, rank)
                v_d = v_0 @ xyzs
                v = mapping.var.V[:,:rank] @ v_d

        else: # energy mapping
            e_0, f_0 = mapping.mean(lengths, with_derivatives=True)
            e = np.sum(e_0) # energy

            # predict forces and stress
            vir = np.zeros(6)
            vir_order = ((0,0), (1,1), (2,2), (1,2), (0,2), (0,1)) # match the ASE order
            f_d1 = np.diag(f_0[:,0,0]) @ xyzs[:,0,:]
            f_d2 = np.diag(f_0[:,1,0]) @ xyzs[:,1,:]
            f_d = f_d1 + f_d2
            f = 3 * np.sum(f_d, axis=0) # force: need to check prefactor 3

            for i in range(6):
                vir_i1 = f_d1[:,vir_order[i][0]]\
                       * xyzs[:,0,vir_order[i][1]] * lengths[:,0]
                vir_i2 = f_d2[:,vir_order[i][0]]\
                       * xyzs[:,1,vir_order[i][1]] * lengths[:,1]
                vir[i] = np.sum(vir_i1 + vir_i2)
            vir *= 1.5

            # predict var
            v = 0
            if not mean_only:
                v_0 = np.expand_dims(np.sum(mapping.var(lengths, rank), axis=1),
                                     axis=1)
                v = mapping.var.V[:,:rank] @ v_0

        return f, vir, v, e




class Map3body:

    def __init__(self, grid_num, bounds, bond_struc: Structure,
                 map_force: bool = False, svd_rank: int = 0, mean_only: bool=False,
                 load_grid: str = '', update: bool = True, n_cpus: int = None,
                 n_sample: int = 100):
        '''
        Build 3-body MGP

        bond_struc: Mock Structure object which contains 3 atoms to get map
        from
        '''
        self.grid_num = grid_num
        self.bounds = bounds
        self.bond_struc = bond_struc
        self.map_force = map_force
        self.svd_rank = svd_rank
        self.mean_only = mean_only
        self.load_grid = load_grid
        self.update = update
        self.n_sample = n_sample

        if self.map_force: # the force mapping use cos angle in the 3rd dim
            self.bounds[1][2] = 1
            self.bounds[0][2] = -1

        spc = bond_struc.coded_species
        self.species_code = Z_to_element(spc[0]) + '_' + \
            Z_to_element(spc[1]) + '_' + Z_to_element(spc[2])
        self.kv3name = f'kv3_{self.species_code}'

        self.build_map_container()
        self.n_cpus = n_cpus
        self.bounds = bounds
        self.mean_only = mean_only

    def GenGrid(self, GP):
        '''
        To use GP to predict value on each grid point, we need to generate the
        kernel vector kv whose length is the same as the training set size.

        1. We divide the training set into several batches, corresponding to
           different segments of kv
        2. Distribute each batch to a processor, i.e. each processor calculate
           the kv segment of one batch for all grids
        3. Collect kv segments and form a complete kv vector for each grid,
           and calculate the grid value by multiplying the complete kv vector
           with GP.alpha
        '''

        if self.n_cpus is None:
            processes = mp.cpu_count()
        else:
            processes = self.n_cpus

        # ------ get 3body kernel info ------
        kernel_info = get_kernel_term(GP, 'threebody')

        # ------ construct grids ------
        n1, n2, n12 = self.grid_num
        bonds1 = np.linspace(self.bounds[0][0], self.bounds[1][0], n1)
        bonds2 = np.linspace(self.bounds[0][1], self.bounds[1][1], n2)
        bonds12 = np.linspace(self.bounds[0][2], self.bounds[1][2], n12)
        grid_means = np.zeros([n1, n2, n12])

        if not self.mean_only:
            grid_vars = np.zeros([n1, n2, n12, len(GP.alpha)])
        else:
            grid_vars = None

        env12 = AtomicEnvironment(self.bond_struc, 0, GP.cutoffs,
            cutoffs_mask=GP.hyps_mask)
        n_envs = len(GP.training_data)
        n_strucs = len(GP.training_structures)
        n_kern = n_envs * 3 + n_strucs

        mapk = str_to_mapped_kernel('3', GP.component, GP.hyps_mask)
        mapped_kernel_info = (kernel_info[0], mapk[0], mapk[1],
                              kernel_info[3], kernel_info[4], kernel_info[5])

        if processes == 1:
            if self.update:
                raise NotImplementedError("the update function is "
                                          "not yet implemented")
            else:
                if (n_envs > 0):
                    k12_v_force = \
                        self._GenGrid_numba(GP.name, 0, n_envs, self.bounds,
                                        n1, n2, n12, env12, mapped_kernel_info)
                if (n_strucs > 0):
                    k12_v_energy = \
                        self._GenGrid_energy(GP.name, 0, n_strucs, bonds1, bonds2,
                                         bonds12, env12, kernel_info)
        else:

            # ------------ force kernels -------------
            if (n_envs > 0):
                if self.update:

                    self.UpdateGrid()



                else:
                    block_id, nbatch = \
                        partition_vector(self.n_sample, n_envs, processes)

                    k12_slice = []
                    with mp.Pool(processes=processes) as pool:
                        for ibatch in range(nbatch):
                            s, e = block_id[ibatch]
                            k12_slice.append(pool.apply_async(
                                self._GenGrid_inner,
                                args=(GP.name, s, e, bonds1, bonds2, bonds12,
                                      env12, kernel_info)))
                        k12_matrix = []
                        for ibatch in range(nbatch):
                            k12_matrix += [k12_slice[ibatch].get()]
                        pool.close()
                        pool.join()

                    del k12_slice
                    k12_v_force = np.vstack(k12_matrix)
                    del k12_matrix

            # set OMB_NUM_THREADS mkl threads number to # of logical cores, per_atom_par=False
            # ------------ force kernels -------------
            if (n_strucs > 0):
                if self.update:

                    self.UpdateGrid()



                else:
                    block_id, nbatch = \
                        partition_vector(self.n_sample, n_strucs, processes)

                    k12_slice = []
                    with mp.Pool(processes=processes) as pool:
                        for ibatch in range(nbatch):
                            s, e = block_id[ibatch]
                            k12_slice.append(pool.apply_async(
                                self._GenGrid_energy,
                                args=(GP.name, s, e, bonds1, bonds2, bonds12,
                                      env12, kernel_info)))
                        k12_matrix = []
                        for ibatch in range(nbatch):
                            k12_matrix += [k12_slice[ibatch].get()]
                        pool.close()
                        pool.join()

                    del k12_slice
                    k12_v_energy = np.vstack(k12_matrix)
                    del k12_matrix

        if (n_envs > 0 and n_strucs > 0):
            k12_v_all = np.vstack([k12_v_force, k12_v_energy])
            k12_v_all = np.moveaxis(k12_v_all, 0, -1)
            del k12_v_force
            del k12_v_energy
        elif (n_envs > 0):
            k12_v_all = np.moveaxis(k12_v_force, 0, -1)
            del k12_v_force
        elif (n_strucs > 0):
            k12_v_all = np.moveaxis(k12_v_energy, 0, -1)
            del k12_v_energy
        else:
            return np.zeros(n1, n2, n12), None

        for b12 in range(len(bonds12)):
            for b1 in range(len(bonds1)):
                for b2 in range(len(bonds2)):
                    k12_v = k12_v_all[b1, b2, b12, :]
                    grid_means[b1, b2, b12] = np.matmul(k12_v, GP.alpha)
                    if not self.mean_only:
                        grid_vars[b1, b2, b12, :] = solve_triangular(GP.l_mat,
                            k12_v, lower=True)


        # Construct file names according to current mapping

        # ------ save mean and var to file -------
        np.save('grid3_mean_'+self.species_code, grid_means)
        np.save('grid3_var_'+self.species_code, grid_vars)

        return grid_means, grid_vars

    def UpdateGrid(self):
        raise NotImplementedError("the update function is "
                "not yet implemented")

        if self.kv3name in os.listdir():
            subprocess.run(['rm', '-rf', self.kv3name])

        os.mkdir(self.kv3name)

        # get the size of saved kv vector
        kv_filename = f'{self.kv3name}/{0}'
        if kv_filename in os.listdir(self.kv3name):
            old_kv_file = np.load(kv_filename+'.npy')
            last_size = int(old_kv_file[0,0])
            new_kv_file[i, :, :last_size] = old_kv_file

            k12_v_all = np.zeros([len(bonds1), len(bonds2), len(bonds12),
                                  size * 3])

            for i in range(n12):
                if f'{self.kv3name}/{i}.npy' in os.listdir(self.kv3name):
                    old_kv_file = np.load(f'{self.kv3name}/{i}.npy')
                    last_size = int(old_kv_file[0,0])
                    #TODO k12_v_all[]
                else:
                    last_size = 0

            # parallelize based on grids, since usually the number of
            # the added training points are small
            ngrids = int(ceil(n12 / processes))
            nbatch = int(ceil(n12 / ngrids))

            block_id = []
            for ibatch in range(nbatch):
                s = int(ibatch * processes)
                e = int(np.min(((ibatch+1)*processes, n12)))
                block_id += [(s, e)]

            k12_slice = []
            for ibatch in range(nbatch):
                k12_slice.append(pool.apply_async(self._GenGrid_inner,
                                                  args=(GP.name, last_size, size,
                                                        bonds1, bonds2, bonds12[s:e],
                                                        env12, kernel_info)))

            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                k12_v_all[:, :, s:e, :] = k12_slice[ibatch].get()


    def _GenGrid_inner(self, name, s, e, bonds1, bonds2, bonds12, env12, kernel_info):

        '''
        Calculate kv segments of the given batch of training data for all grids
        '''

        kernel, ek, efk, cutoffs, hyps, hyps_mask = kernel_info

        # open saved k vector file, and write to new file
        size =  (e - s) * 3
        k12_v = np.zeros([len(bonds1), len(bonds2), len(bonds12), size])
        for b12, r12 in enumerate(bonds12):
            for b1, r1 in enumerate(bonds1):
                for b2, r2 in enumerate(bonds2):

                    if self.map_force:
                        cos_angle12 = r12
                        x2 = r2 * cos_angle12
                        y2 = r2 * np.sqrt(1-cos_angle12**2)
                        dist12 = np.linalg.norm(np.array([x2-r1, y2, 0]))
                    else:
                        dist12 = r12

                    env12.bond_array_3 = np.array([[r1, 1, 0, 0],
                                                   [r2, 0, 0, 0]])
                    env12.cross_bond_dists = np.array([[0, dist12], [dist12, 0]])

                    if self.map_force:
                        k12_v[b1, b2, b12, :] = \
                            force_force_vector_unit(name, s, e, env12, kernel, hyps,
                                               cutoffs, hyps_mask, 1)
                    else:
                        k12_v[b1, b2, b12, :] = energy_force_vector_unit(name, s, e,
                                                        env12, efk,
                                                        hyps, cutoffs, hyps_mask)

        # open saved k vector file, and write to new file
        if self.update:
            self.UpdateGrid_inner()

        return np.moveaxis(k12_v, -1, 0)

    def _GenGrid_energy(self, name, s, e, bonds1, bonds2, bonds12, env12, kernel_info):

        '''
        Calculate kv segments of the given batch of training data for all grids
        '''

        kernel, ek, efk, cutoffs, hyps, hyps_mask = kernel_info

        # open saved k vector file, and write to new file
        size = e - s
        k12_v = np.zeros([len(bonds1), len(bonds2), len(bonds12), size])
        for b12, r12 in enumerate(bonds12):
            for b1, r1 in enumerate(bonds1):
                for b2, r2 in enumerate(bonds2):

                    if self.map_force:
                        cos_angle12 = r12
                        x2 = r2 * cos_angle12
                        y2 = r2 * np.sqrt(1-cos_angle12**2)
                        dist12 = np.linalg.norm(np.array([x2-r1, y2, 0]))
                    else:
                        dist12 = r12

                    env12.bond_array_3 = np.array([[r1, 1, 0, 0],
                                                   [r2, 0, 0, 0]])
                    env12.cross_bond_dists = np.array([[0, dist12], [dist12, 0]])

                    if self.map_force:
                        k12_v[b1, b2, b12, :] = \
                            force_energy_vector_unit(name, s, e, env12, efk, hyps,
                                               cutoffs, hyps_mask, 1)
                    else:
                        k12_v[b1, b2, b12, :] = energy_energy_vector_unit(name, s, e,
                                                        env12, ek,
                                                        hyps, cutoffs, hyps_mask)

        # open saved k vector file, and write to new file
        if self.update:
            self.UpdateGrid_inner()

        return np.moveaxis(k12_v, -1, 0)


    def _GenGrid_numba(self, name, s, e, bounds, nb1, nb2, nb12, env12, kernel_info):
        """
        Loop over different parts of the training set. from element s to element e

        Args:
            name: name of the gp instance
            s: start index of the training data parition
            e: end index of the training data parition
            bonds1: list of bond to consider for edge center-1
            bonds2: list of bond to consider for edge center-2
            bonds12: list of bond to consider for edge 1-2
            env12: AtomicEnvironment container of the triplet
            kernel_info: return value of the get_3b_kernel
        """

        kernel, en_kernel, en_force_kernel, cutoffs, hyps, hyps_mask = \
            kernel_info

        training_data = _global_training_data[name]

        ds = [1, 2, 3]
        size = (e-s) * 3

        bonds1 = np.linspace(bounds[0][0], bounds[1][0], nb1)
        bonds2 = np.linspace(bounds[0][0], bounds[1][0], nb2)
        bonds12 = np.linspace(bounds[0][2], bounds[1][2], nb12)

        r1 = np.ones([nb1, nb2, nb12], dtype=np.float64)
        r2 = np.ones([nb1, nb2, nb12], dtype=np.float64)
        r12 = np.ones([nb1, nb2, nb12], dtype=np.float64)
        for b12 in range(nb12):
            for b1 in range(nb1):
                for b2 in range(nb2):
                    r1[b1, b2, b12] = bonds1[b1]
                    r2[b1, b2, b12] = bonds2[b2]
                    r12[b1, b2, b12] = bonds12[b12]
        del bonds1
        del bonds2
        del bonds12

        args = from_mask_to_args(hyps, cutoffs, hyps_mask)

        k_v = []
        for m_index in range(size):
            x_2 = training_data[int(floor(m_index / 3))+s]
            d_2 = ds[m_index % 3]
            k_v += [[en_force_kernel(x_2, r1, r2, r12,
                                     env12.ctype, env12.etypes,
                                     d_2, *args)]]

        return np.vstack(k_v)

    def _GenGrid_energy_numba(self, name, s, e, bounds, nb1, nb2, nb12, env12, kernel_info):
        """
        Loop over different parts of the training set. from element s to element e

        Args:
            name: name of the gp instance
            s: start index of the training data parition
            e: end index of the training data parition
            bonds1: list of bond to consider for edge center-1
            bonds2: list of bond to consider for edge center-2
            bonds12: list of bond to consider for edge 1-2
            env12: AtomicEnvironment container of the triplet
            kernel_info: return value of the get_3b_kernel
        """

        kernel, en_kernel, en_force_kernel, cutoffs, hyps, hyps_mask = \
            kernel_info

        training_structure = _global_training_structures[name]

        ds = [1, 2, 3]
        size = (e-s) * 3

        bonds1 = np.linspace(bounds[0][0], bounds[1][0], nb1)
        bonds2 = np.linspace(bounds[0][0], bounds[1][0], nb2)
        bonds12 = np.linspace(bounds[0][2], bounds[1][2], nb12)

        r1 = np.ones([nb1, nb2, nb12], dtype=np.float64)
        r2 = np.ones([nb1, nb2, nb12], dtype=np.float64)
        r12 = np.ones([nb1, nb2, nb12], dtype=np.float64)
        for b12 in range(nb12):
            for b1 in range(nb1):
                for b2 in range(nb2):
                    r1[b1, b2, b12] = bonds1[b1]
                    r2[b1, b2, b12] = bonds2[b2]
                    r12[b1, b2, b12] = bonds12[b12]
        del bonds1
        del bonds2
        del bonds12

        args = from_mask_to_args(hyps, cutoffs, hyps_mask)

        k_v = []
        for m_index in range(size):
            structure = training_structures[m_index + s]
            kern_curr = 0
            for environment in structure:
                kern_curr += en_kernel(x, environment, *args)
            kv += [kern_curr]

        return np.hstack(k_v)


    def UpdateGrid_inner(self):
        raise NotImplementedError("the update function is not yet"\
                "implemented")

        s, e = block
        chunk = e - s
        new_kv_file = np.zeros((chunk,
                                self.grid_num[0]*self.grid_num[1]+1,
                                total_size))
        new_kv_file[:,0,0] = np.ones(chunk) * total_size
        for i in range(s, e):
            kv_filename = f'{self.kv3name}/{i}'
            if kv_filename in os.listdir(self.kv3name):
                old_kv_file = np.load(kv_filename+'.npy')
                last_size = int(old_kv_file[0,0])
                new_kv_file[i, :, :last_size] = old_kv_file
            else:
                last_size = 0
        ds = [1, 2, 3]
        nop = self.grid_num[0]

        k12_v = new_kv_file[:,1:,:]
        for i in range(s, e):
            np.save(f'{self.kv3name}/{i}', new_kv_file[i,:,:])



    def build_map_container(self):
        '''
        build 3-d spline function for mean,
        3-d for the low rank approximation of L^{-1}k*
        '''

        # create spline interpolation class object
        self.mean = CubicSpline(self.bounds[0], self.bounds[1],
                                orders=self.grid_num)

        if not self.mean_only:
            self.var = PCASplines(self.bounds[0], self.bounds[1],
                                  orders=self.grid_num,
                                  svd_rank=self.svd_rank)

    def build_map(self, GP):
        # Load grid or generate grid values
        # If load grid was not specified, will be none
        if not self.load_grid:
            y_mean, y_var = self.GenGrid(GP)
        # If load grid is blank string '' or pre-fix, load in
        else:
            y_mean = np.load(self.load_grid+'grid3_mean_' +
                             self.species_code+'.npy')
            y_var = np.load(self.load_grid+'grid3_var_' +
                            self.species_code+'.npy')

        self.mean.set_values(y_mean)
        if not self.mean_only:
            self.var.set_values(y_var)

    def write(self, f, spc):
        a = self.bounds[0]
        b = self.bounds[1]
        order = self.grid_num

        coefs_3 = self.mean.__coeffs__

        elem1 = Z_to_element(spc[0])
        elem2 = Z_to_element(spc[1])
        elem3 = Z_to_element(spc[2])

        header_3 = '{elem1} {elem2} {elem3} {a1} {a2} {a3} {b1}'\
                   ' {b2} {b3:.10e} {order1} {order2} {order3}\n'\
            .format(elem1=elem1, elem2=elem2, elem3=elem3,
                    a1=a[0], a2=a[1], a3=a[2],
                    b1=b[0], b2=b[1], b3=b[2],
                    order1=order[0], order2=order[1], order3=order[2])
        f.write(header_3)

        n = 0
        for i in range(coefs_3.shape[0]):
            for j in range(coefs_3.shape[1]):
                for k in range(coefs_3.shape[2]):
                    coef = coefs_3[i, j, k]
                    f.write('{:.10e} '.format(coef))
                    if n % 5 == 4:
                        f.write('\n')
                    n += 1

        f.write('\n')

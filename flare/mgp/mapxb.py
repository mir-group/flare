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


class MapXbody:
    def __init__(self,
                 grid_num: List,
                 lower_bound: List,
                 svd_rank: int=0,
                 struc_params: dict={},
                 map_force: bool=False,
                 GP: GaussianProcess=None,
                 mean_only: bool=False,
                 container_only: bool=True,
                 lmp_file_name: str='lmp.mgp',
                 n_cpus: int=None,
                 n_sample: int=100):

        # load all arguments as attributes
        self.grid_num = np.array(grid_num)
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

        # to be replaced in subclass
        # self.kernel_name = "xbody"
        # self.singlexbody = SingleMapXbody
        # self.bounds = 0

        # if GP exists, the GP setup overrides the grid_params setup
        if GP is not None:

            self.cutoffs = deepcopy(GP.cutoffs)
            self.hyps_mask = deepcopy(GP.hyps_mask)

        # build_bond_struc is defined in subclass
        self.build_bond_struc(struc_params) 

        # build map
        self.build_map_container(GP)
        if not container_only and (GP is not None) and \
                (len(GP.training_data) > 0):
            self.build_map(GP)

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
            m = self.singlexbody((self.grid_num, self.bounds, b_struc, 
                                  self.map_force, self.svd_rank, self.mean_only,
                                  None, None, self.n_cpus, self.n_sample))
            self.maps.append(m)


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

        kern = 0
        if self.map_force:
            predict_comp = self.predict_single_f_map
            if not mean_only:
                kern = np.zeros(3)
                for d in range(3):
                    kern[d] = force_kernel(atom_env, atom_env, d+1, d+1, *args)
        else:
            predict_comp = self.predict_single_e_map
            if not mean_only:
                kern = en_kernel(atom_env, atom_env, *args)

        spcs, comp_r, comp_xyz = self.get_arrays(atom_env)

        # predict for each species
        f_spcs = 0
        vir_spcs = 0
        v_spcs = 0
        e_spcs = 0
        for i, spc in enumerate(spcs):
            lengths = np.array(comp_r[i])
            xyzs = np.array(comp_xyz[i])
            map_ind = self.spc.index(spc)
            f, vir, v, e = predict_comp(lengths, xyzs,
                    self.maps[map_ind], mean_only, rank)
            f_spcs += f
            vir_spcs += vir
            v_spcs += v
            e_spcs += e

        return f_spcs, vir_spcs, kern, v_spcs, e_spcs


    def predict_single_f_map(self, lengths, xyzs, mapping, mean_only, rank):

        lengths = np.array(lengths)
        xyzs = np.array(xyzs)

        # predict mean
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

        return f, vir, v, e

    def predict_single_e_map(self, lengths, xyzs, mapping, mean_only, rank):
        '''
        predict force and variance contribution of one component
        '''
        lengths = np.array(lengths)
        xyzs = np.array(xyzs)

        e_0, f_0 = mapping.mean(lengths, with_derivatives=True)
        e = np.sum(e_0) # energy

        # predict forces and stress
        vir = np.zeros(6)
        vir_order = ((0,0), (1,1), (2,2), (1,2), (0,2), (0,1)) # match the ASE order

        f_d = np.diag(f_0[:,0,0]) @ xyzs
        f = self.bodies * np.sum(f_d, axis=0) 

        for i in range(6):
            vir_i = f_d[:,vir_order[i][0]]\
                    * xyzs[:,vir_order[i][1]] * lengths[:,0]
            vir[i] = np.sum(vir_i)

        vir *= self.bodies / 2

        # predict var
        v = 0
        if not mean_only:
            v_0 = np.expand_dims(np.sum(mapping.var(lengths, rank), axis=1),
                                 axis=1)
            v = mapping.var.V[:,:rank] @ v_0

        return f, vir, v, e






class SingleMapXbody:
    def __init__(self, grid_num: int, bounds, bond_struc: Structure,
                 map_force=False, svd_rank=0, mean_only: bool=False,
                 load_grid=None, update=None, 
                 n_cpus: int=None, n_sample: int=100):
        '''
        Build 2-body MGP

        bond_struc: Mock structure used to sample 2-body forces on 2 atoms
        '''

        self.grid_num = grid_num
        self.bounds = bounds
        self.bond_struc = bond_struc
        self.map_force = map_force
        self.svd_rank = svd_rank
        self.mean_only = mean_only
        self.load_grid = load_grid
        self.update = update
        self.n_cpus = n_cpus
        self.n_sample = n_sample

        self.build_map_container()


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

        kernel_info = get_kernel_term(GP, self.kernel_name)

        if (self.n_cpus is None):
            processes = mp.cpu_count()
        else:
            processes = self.n_cpus

        # ------ construct grids ------
        n_grid = np.prod(self.grid_num)
        grid_env = AtomicEnvironment(self.bond_struc, 0, GP.cutoffs,
            cutoffs_mask=GP.hyps_mask)

        grid_mean = np.zeros([n_grid])
        if not self.mean_only:
            grid_vars = np.zeros([n_grid, len(GP.alpha)])
        else:
            grid_vars = None

        # -------- get training data info ----------
        n_envs = len(GP.training_data)
        n_strucs = len(GP.training_structures)
        n_kern = n_envs * 3 + n_strucs

        if (n_envs == 0) and (n_strucs == 0):
            return np.zeros([n_grid]), None

        if self.kernel_name == "threebody":
            mapk = str_to_mapped_kernel(self.kernel_name, GP.component, GP.hyps_mask)
            mapped_kernel_info = (kernel_info[0], mapk[0], mapk[1],
                                  kernel_info[3], kernel_info[4], kernel_info[5])

        # ------- call gengrid functions ---------------
        args = [GP.name, grid_env, kernel_info]
        if processes == 1:
            k12_v_force = self._gengrid_serial(args, True, n_envs)
#            k12_v_force = \
#                self._GenGrid_numba(GP.name, 0, n_envs, self.bounds,
#                                n1, n2, n12, env12, mapped_kernel_info)
            k12_v_energy = self._gengrid_serial(args, False, n_strucs)

        else:
            k12_v_force = self._gengrid_par(args, True, n_envs, processes)
            k12_v_energy = self._gengrid_par(args, False, n_strucs, processes)

        k12_v_all = np.hstack([k12_v_force, k12_v_energy])
        del k12_v_force
        del k12_v_energy

        # ------- compute bond means and variances ---------------
        grid_mean = k12_v_all @ GP.alpha
        grid_mean = np.reshape(grid_mean, self.grid_num)

        if not self.mean_only:
            grid_vars = solve_triangular(GP.l_mat, k12_v_all.T, lower=True).T
            tensor_shape = np.array([*self.grid_num, grid_vars.shape[1]])
            grid_vars = np.reshape(grid_vars, tensor_shape)

        # ------ save mean and var to file -------
        np.save(f'grid{self.bodies}_mean{self.species_code}', grid_mean)
        np.save(f'grid{self.bodies}_var{self.species_code}', grid_vars)

        return grid_mean, grid_vars


    def _gengrid_serial(self, args, force_block, n_envs):
        if n_envs == 0:
            n_grid = len(args[1])
            return np.empty((n_grid, 0))

        k12_v = self._gengrid_inner(*args, force_block, 0, n_envs)
        return k12_v


    def _gengrid_par(self, args, force_block, n_envs, processes):
        if n_envs == 0:
            n_grid = len(args[1])
            return np.empty((n_grid, 0))

        with mp.Pool(processes=processes) as pool:

            block_id, nbatch = \
                partition_vector(self.n_sample, n_envs, processes)

            k12_slice = []
            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                k12_slice.append(pool.apply_async(self._gengrid_inner, 
                    args = args + [force_block, s, e]))
            k12_matrix = []
            for ibatch in range(nbatch):
                k12_matrix += [k12_slice[ibatch].get()]
            pool.close()
            pool.join()
        del k12_slice
        k12_v_force = np.vstack(k12_matrix)
        del k12_matrix

        return k12_v_force


    def _gengrid_inner(self, name, grid_env, kern_info, force_block, s, e):
        '''
        Calculate kv segments of the given batch of training data for all grids
        '''

        kernel, ek, efk, cutoffs, hyps, hyps_mask = kern_info
        if force_block:
            size = (e - s) * 3
            force_x_vector_unit = force_force_vector_unit
            force_x_kern = kernel
            energy_x_vector_unit = energy_force_vector_unit
            energy_x_kern = efk
        else:
            size = e - s
            force_x_vector_unit = force_energy_vector_unit
            force_x_kern = efk
            energy_x_vector_unit = energy_energy_vector_unit
            energy_x_kern = ek
           
        grids = self.construct_grids()
        k12_v = np.zeros([len(grids), size])

        for b in range(grids.shape[0]):
            grid_pt = grids[b]
            grid_env = self.set_env(grid_env, grid_pt)

            if self.map_force:
                k12_v[b, :] = force_x_vector_unit(name, s, e, grid_env, 
                    force_x_kern, hyps, cutoffs, hyps_mask, 1)
            else:
                k12_v[b, :] = energy_x_vector_unit(name, s, e, grid_env,
                    energy_x_kern, hyps, cutoffs, hyps_mask)

        return k12_v 


    def build_map_container(self):
        '''
        build 1-d spline function for mean, 2-d for var
        '''
        self.mean = CubicSpline(self.bounds[0], self.bounds[1],
                                orders=self.grid_num)

        if not self.mean_only:
            self.var = PCASplines(self.bounds[0], self.bounds[1],
                                  orders=self.grid_num,
                                  svd_rank=self.svd_rank)

    def build_map(self, GP):
        if not self.load_grid:
            y_mean, y_var = self.GenGrid(GP)
        # If load grid is blank string '' or pre-fix, load in
        else:
            y_mean = np.load(f'{self.load_grid}grid{self.bodies}_mean_{self.species_code}.npy')
            y_var = np.load(f'{self.load_grid}grid{self.bodies}_var_{self.species_code}.npy')

        y_mean, y_var = self.GenGrid(GP)
        print(y_mean.shape)
        print(y_var.shape)
        self.mean.set_values(y_mean)
        if not self.mean_only:
            self.var.set_values(y_var)

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
        header_2 = '{elem1} {elem2} {a} {b} {order}\n'\
            .format(elem1=elem1, elem2=elem2, a=a, b=b, order=order)
        f.write(header_2)

        for c, coef in enumerate(coefs_2):
            f.write('{:.10e} '.format(coef))
            if c % 5 == 4 and c != len(coefs_2)-1:
                f.write('\n')

        f.write('\n')


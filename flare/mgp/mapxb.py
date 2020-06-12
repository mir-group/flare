import warnings
import numpy as np
import multiprocessing as mp

from copy import deepcopy
from math import ceil, floor
from scipy.linalg import solve_triangular
from typing import List

from flare.env import AtomicEnvironment
from flare.kernels.utils import from_mask_to_args
from flare.gp import GaussianProcess
from flare.gp_algebra import partition_vector, energy_force_vector_unit, \
    force_energy_vector_unit, energy_energy_vector_unit, force_force_vector_unit,\
    _global_training_data, _global_training_structures
from flare.parameters import Parameters
from flare.struc import Structure

from flare.mgp.utils import get_kernel_term, str_to_mapped_kernel
from flare.mgp.splines_methods import PCASplines, CubicSpline

global_use_grid_kern = True

class MapXbody:
    def __init__(self,
                 grid_num: List,
                 lower_bound: List or str='auto',
                 upper_bound: List or str='auto',
                 svd_rank = 'auto',
                 species_list: list=[],
                 map_force: bool=False,
                 GP: GaussianProcess=None,
                 mean_only: bool=True,
                 container_only: bool=True,
                 lmp_file_name: str='lmp.mgp',
                 load_grid: str=None,
                 lower_bound_relax: float=0.1,
                 n_cpus: int=None,
                 n_sample: int=100):

        # load all arguments as attributes
        self.grid_num = np.array(grid_num)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.svd_rank = svd_rank
        self.species_list = species_list
        self.map_force = map_force
        self.mean_only = mean_only
        self.lmp_file_name = lmp_file_name
        self.load_grid = load_grid
        self.lower_bound_relax = lower_bound_relax
        self.n_cpus = n_cpus
        self.n_sample = n_sample
        self.spc = []
        self.spc_set = []

        self.build_bond_struc(species_list)

        # build map container only when the bounds are specified
        bounds = [self.lower_bound, self.upper_bound]
        self.build_map_container(bounds)

        if (not container_only) and (GP is not None) and \
                (len(GP.training_data) > 0):
            self.build_map(GP)

    def build_bond_struc(self, species_list):
        raise NotImplementedError("need to be implemented in child class")

    def get_arrays(self, atom_env):
        raise NotImplementedError("need to be implemented in child class")

    def build_map_container(self, bounds):
        '''
        construct an empty spline container without coefficients.
        '''

        self.maps = []
        for spc in self.spc:
            m = self.singlexbody((self.grid_num, bounds, spc,
                                  self.map_force, self.svd_rank, self.mean_only,
                                  self.load_grid, self.lower_bound_relax,
                                  self.n_cpus, self.n_sample))
            self.maps.append(m)


    def build_map(self, GP):
        '''
        generate/load grids and get spline coefficients
        '''

        self.kernel_info = get_kernel_term(GP, self.kernel_name)

        for m in self.maps:
            m.build_map(GP)


    def predict(self, atom_env, mean_only):

        if self.mean_only:  # if not build mapping for var
            mean_only = True

        force_kernel, en_kernel, _, cutoffs, hyps, hyps_mask = self.kernel_info

        args = from_mask_to_args(hyps, cutoffs, hyps_mask)

        kern = 0
        if not mean_only:
            if self.map_force:
                kern = np.zeros(3)
                for d in range(3):
                    kern[d] = force_kernel(atom_env, atom_env, d+1, d+1, *args)
            else:
                kern = en_kernel(atom_env, atom_env, *args)

        spcs, comp_r, comp_xyz = self.get_arrays(atom_env)

        # predict for each species
        f_spcs = np.zeros(3)
        vir_spcs = np.zeros(6)
        v_spcs = np.zeros(3) if self.map_force else 0
        e_spcs = 0
        for i, spc in enumerate(spcs):
            lengths = np.array(comp_r[i])
            xyzs = np.array(comp_xyz[i])
            map_ind = self.find_map_index(spc)

            print('spc, lengths, xyz', spc)
            print(np.hstack([lengths, xyzs]))
            f, vir, v, e = self.maps[map_ind].predict(lengths, xyzs,
                self.map_force, mean_only)
            f_spcs += f
            vir_spcs += vir
            v_spcs += v
            e_spcs += e

        return f_spcs, vir_spcs, kern, v_spcs, e_spcs


    def write(self, f):
        for m in self.maps:
            m.write(f)



class SingleMapXbody:
    def __init__(self, grid_num: int, bounds, species: str,
                 map_force=False, svd_rank=0, mean_only: bool=False,
                 load_grid=None, lower_bound_relax=0.1,
                 n_cpus: int=None, n_sample: int=100):

        self.grid_num = grid_num
        self.bounds = deepcopy(bounds)
        self.species = species
        self.map_force = map_force
        self.svd_rank = svd_rank
        self.mean_only = mean_only
        self.load_grid = load_grid
        self.lower_bound_relax = lower_bound_relax
        self.n_cpus = n_cpus
        self.n_sample = n_sample

        self.auto_lower = (bounds[0] == 'auto')
        self.auto_upper = (bounds[1] == 'auto')

        self.hyps_mask = None
        self.use_grid_kern = global_use_grid_kern

        if not self.auto_lower and not self.auto_upper:
            self.build_map_container()

    def set_bounds(self, lower_bound, upper_bound):
        raise NotImplementedError("need to be implemented in child class")

    def construct_grids(self):
        raise NotImplementedError("need to be implemented in child class")

    def set_env(self, grid_env, r):
        raise NotImplementedError("need to be implemented in child class")

    def skip_grid(self, r):
        raise NotImplementedError("need to be implemented in child class")

    def get_grid_env(self, GP):
        if isinstance(GP.cutoffs, dict):
            max_cut = np.max(list(GP.cutoffs.values()))
        else:
            max_cut = np.max(GP.cutoffs)
        big_cell = np.eye(3) * (2 * max_cut + 1)
        positions = [[(i+1)/(self.bodies+1)*0.1, 0, 0]
                     for i in range(self.bodies)]
        grid_struc = Structure(big_cell, self.species, positions)
        grid_env = AtomicEnvironment(grid_struc, 0, GP.cutoffs,
            cutoffs_mask=GP.hyps_mask)

        return grid_env


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
        grid_mean = np.zeros([n_grid])
        if not self.mean_only:
            grid_vars = np.zeros([n_grid, len(GP.alpha)])
        else:
            grid_vars = None

        grid_env = self.get_grid_env(GP)

        # -------- get training data info ----------
        n_envs = len(GP.training_data)
        n_strucs = len(GP.training_structures)

        if (n_envs == 0) and (n_strucs == 0):
            warnings.warn("No training data, will return 0")
            return np.zeros([n_grid]), None

        # ------- call gengrid functions ---------------
        args = [GP.name, grid_env, kernel_info]
        self.use_grid_kern = True
        if self.use_grid_kern:
            try:
                mapk = str_to_mapped_kernel(self.kernel_name, GP.component, GP.hyps_mask)
                mapped_kernel_info = (mapk,
                                      kernel_info[3], kernel_info[4], kernel_info[5])
            except:
                self.use_grid_kern = False

        if processes == 1:
            args = [GP.name, grid_env, kernel_info]
            if self.use_grid_kern: # TODO: finish force mapping
                k12_v_force = self._gengrid_numba(GP.name, True, 0, n_envs, grid_env,
                                                  mapped_kernel_info)
                k12_v_energy = self._gengrid_numba(GP.name, False, 0, n_strucs, grid_env,
                                                  mapped_kernel_info)
            else:
                k12_v_force = self._gengrid_serial(args, True, n_envs)
                k12_v_energy = self._gengrid_serial(args, False, n_strucs)

            k12_v_force_inner = self._gengrid_serial(args, True, n_envs)

            try:
                assert np.allclose(k12_v_force, k12_v_force_inner, rtol=1e-3)
            except:
                print(k12_v_force)
                print(k12_v_force_inner)

                print(np.array(np.isclose(k12_v_force, k12_v_force_inner), dtype=int))
                raise Exception
        else:
            if self.use_grid_kern:
                args = [GP.name, grid_env, mapped_kernel_info]
            else:
                args = [GP.name, grid_env, kernel_info]
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
        np.save(f'grid{self.bodies}_mean_{self.species_code}', grid_mean)
        np.save(f'grid{self.bodies}_var_{self.species_code}', grid_vars)

        return grid_mean, grid_vars


    def _gengrid_serial(self, args, force_block, n_envs):
        if n_envs == 0:
            n_grid = np.prod(self.grid_num)
            return np.empty((n_grid, 0))

        k12_v = self._gengrid_inner(*args, force_block, 0, n_envs)
        return k12_v


    def _gengrid_par(self, args, force_block, n_envs, processes):

        if n_envs == 0:
            n_grid = np.prod(self.grid_num)
            return np.empty((n_grid, 0))

        with mp.Pool(processes=processes) as pool:

            block_id, nbatch = \
                partition_vector(self.n_sample, n_envs, processes)

            threebody = False
            if self.use_grid_kern:
                GP_name, grid_env, mapped_kernel_info = args
                threebody = True

            k12_slice = []
            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                # if threebody:
                #     k12_slice.append(pool.apply_async(self._gengrid_numba,
                #         args = (GP_name, force_block, s, e, grid_env, mapped_kernel_info)))
                # else:
                k12_slice.append(pool.apply_async(self._gengrid_inner,
                    args = args + [force_block, s, e]))
            k12_matrix = []
            for ibatch in range(nbatch):
                k12_matrix += [k12_slice[ibatch].get()]
            pool.close()
            pool.join()
        del k12_slice
        k12_v_force = np.hstack(k12_matrix)
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

            if not self.skip_grid(grid_pt):
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
            if self.svd_rank == 'auto':
                warnings.warn("The containers for variance are not built because svd_rank='auto'")
            if isinstance(self.svd_rank, int):
                self.var = PCASplines(self.bounds[0], self.bounds[1],
                                      orders=self.grid_num,
                                      svd_rank=self.svd_rank)

    def build_map(self, GP):

        rebuild_container = False

        # double check the container and the GP is consistent
        if not Parameters.compare_dict(GP.hyps_mask, self.hyps_mask):
            rebuild_container = True

        # check if bounds are updated
        lower_bound = self.bounds[0]
        min_dist = self.search_lower_bound(GP)
        if min_dist < np.max(lower_bound): # change lower bound
            warnings.warn('The minimal distance in training data is lower than \
                    the current lower bound, will reset lower bound')

        if self.auto_lower or (min_dist < np.max(lower_bound)):
            lower_bound = np.max((min_dist - self.lower_bound_relax, 0))
            rebuild_container = True

        upper_bound = self.bounds[1]
        if self.auto_upper:
            upper_bound = Parameters.get_cutoff(self.kernel_name,
                self.species, GP.hyps_mask)
            rebuild_container = True

        if rebuild_container:
            self.set_bounds(lower_bound, upper_bound)
            self.build_map_container()

        if not self.load_grid:
            y_mean, y_var = self.GenGrid(GP)
        else:
            y_mean = np.load(f'{self.load_grid}grid{self.bodies}_mean_{self.species_code}.npy')
            y_var = np.load(f'{self.load_grid}grid{self.bodies}_var_{self.species_code}.npy')

        self.mean.set_values(y_mean)
        if not self.mean_only:
            if self.svd_rank == 'auto':
                self.var = PCASplines(self.bounds[0], self.bounds[1],
                                      orders=self.grid_num,
                                      svd_rank=np.min(y_var.shape))
                self.var.set_values(y_var)

        self.hyps_mask = deepcopy(GP.hyps_mask)


    def search_lower_bound(self, GP):
        '''
        If the lower bound is set to be 'auto', search the minimal interatomic
        distances in the training set of GP.
        '''
        upper_bound = Parameters.get_cutoff(self.kernel_name,
                self.species, GP.hyps_mask)

        lower_bound = np.min(upper_bound)
        for env in _global_training_data[GP.name]:
            min_dist = env.bond_array_2[0][0]
            if min_dist < lower_bound:
                lower_bound = min_dist

        for struc in _global_training_structures[GP.name]:
            for env in struc:
                min_dist = env.bond_array_2[0][0]
                if min_dist < lower_bound:
                    lower_bound = min_dist

        return lower_bound


    def predict(self, lengths, xyzs, map_force, mean_only):
        '''
        predict force and variance contribution of one component
        '''

        assert map_force == self.map_force, f'The mapping is built for'\
            'map_force={self.map_force}, can not predict for map_force={map_force}'

        lengths = np.array(lengths)
        xyzs = np.array(xyzs)

        if self.map_force:
            # predict forces and energy
            e = 0
            f_0 = self.mean(lengths)
            f_d = np.diag(f_0) @ xyzs
            f = np.sum(f_d, axis=0)

            # predict var
            v = np.zeros(3)
            if not mean_only:
                v_0 = self.var(lengths)
                v_d = v_0 @ xyzs
                v = self.var.V @ v_d

        else:
            # predict forces and energy
            e_0, f_0 = self.mean(lengths, with_derivatives=True)
            print('f_0')
            print(f_0)
            e = np.sum(e_0) # energy
            if lengths.shape[1] == 1:
                f_d = np.diag(f_0[:,0,0]) @ xyzs
            else:
                f_d = np.diag(f_0[:,0,0]) @ xyzs
            f = self.bodies * np.sum(f_d, axis=0)

            # predict var
            v = 0
            if not mean_only:
                v_0 = np.expand_dims(np.sum(self.var(lengths), axis=1),
                                     axis=1)
                v = self.var.V @ v_0

        # predict virial stress
        vir = np.zeros(6)
        vir_order = ((0,0), (1,1), (2,2), (1,2), (0,2), (0,1)) # match the ASE order
        for i in range(6):
            vir_i = f_d[:,vir_order[i][0]]\
                    * xyzs[:,vir_order[i][1]] * lengths[:,0]
            vir[i] = np.sum(vir_i)

        vir *= self.bodies / 2
        return f, vir, v, e


    def write(self, f):
        '''
        Write LAMMPS coefficient file

        This implementation only works for 2b and 3b. User should
        implement overload in the actual class if the new kernel
        has different coefficient format

        In the future, it should be changed to writing in bin/hex
        instead of decimal
        '''

        # write header
        elems = self.species_code.split('_')
        a = self.bounds[0]
        b = self.bounds[1]
        order = self.grid_num

        header = ' '.join(elems)
        header += ' '+' '.join(map(repr, a))
        header += ' '+' '.join(map(repr, b))
        header += ' '+' '.join(map(str, order))
        f.write(header + '\n')

        # write coefficients
        coefs = self.mean.__coeffs__
        self.write_flatten_coeff(f, coefs)

    def write_flatten_coeff(self, f, coefs):
        """
        flatten the coefficient and write it as
        a block. each line has no more than 5 element.
        the accuracy is restricted to .10
        """
        coefs = coefs.reshape([-1])
        for c, coef in enumerate(coefs):
            f.write(' '+repr(coef))
            if c % 5 == 4 and c != len(coefs)-1:
                f.write('\n')
        f.write('\n')

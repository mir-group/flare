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

from flare.mgp.mapxb import MapXbody, SingleMapXbody
from flare.mgp.utils import get_bonds, get_kernel_term
from flare.mgp.splines_methods import PCASplines, CubicSpline


class Map2body(MapXbody):
    def __init__(self, args):

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




class SingleMap2body:
    def __init__(self, grid_num: int, bounds, bond_struc: Structure,
                 map_force=False, svd_rank=0, mean_only: bool=False,
                 load_grid=None, update=None, n_cpus: int=None, n_sample: int=100):
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
        self.n_cpus = n_cpus
        self.n_sample = n_sample

        spc = bond_struc.coded_species
        self.species_code = Z_to_element(spc[0]) + '_' + Z_to_element(spc[1])

#        arg_dict = inspect.getargvalues(inspect.currentframe())[3]
#        del arg_dict['self']
#        self.__dict__.update(arg_dict)

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

        kernel_info = get_kernel_term(GP, 'twobody')

        if (self.n_cpus is None):
            processes = mp.cpu_count()
        else:
            processes = self.n_cpus

        # ------ construct grids ------
        nop = self.grid_num
        bond_lengths = np.linspace(self.bounds[0][0], self.bounds[1][0], nop)
        env12 = AtomicEnvironment(
            self.bond_struc, 0, GP.cutoffs, cutoffs_mask=GP.hyps_mask)

        # --------- calculate force kernels ---------------
        n_envs = len(GP.training_data)
        n_strucs = len(GP.training_structures)
        n_kern = n_envs * 3 + n_strucs

        if (n_envs > 0):
            with mp.Pool(processes=processes) as pool:

                block_id, nbatch = \
                    partition_vector(self.n_sample, n_envs, processes)

                k12_slice = []
                for ibatch in range(nbatch):
                    s, e = block_id[ibatch]
                    k12_slice.append(pool.apply_async(
                        self._GenGrid_inner, args=(GP.name, s, e, bond_lengths,
                                                   env12, kernel_info)))
                k12_matrix = []
                for ibatch in range(nbatch):
                    k12_matrix += [k12_slice[ibatch].get()]
                pool.close()
                pool.join()
            del k12_slice
            k12_v_force = np.vstack(k12_matrix)
            del k12_matrix

        # --------- calculate energy kernels ---------------
        if (n_strucs > 0):
            with mp.Pool(processes=processes) as pool:
                block_id, nbatch = \
                    partition_vector(self.n_sample, n_strucs, processes)

                k12_slice = []
                for ibatch in range(nbatch):
                    s, e = block_id[ibatch]
                    k12_slice.append(pool.apply_async(
                        self._GenGrid_energy,
                        args=(GP.name, s, e, bond_lengths, env12, kernel_info)))
                k12_matrix = []
                for ibatch in range(nbatch):
                    k12_matrix += [k12_slice[ibatch].get()]
                pool.close()
                pool.join()
            del k12_slice
            k12_v_energy = np.vstack(k12_matrix)
            del k12_matrix

        if (n_strucs > 0 and n_envs > 0):
            k12_v_all = np.vstack([k12_v_force, k12_v_energy])
            k12_v_all = np.moveaxis(k12_v_all, 0, -1)
            del k12_v_force
            del k12_v_energy
        elif (n_strucs > 0):
            k12_v_all = np.moveaxis(k12_v_energy, 0, -1)
            del k12_v_energy
        elif (n_envs > 0):
            k12_v_all = np.moveaxis(k12_v_force, 0, -1)
            del k12_v_force
        else:
            return np.zeros([nop]), None

        # ------- compute bond means and variances ---------------
        bond_means = np.zeros([nop])
        if not self.mean_only:
            bond_vars = np.zeros([nop, len(GP.alpha)])
        else:
            bond_vars = None
        for b, _ in enumerate(bond_lengths):
            k12_v = k12_v_all[b, :]
            bond_means[b] = np.matmul(k12_v, GP.alpha)
            if not self.mean_only:
                bond_vars[b, :] = solve_triangular(GP.l_mat, k12_v, lower=True)

        write_species_name = ''
        for x in self.bond_struc.coded_species:
            write_species_name += "_" + Z_to_element(x)
        # ------ save mean and var to file -------
        np.save('grid2_mean' + write_species_name, bond_means)
        np.save('grid2_var' + write_species_name, bond_vars)

        return bond_means, bond_vars

    def _GenGrid_inner(self, name, s, e, bond_lengths,
                       env12, kernel_info):
        '''
        Calculate kv segments of the given batch of training data for all grids
        '''

        kernel, ek, efk, cutoffs, hyps, hyps_mask = kernel_info
        size = e - s
        k12_v = np.zeros([len(bond_lengths), size*3])
        for b, r in enumerate(bond_lengths):
            env12.bond_array_2 = np.array([[r, 1, 0, 0]])
            if self.map_force:
                k12_v[b, :] = force_force_vector_unit(name, s, e, env12, kernel, hyps,
                                               cutoffs, hyps_mask, 1)

            else:
                k12_v[b, :] = energy_force_vector_unit(name, s, e,
                        env12, efk, hyps, cutoffs, hyps_mask)
        return np.moveaxis(k12_v, 0, -1)

    def _GenGrid_energy(self, name, s, e, bond_lengths, env12, kernel_info):
        '''
        Calculate kv segments of the given batch of training data for all grids
        '''

        kernel, ek, efk, cutoffs, hyps, hyps_mask = kernel_info
        size = e - s
        k12_v = np.zeros([len(bond_lengths), size])
        for b, r in enumerate(bond_lengths):
            env12.bond_array_2 = np.array([[r, 1, 0, 0]])

            if self.map_force:
                k12_v[b, :] = force_energy_vector_unit(name, s, e, env12, efk,
                    hyps, cutoffs, hyps_mask, 1)
            else:
                k12_v[b, :] = energy_energy_vector_unit(name, s, e,
                    env12, ek, hyps, cutoffs, hyps_mask)
        return np.moveaxis(k12_v, 0, -1)


    def build_map_container(self):
        '''
        build 1-d spline function for mean, 2-d for var
        '''
        self.mean = CubicSpline(self.bounds[0], self.bounds[1],
                                orders=[self.grid_num])

        if not self.mean_only:
            self.var = PCASplines(self.bounds[0], self.bounds[1],
                                  orders=[self.grid_num],
                                  svd_rank=self.svd_rank)

    def build_map(self, GP):
        y_mean, y_var = self.GenGrid(GP)
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
        header_2 = f'{elem1} {elem2} {a} {b} {order}\n'
        f.write(header_2)

        for c, coef in enumerate(coefs_2):
            f.write('{:.10e} '.format(coef))
            if c % 5 == 4 and c != len(coefs_2)-1:
                f.write('\n')

        f.write('\n')

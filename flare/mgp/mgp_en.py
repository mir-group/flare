import time
import os
import math
import inspect
import subprocess
import numpy as np
import multiprocessing as mp
import json
import warnings

from copy import deepcopy
import pickle as pickle
from scipy.linalg import solve_triangular
from typing import List

from flare.struc import Structure
from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.gp_algebra import partition_vector, energy_force_vector_unit, \
    energy_energy_vector_unit
from flare.kernels.utils import from_mask_to_args, str_to_kernel_set
from flare.cutoffs import quadratic_cutoff
from flare.util import Z_to_element
from flare.mgp.utils import get_bonds, get_triplets, get_triplets_en, \
        get_2bkernel, get_3bkernel
from flare.mgp.splines_methods import PCASplines, CubicSpline
from flare.util import Z_to_element, NumpyEncoder


class MappedGaussianProcess:
    '''
    Build Mapped Gaussian Process (MGP)
    and automatically save coefficients for LAMMPS pair style.

    :param: struc_params : Parameters for a dummy structure which will be
        internally used to probe/store forces associated with different atomic
        configurations
    :param: grid_params : Parameters for the mapping itself, such as
        grid size of spline fit, etc.
    :param: mean_only : if True: only build mapping for mean (force)
    :param: container_only : if True: only build splines container
        (with no coefficients)
    :param: GP: None or a GaussianProcess object. If a GP is input,
        and autorun is true, automatically build a mapping corresponding
        to the GaussianProcess.
    :param: lmp_file_name : LAMMPS coefficient file name
    :param: autorun: Attempt to build map immediately
    Examples:

    >>> struc_params = {'species': [0, 1],
                        'cube_lat': cell, # should input the cell matrix
                        'mass_dict': {'0': 27 * unit, '1': 16 * unit}}
    >>> grid_params =  {'bounds_2': [[1.2], [3.5]],
                                    # [[lower_bound], [upper_bound]]
                                    # These describe the lower and upper
                                    # bounds used to specify the 2-body spline
                                    # fits.
                        'bounds_3': [[1.2, 1.2, 1.2], [3.5, 3.5, 3.5]],
                                    # [[lower,lower,lower],[upper,upper,upper]]
                                    # Values describe lower and upper bounds
                                    # for the bondlength-bondlength-bondlength
                                    # grid used to construct and fit 3-body
                                    # kernels; note that for force MGPs
                                    # bondlength-bondlength-costheta
                                    # are the bounds used instead.
                        'bodies':   [2, 3] # use 2+3 body
                        'grid_num_2': 64,# Fidelity of the grid
                        'grid_num_3': [16, 16, 16],# Fidelity of the grid
                        'svd_rank_2': 64, #Fidelity of uncertainty estimation
                        'svd_rank_3': 16**3,
                        'update': True, # if True: accelerating grids
                                        # generating by saving intermediate
                                        # coeff when generating grids
                        'load_grid': None  # Used to load from file
                        }
    '''

    def __init__(self, grid_params: dict, struc_params: dict,
                 GP: GaussianProcess = None, mean_only: bool = False,
                 container_only: bool = True, lmp_file_name: str = 'lmp.mgp',
                 n_cpus: int = None, n_sample: int = 100,
                 autorun: bool = True):

        # load all arguments as attributes
        self.mean_only = mean_only
        self.lmp_file_name = lmp_file_name
        self.n_cpus = n_cpus
        self.n_sample = n_sample
        self.grid_params = grid_params
        self.struc_params = struc_params

        # It would be helpful to define the attributes explicitly.
        self.__dict__.update(grid_params)

        # if GP exists, the GP setup overrides the grid_params setup
        if GP is not None:

            self.cutoffs = GP.cutoffs

            self.bodies = []
            if "two" in GP.kernel_name:
                self.bodies.append(2)
                self.kernel2b_info = get_2bkernel(GP)
            if "three" in GP.kernel_name:
                self.bodies.append(3)
                self.kernel3b_info = get_3bkernel(GP)

        self.build_bond_struc(struc_params)
        self.maps_2 = []
        self.maps_3 = []
        self.build_map_container()
        self.mean_only = mean_only

        if not container_only and (GP is not None) and \
                (len(GP.training_data) > 0) and autorun:
            self.build_map(GP)

    def build_map_container(self):
        '''
        construct an empty spline container without coefficients.
        '''
        if 2 in self.bodies:
            for b_struc in self.bond_struc[0]:
                map_2 = Map2body(self.grid_num_2, self.bounds_2,
                                 b_struc, self.svd_rank_2,
                                 self.mean_only, self.n_cpus, self.n_sample)
                self.maps_2.append(map_2)
        if 3 in self.bodies:
            for b_struc in self.bond_struc[1]:
                map_3 = Map3body(self.grid_num_3, self.bounds_3,
                                 b_struc, self.svd_rank_3,
                                 self.mean_only,
                                 self.grid_params['load_grid'],
                                 self.update, self.n_cpus, self.n_sample)
                self.maps_3.append(map_3)

    def build_map(self, GP):
        '''
        generate/load grids and get spline coefficients
        '''

        if 2 in self.bodies:
            self.kernel2b_info = get_2bkernel(GP)
        if 3 in self.bodies:
            self.kernel3b_info = get_3bkernel(GP)

        for map_2 in self.maps_2:
            map_2.build_map(GP)
        for map_3 in self.maps_3:
            map_3.build_map(GP)

        # write to lammps pair style coefficient file
        self.write_lmp_file(self.lmp_file_name)

    def build_bond_struc(self, struc_params):

        '''
        build a bond structure, used in grid generating
        '''

        cutoff = np.min(self.cutoffs)
        cell = struc_params['cube_lat']
        mass_dict = struc_params['mass_dict']
        species_list = struc_params['species']
        N_spc = len(species_list)

        # 2 body (2 atoms (1 bond) config)
        bond_struc_2 = []
        spc_2 = []
        spc_2_set = []
        if 2 in self.bodies:
            bodies = 2
            for spc1_ind, spc1 in enumerate(species_list):
                for spc2 in species_list[spc1_ind:]:
                    species = [spc1, spc2]
                    spc_2.append(species)
                    spc_2_set.append(set(species))
                    positions = [[(i+1)/(bodies+1)*cutoff, 0, 0]
                                 for i in range(bodies)]
                    spc_struc = \
                        Structure(cell, species, positions, mass_dict)
                    spc_struc.coded_species = np.array(species)
                    bond_struc_2.append(spc_struc)

        #  3 body (3 atoms (1 triplet) config)
        bond_struc_3 = []
        spc_3 = []
        if 3 in self.bodies:
            bodies = 3
            for spc1_ind in range(N_spc):
                spc1 = species_list[spc1_ind]
                for spc2_ind in range(N_spc):  # (spc1_ind, N_spc):
                    spc2 = species_list[spc2_ind]
                    for spc3_ind in range(N_spc):  # (spc2_ind, N_spc):
                        spc3 = species_list[spc3_ind]
                        species = [spc1, spc2, spc3]
                        spc_3.append(species)
                        positions = [[(i+1)/(bodies+1)*cutoff, 0, 0]
                                     for i in range(bodies)]
                        spc_struc = Structure(cell, species, positions,
                                              mass_dict)
                        spc_struc.coded_species = np.array(species)
                        bond_struc_3.append(spc_struc)
#                        if spc1 != spc2:
#                            species = [spc2, spc3, spc1]
#                            spc_3.append(species)
#                            positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                        for i in range(bodies)]
#                            spc_struc = Structure(cell, species, positions,
#                                                        mass_dict)
#                            spc_struc.coded_species = np.array(species)
#                            bond_struc_3.append(spc_struc)
#                        if spc2 != spc3:
#                            species = [spc3, spc1, spc2]
#                            spc_3.append(species)
#                            positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                        for i in range(bodies)]
#                            spc_struc = Structure(cell, species, positions,
#                                                        mass_dict)
#                            spc_struc.coded_species = np.array(species)
#                            bond_struc_3.append(spc_struc)

        self.bond_struc = [bond_struc_2, bond_struc_3]
        self.spcs = [spc_2, spc_3]
        self.spcs_set = [spc_2_set, spc_3]

    def predict(self, atom_env: AtomicEnvironment, mean_only: bool = False)\
            -> (float, 'ndarray', 'ndarray', float):
        '''
        predict force, variance, stress and local energy for given
            atomic environment
        Args:
            atom_env: atomic environment (with a center atom and its neighbors)
            mean_only: if True: only predict force (variance is always 0)
        Return:
            force: 3d array of atomic force
            variance: 3d array of the predictive variance
            stress: 6d array of the virial stress
            energy: the local energy (atomic energy)
        '''
        if self.mean_only:  # if not build mapping for var
            mean_only = True

        # ---------------- predict for two body -------------------
        f2 = vir2 = kern2 = v2 = e2 = 0
        if 2 in self.bodies:

            f2, vir2, kern2, v2, e2 = \
                self.predict_multicomponent(2, atom_env, self.kernel2b_info,
                                            self.spcs_set[0],
                                            self.maps_2, mean_only)

        # ---------------- predict for three body -------------------
        f3 = vir3 = kern3 = v3 = e3 = 0
        if 3 in self.bodies:

            f3, vir3, kern3, v3, e3 = \
                self.predict_multicomponent(3, atom_env, self.kernel3b_info,
                                            self.spcs[1], self.maps_3,
                                            mean_only)

        force = f2 + f3
        variance = kern2 + kern3 - np.sum((v2 + v3)**2, axis=0)
        virial = vir2 + vir3
        energy = e2 + e3

        return force, variance, virial, energy

    def predict_multicomponent(self, body, atom_env, kernel_info,
                               spcs_list, mappings, mean_only):
        '''
        Add up results from `predict_component` to get the total contribution
        of all species
        '''

        kernel, en_kernel, en_force_kernel, cutoffs, hyps, hyps_mask = \
            kernel_info

        args = from_mask_to_args(hyps, hyps_mask, cutoffs)

        kern = np.zeros(3)
        for d in range(3):
            kern[d] = \
                kernel(atom_env, atom_env, d+1, d+1, *args)

        if (body == 2):
            spcs, comp_r, comp_xyz = \
                get_bonds(atom_env.ctype, atom_env.etypes,
                          atom_env.bond_array_2)
            set_spcs = []
            for spc in spcs:
                set_spcs += [set(spc)]
            spcs = set_spcs
        elif (body == 3):
            spcs, comp_r, comp_xyz = \
                get_triplets_en(
                    atom_env.ctype, atom_env.etypes, atom_env.bond_array_3,
                    atom_env.cross_bond_inds, atom_env.cross_bond_dists,
                    atom_env.triplet_counts)

        # predict for each species
        f_spcs = 0
        vir_spcs = 0
        v_spcs = 0
        e_spcs = 0
        for i, spc in enumerate(spcs):
            lengths = np.array(comp_r[i])
            xyzs = np.array(comp_xyz[i])
            map_ind = spcs_list.index(spc)
            f, vir, v, e = \
                self.predict_component(lengths, xyzs, mappings[map_ind],
                                       mean_only)
            f_spcs += f
            vir_spcs += vir
            v_spcs += v
            e_spcs += e

        return f_spcs, vir_spcs, kern, v_spcs, e_spcs

    def predict_component(self, lengths, xyzs, mapping, mean_only):
        '''
        predict force and variance contribution of one component
        '''
        lengths = np.array(lengths)
        xyzs = np.array(xyzs)

        # predict mean
        e_0, f_0 = mapping.mean(lengths, with_derivatives=True)
        e = np.sum(e_0)  # energy

        # predict forces and stress
        vir = np.zeros(6)
        # match the ASE order
        vir_order = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

        # two-body
        if lengths.shape[-1] == 1:
            f_d = np.diag(f_0[:, 0, 0]) @ xyzs
            f = 2 * np.sum(f_d, axis=0)  # force: need to check prefactor 2

            for i in range(6):
                vir_i = f_d[:, vir_order[i][0]]\
                        * xyzs[:, vir_order[i][1]] * lengths[:, 0]
                vir[i] = np.sum(vir_i)

        # three-body
        if lengths.shape[-1] == 3:
            # factor1 = 1/lengths[:,1] - 1/lengths[:,0] * lengths[:,2]
            # factor2 = 1/lengths[:,0] - 1/lengths[:,1] * lengths[:,2]
            # f_d1 = np.diag(f_0[:,0,0]+f_0[:,2,0]*factor1) @ xyzs[:,0,:]
            # f_d2 = np.diag(f_0[:,1,0]+f_0[:,2,0]*factor2) @ xyzs[:,1,:]
            f_d1 = np.diag(f_0[:, 0, 0]) @ xyzs[:, 0, :]
            f_d2 = np.diag(f_0[:, 1, 0]) @ xyzs[:, 1, :]
            f_d = f_d1 + f_d2
            f = 3 * np.sum(f_d, axis=0)  # force: need to check prefactor 3

            for i in range(6):
                vir_i1 = f_d1[:, vir_order[i][0]]\
                       * xyzs[:, 0, vir_order[i][1]] * lengths[:, 0]
                vir_i2 = f_d2[:, vir_order[i][0]]\
                    * xyzs[:, 1, vir_order[i][1]] * lengths[:, 1]
                vir[i] = np.sum(vir_i1 + vir_i2)
            vir *= 1.5

        # predict var
        v = np.zeros(3)
        if not mean_only:
            v_0 = mapping.var(lengths)
            v_d = v_0 @ xyzs
            v = mapping.var.V @ v_d
        return f, vir, v, e

    def write_lmp_file(self, lammps_name):
        '''
        write the coefficients to a file that can be used by lammps pair style
        '''

        # write header
        f = open(lammps_name, 'w')

        header_comment = '''# #2bodyarray #3bodyarray\n# elem1 elem2 a b order
        '''
        f.write(header_comment)

        twobodyarray = len(self.spcs[0])
        threebodyarray = len(self.spcs[1])
        header = '\n{} {}\n'.format(twobodyarray, threebodyarray)
        f.write(header)

        # write two body
        if twobodyarray > 0:
            for ind, spc in enumerate(self.spcs[0]):
                self.maps_2[ind].write(f, spc)

        # write three body
        if threebodyarray > 0:
            for ind, spc in enumerate(self.spcs[1]):
                self.maps_3[ind].write(f, spc)

        f.close()

    def as_dict(self) -> dict:
        """
        Dictionary representation of the MGP model.
        """

        out_dict = deepcopy(dict(vars(self)))

        # Uncertainty mappings currently not serializable;
        if not self.mean_only:
            warnings.warn("Uncertainty mappings cannot be serialized, "
                          "and so the MGP dict outputted will not have "
                          "them.", Warning)
            out_dict['mean_only'] = True

        # Iterate through the mappings for various bodies
        for i in self.bodies:
            kern_info = f'kernel{i}b_info'
            kernel, ek, efk, cutoffs, hyps, hyps_mask = out_dict[kern_info]
            out_dict[kern_info] = (kernel.__name__, efk.__name__,
                                   cutoffs, hyps, hyps_mask)

        # only save the coefficients
        out_dict['maps_2'] = [map_2.mean.__coeffs__ for map_2 in self.maps_2]
        out_dict['maps_3'] = [map_3.mean.__coeffs__ for map_3 in self.maps_3]

        # don't need these since they are built in the __init__ function
        key_list = ['bond_struc', 'spcs_set', ]
        for key in key_list:
            if out_dict.get(key) is not None:
                del out_dict[key]

        return out_dict

    @staticmethod
    def from_dict(dictionary: dict):
        """
        Create MGP object from dictionary representation.
        """
        new_mgp = MappedGaussianProcess(
            grid_params=dictionary['grid_params'],
            struc_params=dictionary['struc_params'],
            GP=None, mean_only=dictionary['mean_only'], container_only=True,
            lmp_file_name=dictionary['lmp_file_name'],
            n_cpus=dictionary['n_cpus'], n_sample=dictionary['n_sample'],
            autorun=False)

        # Restore kernel_info
        for i in dictionary['bodies']:
            kern_info = f'kernel{i}b_info'
            hyps_mask = dictionary[kern_info][-1]
            if (hyps_mask is None):
                multihyps = False
            else:
                multihyps = True

            kernel_info = dictionary[kern_info]
            kernel_name = kernel_info[0]
            kernel, _, ek, efk = str_to_kernel_set(kernel_name, multihyps)
            kernel_info[0] = kernel
            kernel_info[1] = ek
            kernel_info[2] = efk
            setattr(new_mgp, kern_info, kernel_info)

        # Fill up the model with the saved coeffs
        for m, map_2 in enumerate(new_mgp.maps_2):
            map_2.mean.__coeffs__ = np.array(dictionary['maps_2'][m])
        for m, map_3 in enumerate(new_mgp.maps_3):
            map_3.mean.__coeffs__ = np.array(dictionary['maps_3'][m])

        # Set GP
        if dictionary.get('GP'):
            new_mgp.GP = GaussianProcess.from_dict(dictionary.get("GP"))

        return new_mgp

    def write_model(self, name: str, format='json'):
        """
        Write everything necessary to re-load and re-use the model
        :param model_name:
        :return:
        """
        if 'json' in format.lower():
            with open(f'{name}.json', 'w') as f:
                json.dump(self.as_dict(), f, cls=NumpyEncoder)

        elif 'pickle' in format.lower() or 'binary' in format.lower():
            with open(f'{name}.pickle', 'wb') as f:
                pickle.dump(self, f)

        else:
            raise ValueError("Requested format not found.")


#    @staticmethod
#    def load_model(elements: List[int], directory:str = './',
#                   kernel: str = '2+3',
#                   load_var: bool = False):
#        """
#        Loads the relevant files in a directory to an MGP model
#        :param elements:
#        :param directory:
#        :param kernel:
#        :return:
#        """
#
#        # Check to see if two body or three body kernels will be loaded
#        # so "2+3" makes b2 and b3 true
#        b2 = '2' in kernel or 'two' in kernel
#        b3 = '3' in kernel or 'three' in kernel
#
#
#        raise NotImplementedError

    @staticmethod
    def from_file(filename: str):
        if '.json' in filename:
            with open(filename, 'r') as f:
                model = \
                    MappedGaussianProcess.from_dict(json.loads(f.readline()))
            return model

        elif 'pickle' in filename:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise NotImplementedError


class Map2body:
    def __init__(self, grid_num: int, bounds, bond_struc: Structure,
                 svd_rank=0, mean_only: bool = False, n_cpus: int = None,
                 n_sample: int = 100):
        '''
        Build 2-body MGP

        bond_struc: Mock structure used to sample 2-body forces on 2 atoms
        '''

        self.grid_num = grid_num
        self.bounds = bounds
        self.bond_struc = bond_struc
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

        kernel_info = get_2bkernel(GP)

        if (self.n_cpus is None):
            processes = mp.cpu_count()
        else:
            processes = self.n_cpus

        # ------ construct grids ------
        nop = self.grid_num
        bond_lengths = np.linspace(self.bounds[0][0], self.bounds[1][0], nop)
        bond_means = np.zeros([nop])
        if not self.mean_only:
            bond_vars = np.zeros([nop, len(GP.alpha)])
        else:
            bond_vars = None
        env12 = AtomicEnvironment(self.bond_struc, 0, GP.cutoffs)

        # --------- calculate force kernels ---------------
        with mp.Pool(processes=processes) as pool:
            n_envs = len(GP.training_data)
            n_strucs = len(GP.training_structures)
            n_kern = n_envs * 3 + n_strucs

            block_id, nbatch = \
                partition_vector(self.n_sample, n_envs, processes)

            k12_slice = []
            k12_v_all = np.zeros([len(bond_lengths), n_kern])
            count = 0
            base = 0
            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                k12_slice.append(pool.apply_async(
                    self._GenGrid_inner, args=(GP.name, s, e, bond_lengths,
                                               env12, kernel_info)))
                count += 1

                # What is the point of this if statement?
                if (count > processes * 2):
                    for ibase in range(count):
                        s, e = block_id[ibase+base]
                        k12_v_all[:, s * 3:e * 3] = k12_slice[ibase].get()
                    del k12_slice
                    k12_slice = []
                    count = 0
                    base = ibatch+1

            if (count > 0):
                for ibase in range(count):
                    s, e = block_id[ibase+base]
                    k12_v_all[:, s * 3:e * 3] = k12_slice[ibase].get()
                del k12_slice

            pool.close()
            pool.join()

        # --------- calculate energy kernels ---------------
        with mp.Pool(processes=processes) as pool:
            block_id, nbatch = \
                partition_vector(self.n_sample, n_strucs, processes)

            k12_slice = []
            count = 0
            base = 0
            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                k12_slice.append(pool.apply_async(
                    self._GenGrid_energy,
                    args=(GP.name, s, e, bond_lengths, env12, kernel_info)))
                count += 1

                # What is the point of this if statement?
                if (count > processes * 2):
                    for ibase in range(count):
                        s, e = block_id[ibase+base]
                        k12_v_all[:, n_envs * 3 + s:n_envs * 3 + e] = \
                            k12_slice[ibase].get()
                    del k12_slice
                    k12_slice = []
                    count = 0
                    base = ibatch+1

            if (count > 0):
                for ibase in range(count):
                    s, e = block_id[ibase+base]
                    k12_v_all[:, n_envs * 3 + s:n_envs * 3 + e] = \
                        k12_slice[ibase].get()
                del k12_slice

            pool.close()
            pool.join()

        # ------- compute bond means and variances ---------------
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

        kernel, en_kernel, en_force_kernel, cutoffs, hyps, hyps_mask = \
            kernel_info
        size = e - s
        k12_v = np.zeros([len(bond_lengths), size*3])
        for b, r in enumerate(bond_lengths):
            env12.bond_array_2 = np.array([[r, 1, 0, 0]])
            k12_v[b, :] = \
                energy_force_vector_unit(name, s, e, env12, en_force_kernel,
                                         hyps, cutoffs, hyps_mask)
        return k12_v

    def _GenGrid_energy(self, name, s, e, bond_lengths, env12, kernel_info):

        '''
        Calculate kv segments of the given batch of training data for all grids
        '''

        kernel, en_kernel, en_force_kernel, cutoffs, hyps, hyps_mask = \
            kernel_info
        size = e - s
        k12_v = np.zeros([len(bond_lengths), size])
        for b, r in enumerate(bond_lengths):
            env12.bond_array_2 = np.array([[r, 1, 0, 0]])
            k12_v[b, :] = \
                energy_energy_vector_unit(name, s, e, env12, en_kernel,
                                          hyps, cutoffs, hyps_mask)
        return k12_v

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
        header_2 = '{elem1} {elem2} {a} {b} {order}\n'\
            .format(elem1=elem1, elem2=elem2, a=a, b=b, order=order)
        f.write(header_2)

        for c, coef in enumerate(coefs_2):
            f.write('{:.10e} '.format(coef))
            if c % 5 == 4 and c != len(coefs_2)-1:
                f.write('\n')

        f.write('\n')


class Map3body:

    def __init__(self, grid_num, bounds, bond_struc: Structure,
                 svd_rank: int = 0, mean_only: bool = False,
                 load_grid: str = '', update: bool = True, n_cpus=None,
                 n_sample=100):
        '''
        Build 3-body MGP

        bond_struc: Mock Structure object which contains 3 atoms to get map
        from
        '''
        self.grid_num = grid_num
        self.bounds = bounds
        self.bond_struc = bond_struc
        self.svd_rank = svd_rank
        self.mean_only = mean_only
        self.load_grid = load_grid
        self.update = update
        self.n_sample = n_sample

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
        kernel_info = get_3bkernel(GP)

        # ------ construct grids ------
        n1, n2, n12 = self.grid_num
        bonds1 = np.linspace(self.bounds[0][0], self.bounds[1][0], n1)
        bonds2 = np.linspace(self.bounds[0][0], self.bounds[1][0], n2)
        bonds12 = np.linspace(self.bounds[0][2], self.bounds[1][2], n12)
        grid_means = np.zeros([n1, n2, n12])

        if not self.mean_only:
            grid_vars = np.zeros([n1, n2, n12, len(GP.alpha)])
        else:
            grid_vars = None

        env12 = AtomicEnvironment(self.bond_struc, 0, GP.cutoffs)
        n_envs = len(GP.training_data)
        n_strucs = len(GP.training_structures)
        n_kern = n_envs * 3 + n_strucs

        if processes == 1:
            if self.update:
                raise NotImplementedError("the update function is "
                                          "not yet implemented")
            else:
                k12_v_all = \
                        np.zeros([len(bonds1), len(bonds2), len(bonds12),
                                  n_kern])
                k12_v_all[:, :, :, :n_envs * 3] = \
                    self._GenGrid_inner(GP.name, 0, n_envs, bonds1, bonds2,
                                        bonds12, env12, kernel_info)
                k12_v_all[:, :, :, n_envs*3:] = \
                    self._GenGrid_energy(GP.name, 0, n_strucs, bonds1, bonds2,
                                         bonds12, env12, kernel_info)
        else:

            # ------------ force kernels -------------
            with mp.Pool(processes=processes) as pool:

                if self.update:

                    raise NotImplementedError(
                        "the update function is "
                        "not yet implemented")

                else:
                    block_id, nbatch = \
                        partition_vector(self.n_sample, n_envs, processes)

                    k12_slice = []
                    # print('before for', ns, nsample, time.time())
                    count = 0
                    base = 0
                    k12_v_all = \
                        np.zeros([len(bonds1), len(bonds2), len(bonds12),
                                  n_envs * 3 + n_strucs])
                    for ibatch in range(nbatch):
                        s, e = block_id[ibatch]
                        k12_slice.append(pool.apply_async(
                            self._GenGrid_inner,
                            args=(GP.name, s, e, bonds1, bonds2, bonds12,
                                  env12, kernel_info)))
                        # print('send', ibatch, ns, s, e, time.time())
                        count += 1
                        if (count > processes*2):
                            for ibase in range(count):
                                s, e = block_id[ibase+base]
                                k12_v_all[:, :, :, s * 3:e * 3] = \
                                    k12_slice[ibase].get()
                            del k12_slice
                            k12_slice = []
                            count = 0
                            base = ibatch+1
                    if (count > 0):
                        for ibase in range(count):
                            s, e = block_id[ibase+base]
                            k12_v_all[:, :, :, s * 3:e * 3] = \
                                k12_slice[ibase].get()
                        del k12_slice

            # ------------ force kernels -------------
            with mp.Pool(processes=processes) as pool:

                if self.update:

                    raise NotImplementedError(
                        "the update function is "
                        "not yet implemented")

                else:
                    block_id, nbatch = \
                        partition_vector(self.n_sample, n_strucs, processes)

                    k12_slice = []
                    # print('before for', ns, nsample, time.time())
                    count = 0
                    base = 0
                    for ibatch in range(nbatch):
                        s, e = block_id[ibatch]
                        k12_slice.append(pool.apply_async(
                            self._GenGrid_energy,
                            args=(GP.name, s, e, bonds1, bonds2, bonds12,
                                  env12, kernel_info)))
                        # print('send', ibatch, ns, s, e, time.time())
                        count += 1
                        if (count > processes*2):
                            for ibase in range(count):
                                s, e = block_id[ibase+base]
                                k12_v_all[:, :, :,
                                          n_envs * 3 + s:n_envs * 3 + e] = \
                                    k12_slice[ibase].get()
                            del k12_slice
                            k12_slice = []
                            count = 0
                            base = ibatch+1
                    if (count > 0):
                        for ibase in range(count):
                            s, e = block_id[ibase+base]
                            k12_v_all[:, :, :,
                                      n_envs * 3 + s:n_envs * 3 + e] = \
                                k12_slice[ibase].get()
                        del k12_slice

                pool.close()
                pool.join()

        for b12 in range(len(bonds12)):
            for b1 in range(len(bonds1)):
                for b2 in range(len(bonds2)):
                    k12_v = k12_v_all[b1, b2, b12, :]
                    grid_means[b1, b2, b12] = np.matmul(k12_v, GP.alpha)
                    if not self.mean_only:
                        grid_vars[b1, b2, b12, :] = \
                            solve_triangular(GP.l_mat, k12_v, lower=True)

        # Construct file names according to current mapping

        # ------ save mean and var to file -------
        np.save('grid3_mean_'+self.species_code, grid_means)
        np.save('grid3_var_'+self.species_code, grid_vars)

        return grid_means, grid_vars

    def _GenGrid_inner(self, name, s, e, bonds1, bonds2, bonds12, env12,
                       kernel_info):

        '''
        Calculate kv segments of the given batch of training data for all grids
        '''

        kernel, en_kernel, en_force_kernel, cutoffs, hyps, hyps_mask = \
            kernel_info

        # open saved k vector file, and write to new file
        size = (e - s) * 3
        k12_v = np.zeros([len(bonds1), len(bonds2), len(bonds12), size])
        for b12, r12 in enumerate(bonds12):
            for b1, r1 in enumerate(bonds1):
                for b2, r2 in enumerate(bonds2):

                    env12.bond_array_3 = np.array([[r1, 1, 0, 0],
                                                   [r2, 0, 0, 0]])
                    env12.cross_bond_dists = np.array([[0, r12], [r12, 0]])
                    k12_v[b1, b2, b12, :] = \
                        energy_force_vector_unit(
                            name, s, e, env12, en_force_kernel, hyps, cutoffs,
                            hyps_mask)

        # open saved k vector file, and write to new file
        if self.update:

            raise NotImplementedError("the update function is not yet"
                                      "implemented")

            # s, e = block
            # chunk = e - s
            # new_kv_file = \
            #     np.zeros((chunk, self.grid_num[0] * self.grid_num[1] + 1,
            #               total_size))
            # new_kv_file[:,0,0] = np.ones(chunk) * total_size
            # for i in range(s, e):
            #     kv_filename = f'{self.kv3name}/{i}'
            #     if kv_filename in os.listdir(self.kv3name):
            #         old_kv_file = np.load(kv_filename+'.npy')
            #         last_size = int(old_kv_file[0,0])
            #         new_kv_file[i, :, :last_size] = old_kv_file
            #     else:
            #         last_size = 0
            # ds = [1, 2, 3]
            # nop = self.grid_num[0]

            # k12_v = new_kv_file[:, 1:, :]
            # for i in range(s, e):
            #     np.save(f'{self.kv3name}/{i}', new_kv_file[i, :, :])

        return k12_v

    def _GenGrid_energy(self, name, s, e, bonds1, bonds2, bonds12, env12,
                        kernel_info):

        '''
        Calculate kv segments of the given batch of training data for all grids
        '''

        kernel, en_kernel, en_force_kernel, cutoffs, hyps, hyps_mask = \
            kernel_info

        # open saved k vector file, and write to new file
        size = e - s
        k12_v = np.zeros([len(bonds1), len(bonds2), len(bonds12), size])
        for b12, r12 in enumerate(bonds12):
            for b1, r1 in enumerate(bonds1):
                for b2, r2 in enumerate(bonds2):

                    env12.bond_array_3 = np.array([[r1, 1, 0, 0],
                                                   [r2, 0, 0, 0]])
                    env12.cross_bond_dists = np.array([[0, r12], [r12, 0]])
                    k12_v[b1, b2, b12, :] = \
                        energy_energy_vector_unit(
                            name, s, e, env12, en_kernel, hyps, cutoffs,
                            hyps_mask)

        # open saved k vector file, and write to new file
        if self.update:
            raise NotImplementedError("the update function is not yet"
                                      "implemented")

        return k12_v

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

import time
import math
import numpy as np
from scipy.linalg import solve_triangular
import multiprocessing as mp
import subprocess
import os
from copy import deepcopy

from flare import gp, struc, kernels
from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.cutoffs import quadratic_cutoff
import flare.mgp.utils as utils
from flare.mgp.utils import get_bonds, get_triplets, self_two_body_mc_jit, \
    self_three_body_mc_jit, \
    get_2bkernel, get_3bkernel
from flare.mgp.splines_methods import PCASplines, CubicSpline

import flare.cutoffs as cf
from flare.kernels import str_to_kernel
from flare.mc_simple import str_to_mc_kernel
from flare.mc_sephyps import str_to_mc_kernel as str_to_mc_sephyps_kernel



class MappedGaussianProcess:
    '''
    Build Mapped Gaussian Process (MGP) and automatically save coefficients for LAMMPS pair style.
    :param: hyps: GP hyps
    :param: cutoffs: GP cutoffs
    :param: struc_params : information of training data
    :param: grid_params : setting of grids for mapping
    :param: mean_only : if True: only build mapping for mean (force)
    :param: container_only : if True: only build splines container (with no coefficients)
    :param: GP: None or a GaussianProcess object. If input a GP, then build mapping when creating MappedGaussianProcess object
    :param: lmp_file_name : lammps coefficient file name
    Examples:

    >>> struc_params = {'species': [0, 1],
                        'cube_lat': cell, # should input the cell matrix
                        'mass_dict': {'0': 27 * unit, '1': 16 * unit}}
    >>> grid_params =  {'bounds_2': [[1.2], [3.5]],
                                    # [[lower_bound], [upper_bound]]
                        'bounds_3': [[1.2, 1.2, 0], [3.5, 3.5, np.pi]],
                                    # [[lower,lower,0],[upper,upper,np.pi]]
                        'grid_num_2': 64,
                        'grid_num_3': [16, 16, 16],
                        'svd_rank_2': 64,
                        'svd_rank_3': 16**3,
                        'bodies': [2, 3],
                        'update': True, # if True: accelerating grids
                                        # generating by saving intermediate
                                        # coeff when generating grids
                        'load_grid': None}
    '''

    def __init__(self, hyps, cutoffs, grid_params: dict, struc_params: dict,
                 mean_only=False, container_only=True,
                 bond_struc=None, spcs=None,
                 GP=None,
                 lmp_file_name='lmp.mgp', verbose=1, ncpus=None, nsample=100):

        self.hyps = hyps
        self.cutoffs = cutoffs
        self.grid_params = grid_params
        self.struc_params = struc_params
        self.grid_num_2 = grid_params['grid_num_2']
        self.bounds_2 = grid_params['bounds_2']
        self.grid_num_3 = grid_params['grid_num_3']
        self.bounds_3 = grid_params['bounds_3']
        self.bodies = grid_params['bodies']

        self.svd_rank_2 = grid_params['svd_rank_2']
        self.svd_rank_3 = grid_params['svd_rank_3']
        self.update = grid_params['update']
        self.mean_only = mean_only
        self.lmp_file_name = lmp_file_name
        self.kernel_name = "two_plus_three_mc"

        self.multihyps = False
        if (GP is not None):
            self.kernel_name = GP.kernel_name
            self.multihyps = GP.multihyps
            hyps = deepcopy(GP.hyps)
            if (self.multihyps is True):
                self.hyps_mask = deepcopy(GP.hyps_mask)
                if ('map' in self.hyps_mask.keys()):
                    ori_hyps = deepcopy(self.hyps_mask['original'])
                    hm = self.hyps_mask['map']
                    for i, h in enumerate(hyps):
                        ori_hyps[hm[i]]=h
                    self.hyps_mask.pop('map')
                    self.hyps_mask.pop('original')
                else:
                    ori_hyps = hyps
                self.hyps = ori_hyps

        self.GP = GaussianProcess(GP.kernel,GP.kernel_grad, GP.hyps,
                 GP.cutoffs, GP.hyp_labels,
                 GP.energy_force_kernel,
                 GP.energy_kernel,
                 GP.multihyps, GP.hyps_mask)

        self.verbose = verbose
        self.ncpus = ncpus
        self.nsample = nsample

        if (bond_struc is None or spcs is None):
            self.build_bond_struc()
        else:
            self.bond_struc = bond_struc
            self.spcs = spcs

        self.maps_2 = []
        self.maps_3 = []
        self.build_map_container()

        if (not container_only) and (GP is not None) and (len(GP.training_data) > 0):
            self.build_map(GP)

        if 'mc' in self.kernel_name:
            if (self.multihyps is True):
                self.kernel_2b = str_to_mc_sephyps_kernel('two_body_mc')
                self.kernel_3b = str_to_mc_sephyps_kernel('three_body_mc')
            else:
                self.kernel_2b = str_to_mc_kernel('two_body_mc')
                self.kernel_3b = str_to_mc_kernel('three_body_mc')
        else:
            self.kernel_2b = str_to_kernel('two_body')
            self.kernel_3b = str_to_kernel('three_body')

    def build_map_container(self):
        '''
        construct an empty spline container without coefficients
        '''

        if 2 in self.bodies:
            for b_struc in self.bond_struc[0]:
               map_2 = Map2body(self.grid_num_2, self.bounds_2,
                                self.cutoffs,
                                b_struc, self.bodies, self.svd_rank_2,
                                self.mean_only, self.ncpus, self.nsample)
               self.maps_2.append(map_2)
        if 3 in self.bodies:
            for b_struc in self.bond_struc[1]:
                map_3 = Map3body(self.grid_num_3, self.bounds_3, self.cutoffs,
                                 b_struc, self.bodies, self.svd_rank_3,
                                 self.mean_only,
                                 self.grid_params['load_grid'],
                                 self.update, self.ncpus, self.nsample)
                self.maps_3.append(map_3)
        print("self.maps_3")

    def build_map(self, GP):
        '''
        generate/load grids and get spline coefficients
        '''
        for map_2 in self.maps_2:
            map_2.build_map(GP)
        for map_3 in self.maps_3:
            map_3.build_map(GP)

        # write to lammps pair style coefficient file
        self.write_lmp_file(self.lmp_file_name)

    def build_bond_struc(self):

        '''
        build a bond structure, used in grid generating
        '''

        if self.GP is not None:
           self.bodies = []
           if 'two' in self.GP.kernel_name:
               self.bodies += [2]
           if 'three' in self.GP.kernel_name:
               self.bodies += [3]
        else:
            self.bodies = self.grid_params['bodies']
        print("bodies", self.bodies)

        if (self.GP is not None):
            cutoff = np.min(self.GP.cutoffs)
            # structure = self.GP.training_data[0].structure
            # cell = structure.cell
            # mass_dict = structure.mass_dict
            # species_list = list(set(structure.coded_species))
        else:
            cutoff = np.min(self.cutoffs)

        cell = self.struc_params['cube_lat']
        mass_dict = self.struc_params['mass_dict']
        species_list = self.struc_params['species']
        N_spc = len(species_list)

        # ------------------- 2 body (2 atoms (1 bond) config) ---------------
        bond_struc_2 = []
        spc_2 = []
        if (2 in self.bodies):
            bodies = 2
            sorted_species = sorted(species_list)
            for spc1_ind, spc1 in enumerate(sorted_species):
                for spc2 in sorted_species[spc1_ind:]:
                    species = [spc1, spc2]
                    spc_2.append(species)
                    positions = [[(i+1)/(bodies+1)*cutoff, 0, 0]
                                 for i in range(bodies)]
                    spc_struc = \
                        struc.Structure(cell, species, positions, mass_dict)
                    spc_struc.coded_species = np.array(species)
                    bond_struc_2.append(spc_struc)

        # ------------------- 3 body (3 atoms (1 triplet) config) -------------
        bond_struc_3 = []
        spc_3 = []
        if (3 in self.bodies):
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
                        spc_struc = struc.Structure(cell, species, positions,
                                                    mass_dict)
                        spc_struc.coded_species = np.array(species)
                        bond_struc_3.append(spc_struc)
#                        if spc1 != spc2:
#                            species = [spc2, spc3, spc1]
#                            spc_3.append(species)
#                            positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                        for i in range(bodies)]
#                            spc_struc = struc.Structure(cell, species, positions,
#                                                        mass_dict)
#                            spc_struc.coded_species = np.array(species)
#                            bond_struc_3.append(spc_struc)
#                        if spc2 != spc3:
#                            species = [spc3, spc1, spc2]
#                            spc_3.append(species)
#                            positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                        for i in range(bodies)]
#                            spc_struc = struc.Structure(cell, species, positions,
#                                                        mass_dict)
#                            spc_struc.coded_species = np.array(species)
#                            bond_struc_3.append(spc_struc)

        self.bond_struc = [bond_struc_2, bond_struc_3]
        self.spcs = [spc_2, spc_3]

    def predict(self, atom_env: AtomicEnvironment, mean_only: bool=False):
        '''
        predict force and variance for given atomic environment
        :param atom_env: atomic environment (with a center atom and its neighbors)
        :param mean_only: if True: only predict force (variance is always 0)
        '''
        if self.mean_only:  # if not build mapping for var
            mean_only = True

        # ---------------- predict for two body -------------------
        f2 = kern2 = v2 = 0
        if 2 in self.bodies:

            f2, kern2, v2 = \
                self.predict_multicomponent(atom_env,
                                            self.get_2body_comp, self.spcs[0],
                                            self.maps_2, mean_only)

        # ---------------- predict for three body -------------------
        f3 = kern3 = v3 = 0
        if 3 in self.bodies:

            f3, kern3, v3 = \
                self.predict_multicomponent(atom_env,
                                            self.get_3body_comp, self.spcs[1],
                                            self.maps_3, mean_only)

        f = f2 + f3
        v = kern2 + kern3 - np.sum((v2 + v3)**2, axis=0)
        return f, v

    def get_2body_comp(self, atom_env):
        '''
        get bonds grouped by species
        '''
        # bond_array_2 = atom_env.bond_array_2
        # ctype = atom_env.ctype
        # etypes = atom_env.etypes

        kern2 = np.zeros(3)
        for d in range(3):
            if (self.multihyps is True):
                kern2[d] = self.kernel_2b(atom_env, atom_env, d+1, d+1,
                                          self.hyps, self.cutoffs,
                                          hyps_mask=self.hyps_mask)
            else:
                kern2[d] = self.kernel_2b(atom_env, atom_env, d+1, d+1,
                                          self.hyps[:2], self.cutoffs)
            # kern2[d] = \
            #     self_two_body_mc_jit(bond_array_2, ctype, etypes, d+1, sig, ls,
            #                          r_cut, quadratic_cutoff)

        bond_array_2 = atom_env.bond_array_2
        ctype = atom_env.ctype
        etypes = atom_env.etypes
        spcs, comp_r, comp_xyz = get_bonds(ctype, etypes, bond_array_2)
        return kern2, spcs, comp_r, comp_xyz

    def get_3body_comp(self, atom_env):
        '''
        get triplets and grouped by species
        '''
        bond_array_3 = atom_env.bond_array_3
        cross_bond_inds = atom_env.cross_bond_inds
        cross_bond_dists = atom_env.cross_bond_dists
        triplets = atom_env.triplet_counts
        ctype = atom_env.ctype
        etypes = atom_env.etypes

#        kern3 = np.zeros(3)
#        for d in range(3):
#            kern3[d] = self_three_body_mc_jit(bond_array_3, cross_bond_inds,
#                    cross_bond_dists, triplets, ctype, etypes, d+1, sig, ls,
#                    r_cut, quadratic_cutoff)

        kern3 = np.zeros(3)
        for d in range(3):
            if (self.multihyps is True):
                kern3[d] = self.kernel_3b(atom_env, atom_env, d+1, d+1,
                                          self.hyps, self.cutoffs,
                                          hyps_mask=self.hyps_mask)
            else:
                kern3[d] = self.kernel_3b(atom_env, atom_env, d+1, d+1,
                                          self.hyps[-3:-1], self.cutoffs)

        spcs, comp_r, comp_xyz = \
            get_triplets(ctype, etypes, bond_array_3,
                         cross_bond_inds, cross_bond_dists, triplets)
        return kern3, spcs, comp_r, comp_xyz

    def predict_multicomponent(self, atom_env, get_comp,
                               spcs_list, mappings, mean_only):
        '''
        Add up results from `predict_component` to get the total contribution
        of all species
        '''
        f_spcs = 0
        v_spcs = 0

        kern, spcs, comp_r, comp_xyz = get_comp(atom_env)

        # predict for each species
        for i, spc in enumerate(spcs):
            lengths = np.array(comp_r[i])
            xyzs = np.array(comp_xyz[i])
            map_ind = spcs_list.index(spc)
            f, v = self.predict_component(lengths, xyzs, mappings[map_ind],
                                          mean_only)
            print("test", spc, f)
            f_spcs += f
            v_spcs += v

        return f_spcs, kern, v_spcs

    def predict_component(self, lengths, xyzs, mapping, mean_only):
        '''
        predict force and variance contribution of one component
        '''
        lengths = np.array(lengths)
        xyzs = np.array(xyzs)

        # predict mean
        f_0 = mapping.mean(lengths)
        f_d = np.diag(f_0) @ xyzs
        f = np.sum(f_d, axis=0)

        # predict var
        v = np.zeros(3)
        if not mean_only:
            v_0 = mapping.var(lengths)
            v_d = v_0 @ xyzs
            v = mapping.var.V @ v_d
        return f, v

    def write_two_body(self, f):
        a = self.bounds_2[0][0]
        b = self.bounds_2[1][0]
        order = self.grid_num_2

        for ind, spc in enumerate(self.spcs[0]):
            coefs_2 = self.maps_2[ind].mean.__coeffs__

            elem1 = spc[0]
            elem2 = spc[1]
            header_2 = '{elem1} {elem2} {a} {b} {order}\n'\
                .format(elem1=elem1, elem2=elem2, a=a, b=b, order=order)
            f.write(header_2)

            for c, coef in enumerate(coefs_2):
                f.write('{:.10e} '.format(coef))
                if c % 5 == 4 and c != len(coefs_2)-1:
                    f.write('\n')

            f.write('\n')

    def write_three_body(self, f):
        a = self.bounds_3[0]
        b = self.bounds_3[1]
        order = self.grid_num_3

        for ind, spc in enumerate(self.spcs[1]):
            coefs_3 = self.maps_3[ind].mean.__coeffs__

            elem1 = spc[0]
            elem2 = spc[1]
            elem3 = spc[2]

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


    def write_lmp_file(self, lammps_name):
        '''
        write the coefficients to a file that can be used by lammps pair style
        '''
        if (self.verbose > 2):
            print("write mapped force field")

        # write header
        f = open(lammps_name, 'w')

        header_comment = '''# #2bodyarray #3bodyarray
        # elem1 elem2 a b order
        '''
        f.write(header_comment)

        twobodyarray = len(self.spcs[0])
        threebodyarray = len(self.spcs[1])
        header = '\n{} {}\n'.format(twobodyarray, threebodyarray)
        f.write(header)

        # write two body
        if twobodyarray > 0:
            self.write_two_body(f)

        # write three body
        if threebodyarray > 0:
            self.write_three_body(f)

        f.close()


class Map2body:
    def __init__(self, grid_num, bounds, cutoffs, bond_struc, bodies='2',
                 svd_rank=0, mean_only=False, ncpus=None, nsample=100):
        '''
        Build 2-body MGP
        '''

        self.grid_num = grid_num
        self.l_bounds, self.u_bounds = bounds
        self.cutoffs = cutoffs
        self.bond_struc = bond_struc
        self.species = bond_struc.coded_species
        self.bodies = bodies
        self.svd_rank = svd_rank
        self.mean_only = mean_only
        self.ncpus = ncpus
        self.nsample = nsample

        self.build_map_container()

    def GenGrid(self, GP):

        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''

        kernel_info = get_2bkernel(GP)

        if (self.ncpus is None):
            processes = mp.cpu_count()
        else:
            processes = self.ncpus

        # ------ construct grids ------
        nop = self.grid_num
        bond_lengths = np.linspace(self.l_bounds[0], self.u_bounds[0], nop)
        bond_means = np.zeros([nop])
        bond_vars = np.zeros([nop, len(GP.alpha)])
        env12 = AtomicEnvironment(self.bond_struc, 0, self.cutoffs)

        with mp.Pool(processes=processes) as pool:
            # A_list = pool.map(self._GenGrid_inner_most, pool_list)
            # break it into pieces
            size = len(GP.training_data)
            nsample = self.nsample
            ns = int(math.ceil(size/nsample))
            if (ns < processes):
                nsample = int(math.ceil(size/processes))
                ns = int(math.ceil(size/nsample))

            print("prepare the package for parallelization")
            block_id = []
            nbatch = 0
            for ibatch in range(ns):
                s1 = int(nsample*ibatch)
                e1 = int(np.min([s1 + nsample, size]))
                block_id += [(s1, e1)]
                print("block", ibatch, s1, e1)
                nbatch += 1

            k12_slice = []
            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                k12_slice.append(pool.apply_async(self._GenGrid_inner,
                                                  args=(GP.training_data[s:e],
                                                        bond_lengths,
                                                        env12, kernel_info)))
            k12_v_all = np.zeros([len(bond_lengths), size*3])
            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                k12_v_all[:, s*3:e*3] = k12_slice[ibatch].get()
            pool.close()
            pool.join()

        for b, r in enumerate(bond_lengths):
            k12_v = k12_v_all[b, :]
            bond_means[b] = np.matmul(k12_v, GP.alpha)
            if not self.mean_only:
                bond_vars[b, :] = solve_triangular(GP.l_mat, k12_v, lower=True)

        return bond_means, bond_vars

    def _GenGrid_inner(self, training_data, bond_lengths,
            env12, kernel_info):

        '''
        generate grid for each angle, used to parallelize grid generation
        '''

        kernel, cutoffs, hyps, hyps_mask = kernel_info
        size = len(training_data)
        k12_v = np.zeros([len(bond_lengths), size*3])
        for b, r in enumerate(bond_lengths):
            env12.bond_array_2 = np.array([[r, 1, 0, 0]])
            k12_v[b, :] = get_kernel_vector(training_data,
                                            env12, 1,
                                            kernel, hyps,
                                            cutoffs, hyps_mask)
        return k12_v

    def build_map_container(self):

        '''
        build 1-d spline function for mean, 2-d for var
        '''
        self.mean = CubicSpline(self.l_bounds, self.u_bounds,
                                orders=[self.grid_num])

        if not self.mean_only:
            self.var = PCASplines(self.l_bounds, self.u_bounds,
                                  orders=[self.grid_num],
                                  svd_rank=self.svd_rank)

    def build_map(self, GP):
        y_mean, y_var = self.GenGrid(GP)
        self.mean.set_values(y_mean)
        if not self.mean_only:
            self.var.set_values(y_var)



class Map3body:

    def __init__(self, grid_num, bounds, cutoffs, bond_struc, bodies='3',
            svd_rank=0, mean_only=False, load_grid=None, update=True,
            ncpus=None, nsample=100):
        '''
        Build 3-body MGP
        '''

        self.grid_num = grid_num
        self.l_bounds, self.u_bounds = bounds
        self.cutoffs = cutoffs
        self.bond_struc = bond_struc
        self.species = bond_struc.coded_species
        self.bodies = bodies
        self.svd_rank = svd_rank
        self.mean_only = mean_only
        self.load_grid = load_grid
        self.update = update
        self.ncpus = ncpus
        self.nsample = nsample

        self.build_map_container()

        self.kv3name = f'kv3{int(time.time())}'


    def GenGrid(self, GP):
        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''

        if (self.ncpus is None):
            processes = mp.cpu_count()
        else:
            processes = self.ncpus
        if processes == 1:
            self.GenGrid_serial(GP)

        # ------ get 3body kernel info ------
        kernel_info = get_3bkernel(GP)

        # ------ construct grids ------
        nop = self.grid_num[0]
        noa = self.grid_num[2]
        bond_lengths = np.linspace(self.l_bounds[0], self.u_bounds[0], nop)
        angles = np.linspace(self.l_bounds[2], self.u_bounds[2], noa)

        bond_means = np.zeros([nop, nop, noa])
        bond_vars = np.zeros([nop, nop, noa, len(GP.alpha)])
        env12 = AtomicEnvironment(self.bond_struc, 0, self.cutoffs)

        with mp.Pool(processes=processes) as pool:
            if self.update:
                if self.kv3name in os.listdir():
                    subprocess.run(['rm', '-rf', self.kv3name])
                subprocess.run(['mkdir', self.kv3name])

            print("prepare the package for parallelization")
            size = len(GP.training_data)
            nsample = self.nsample
            ns = int(math.ceil(size/nsample))
            if (ns < processes):
                nsample = int(math.ceil(size/processes))
                ns = int(math.ceil(size/nsample))

            k12_slice = []
            for ibatch in range(ns):
                s = nsample*ibatch
                e = np.min([s + nsample, size])
                k12_slice.append(pool.apply_async(self._GenGrid_inner_most,
                                                  args=(GP.training_data[s:e],
                                                        angles, bond_lengths,
                                                        env12, kernel_info)))
                print('send', ibatch, ns, s, e, time.time())
            pool.close()
            pool.join()

            size3 = size*3
            nsample3 = nsample*3
            k12_v_all = np.zeros([len(bond_lengths), len(bond_lengths), len(angles), size3])
            for ibatch in range(ns):
                s = nsample3*ibatch
                e = np.min([s + nsample3, size3])
                k12_v_all[:, :, :, s:e] = k12_slice[ibatch].get()
                print('get', ibatch, ns, time.time())


        for a12, angle in enumerate(angles):
            for b1, r1 in enumerate(bond_lengths):
                for b2, r2 in enumerate(bond_lengths):
                    k12_v = k12_v_all[b1, b2, a12, :]
                    bond_means[b1, b2, a12] = np.matmul(k12_v, GP.alpha)
                    if not self.mean_only:
                        bond_vars[b1, b2, a12, :] = solve_triangular(GP.l_mat, k12_v, lower=True)


        # # ------ save mean and var to file -------
        # np.save('grid3_mean', bond_means)
        # np.save('grid3_var', bond_vars)

        return bond_means, bond_vars

    def GenGrid_serial(self, GP):
        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''
        print("running serial version")

        # ------ get 3body kernel info ------
        kernel_info = get_3bkernel(GP)
        kernel, cutoffs, hyps, hyps_mask = kernel_info

        # ------ construct grids ------
        nop = self.grid_num[0]
        noa = self.grid_num[2]
        bond_lengths = np.linspace(self.l_bounds[0], self.u_bounds[0], nop)
        angles = np.linspace(self.l_bounds[2], self.u_bounds[2], noa)
        bond_means = np.zeros([nop, nop, noa])
        bond_vars = np.zeros([nop, nop, noa, len(GP.alpha)])
        env12 = AtomicEnvironment(self.bond_struc, 0, self.cutoffs)

        if self.update:
            if self.kv3name in os.listdir():
                subprocess.run(['rm', '-rf', self.kv3name])
            subprocess.run(['mkdir', self.kv3name])

        size = len(GP.training_data)
        ds = [1, 2, 3]
        k_v = np.zeros(3)
        k12_v_all = np.zeros([len(bond_lengths), len(bond_lengths), len(angles), size*3])
        for b1, r1 in enumerate(bond_lengths):
            for b2, r2 in enumerate(bond_lengths):
                for a12, angle12 in enumerate(angles):

                    x2 = r2 * np.cos(angle12)
                    y2 = r2 * np.sin(angle12)
                    r12 = np.linalg.norm(np.array([x2-r1, y2, 0]))

                    env12.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])
                    env12.cross_bond_dists = np.array([[0, r12], [r12, 0]])

                    for isample, sample in enumerate(GP.training_data):
                        if (hyps_mask is not None):
                            for d in ds:
                                k_v[d-1] = kernel(env12, sample, 1, d,
                                                  hyps, cutoffs,
                                                  hyps_mask=hyps_mask)

                        else:
                            for d in ds:
                                k_v[d-1] = kernel(env12, sample, 1, d,
                                                  hyps, cutoffs)

                        k12_v_all[b1, b2, a12, isample*3:isample*3+3] = k_v

        for b1, r1 in enumerate(bond_lengths):
            for b2, r2 in enumerate(bond_lengths):
                for a12, angle in enumerate(angles):
                    k12_v = k12_v_all[b1, b2, a12, :]
                    bond_means[b1, b2, a12] = np.matmul(k12_v, GP.alpha)
                    if not self.mean_only:
                        bond_vars[b1, b2, a12, :] = solve_triangular(GP.l_mat, k12_v, lower=True)

        # # ------ save mean and var to file -------
        # np.save('grid3_mean', bond_means)
        # np.save('grid3_var', bond_vars)

        return bond_means, bond_vars

    def _GenGrid_inner_most(self, training_data, angles, bond_lengths, env12, kernel_info):

        '''
        generate grid for each angle, used to parallelize grid generation
        '''

        kernel, cutoffs, hyps, hyps_mask = kernel_info
        # open saved k vector file, and write to new file
        size = len(training_data)*3
        k12_v = np.zeros([len(angles), len(bond_lengths), len(bond_lengths), size])
        for a12, angle12 in enumerate(angles):
            for b1, r1 in enumerate(bond_lengths):
                for b2, r2 in enumerate(bond_lengths):

                    x2 = r2 * np.cos(angle12)
                    y2 = r2 * np.sin(angle12)
                    r12 = np.linalg.norm(np.array([x2-r1, y2, 0]))

                    env12.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])
                    env12.cross_bond_dists = np.array([[0, r12], [r12, 0]])

                    k12_v[b1, b2, a12, :] = get_kernel_vector(training_data,
                                                              env12, 1,
                                                              kernel, hyps,
                                                              cutoffs, hyps_mask)

        return k12_v


    def build_map_container(self):

        '''
        build 3-d spline function for mean,
        3-d for the low rank approximation of L^{-1}k*
        '''

       # create spline interpolation class object
        nop = self.grid_num[0]
        noa = self.grid_num[2]
        self.mean = CubicSpline(self.l_bounds, self.u_bounds,
                                orders=[nop, nop, noa])

        if not self.mean_only:
            self.var = PCASplines(self.l_bounds, self.u_bounds,
                                  orders=[nop, nop, noa],
                                  svd_rank=self.svd_rank)

    def build_map(self, GP):
        # Load grid or generate grid values
        if not self.load_grid:
            y_mean, y_var = self.GenGrid(GP)
        else:
            y_mean = np.load('grid3_mean.npy')
            y_var = np.load('grid3_var.npy')

        self.mean.set_values(y_mean)
        if not self.mean_only:
            self.var.set_values(y_var)

def get_kernel_vector(training_data, x: AtomicEnvironment,
                      d_1: int, kernel, hyps, cutoffs,
                      hyps_mask=None):

    ds = [1, 2, 3]
    size = len(training_data) * 3
    k_v = np.zeros(size, )

    if (hyps_mask is not None):
        for m_index in range(size):
            x_2 = training_data[int(math.floor(m_index / 3))]
            d_2 = ds[m_index % 3]
            k_v[m_index] = kernel(x, x_2, d_1, d_2,
                                  hyps, cutoffs,
                                  hyps_mask=hyps_mask)

    else:
        for m_index in range(size):
            x_2 = training_data[int(math.floor(m_index / 3))]
            d_2 = ds[m_index % 3]
            k_v[m_index] = kernel(x, x_2, d_1, d_2,
                                  hyps, cutoffs)
    return k_v

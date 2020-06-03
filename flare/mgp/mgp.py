'''
:class:`MappedGaussianProcess` uses splines to build up interpolation\
function of the low-dimensional decomposition of Gaussian Process, \
with little loss of accuracy. Refer to \
`Vandermause et al. <https://www.nature.com/articles/s41524-020-0283-z>`_, \
`Glielmo et al. <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.184307>`_
'''
import time
import math
import numpy as np
from scipy.linalg import solve_triangular
import multiprocessing as mp
import subprocess
import os

from flare import gp, struc, gp_algebra
from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.gp_algebra import force_force_vector_unit, partition_vector
from flare.cutoffs import quadratic_cutoff
from flare.kernels.mc_simple import two_body_mc, three_body_mc
from flare.util import Z_to_element
import flare.mgp.utils as utils
from flare.mgp.utils import get_bonds, get_triplets, self_two_body_mc_jit, \
    self_three_body_mc_jit, \
    get_2bkernel, get_3bkernel
from flare.mgp.splines_methods import PCASplines, CubicSpline


class MappedGaussianProcess:
    '''
    Build Mapped Gaussian Process (MGP) and automatically save coefficients for\
    LAMMPS pair style.

    Args:
        hyps (numpy array): GP hyps.
        cutoffs (numpy array): GP cutoffs.
        struc_params (dict): information of training data.
        grid_params (dict): setting of grids for mapping.
        mean_only (Bool): if True: only build mapping for mean (force).
        container_only (Bool): if True: only build splines container\
            (with no coefficients).
        GP (GaussianProcess): None or a GaussianProcess object. If input a GP,\
            then build mapping when creating MappedGaussianProcess object.
        lmp_file_name (str): lammps coefficient file name.

    Example:
        >>> struc_params = {'species': [0, 1],
                            'cube_lat': cell} # input the cell matrix
        >>> grid_params =  {'bounds_2': [[1.2], [3.5]],
                                        # [[lower_bound], [upper_bound]]
                            'bounds_3': [[1.2, 1.2, -1], [3.5, 3.5, 1]],
                                        # [[lower,lower,cos(pi)],[upper,upper,cos(0)]]
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
                 mean_only=False, container_only=True, GP=None,
                 lmp_file_name='lmp.mgp', n_cpus=None, n_sample=100):

        self.hyps = hyps
        self.cutoffs = cutoffs
        self.grid_params = grid_params
        self.struc_params = struc_params
        self.bodies = grid_params['bodies']
        self.grid_num_2 = grid_params['grid_num_2']
        self.bounds_2 = grid_params['bounds_2']
        self.grid_num_3 = grid_params['grid_num_3']
        self.bounds_3 = grid_params['bounds_3']

        self.svd_rank_2 = grid_params['svd_rank_2']
        self.svd_rank_3 = grid_params['svd_rank_3']
        self.update = grid_params['update']
        self.mean_only = mean_only
        self.lmp_file_name = lmp_file_name
        self.n_cpus = n_cpus
        self.n_sample = n_sample

        self.build_bond_struc(struc_params)
        self.maps_2 = []
        self.maps_3 = []
        self.build_map_container()

        if ((not container_only) and (GP is not None) and (len(GP.training_labels_np) > 0)):
            self.build_map(GP)

    def build_map_container(self):
        '''
        construct an empty spline container without coefficients
        '''
        if 2 in self.bodies:
            for b_struc in self.bond_struc[0]:
                map_2 = Map2body(self.grid_num_2, self.bounds_2, self.cutoffs,
                                 b_struc, self.bodies, self.svd_rank_2,
                                 self.mean_only, self.n_cpus, self.n_sample)
                self.maps_2.append(map_2)
        if 3 in self.bodies:
            for b_struc in self.bond_struc[1]:
                map_3 = Map3body(self.grid_num_3, self.bounds_3, self.cutoffs,
                                 b_struc, self.bodies, self.svd_rank_3,
                                 self.mean_only,
                                 self.grid_params['load_grid'],
                                 self.update,
                                 self.n_cpus, self.n_sample)
                self.maps_3.append(map_3)

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

    def build_bond_struc(self, struc_params):

        '''
        build a bond structure, used in grid generating
        '''

        cutoff = np.min(self.cutoffs)
        cell = struc_params['cube_lat']
        species_list = struc_params['species']
        N_spc = len(species_list)

        # ------------------- 2 body (2 atoms (1 bond) config) ---------------
        bond_struc_2 = []
        spc_2 = []
        if 2 in self.bodies:
            bodies = 2
            for spc1_ind, spc1 in enumerate(species_list):
                for spc2 in species_list[spc1_ind:]:
                    species = [spc1, spc2]
                    spc_2.append(species)
                    positions = [[(i+1)/(bodies+1)*cutoff, 0, 0]
                                 for i in range(bodies)]
                    spc_struc = \
                        struc.Structure(cell, species, positions)
                    spc_struc.coded_species = np.array(species)
                    bond_struc_2.append(spc_struc)

        # ------------------- 3 body (3 atoms (1 triplet) config) -------------
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
                        spc_struc = struc.Structure(cell, species, positions)
                        spc_struc.coded_species = np.array(species)
                        bond_struc_3.append(spc_struc)

#                    if spc1 != spc2:
#                        species = [spc2, spc3, spc1]
#                        spc_3.append(species)
#                        positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                    for i in range(bodies)]
#                        spc_struc = struc.Structure(cell, species, positions)
#                        spc_struc.coded_species = np.array(species)
#                        bond_struc_3.append(spc_struc)
#                    if spc2 != spc3:
#                        species = [spc3, spc1, spc2]
#                        spc_3.append(species)
#                        positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                    for i in range(bodies)]
#                        spc_struc = struc.Structure(cell, species, positions)
#                        spc_struc.coded_species = np.array(species)
#                        bond_struc_3.append(spc_struc)

        self.bond_struc = [bond_struc_2, bond_struc_3]
        self.spcs = [spc_2, spc_3]

    def predict(self, atom_env: AtomicEnvironment, mean_only: bool=False):
        '''
        predict force and variance for given atomic environment

        :param atom_env: atomic environment (with a center atom and its neighbors)
        :type  atom_env: :class:`AtomicEnvironment`
        :param mean_only: if True: only predict force (variance is always 0)
        :type  mean_only: Bool
        :return f: force on this atom
        :return v: variance corresponding to the force
        :return vir: stress on this atom
        '''
        if self.mean_only:  # if not build mapping for var
            mean_only = True

        # ---------------- predict for two body -------------------
        f2 = vir2 = kern2 = v2 = 0
        if 2 in self.bodies:
            sig2, ls2 = self.hyps[:2]
            r_cut2 = self.cutoffs[0]

            f2, vir2, kern2, v2 = \
                self.predict_multicomponent(atom_env, sig2, ls2, r_cut2,
                                            self.get_2body_comp, self.spcs[0],
                                            self.maps_2, mean_only)

        # ---------------- predict for three body -------------------
        f3 = vir3 = kern3 = v3 = 0
        if 3 in self.bodies:
            sig3, ls3, _ = self.hyps[-3:]
            r_cut3 = self.cutoffs[1]

            f3, vir3, kern3, v3 = \
                self.predict_multicomponent(atom_env, sig3, ls3, r_cut3,
                                            self.get_3body_comp, self.spcs[1],
                                            self.maps_3, mean_only)

        force = f2 + f3
        variance = kern2 + kern3 - np.sum((v2 + v3)**2, axis=0)
        virial_stress = vir2 + vir3

        return force, variance, virial_stress, 0

    def get_2body_comp(self, atom_env, sig, ls, r_cut):
        '''
        get bonds grouped by species
        '''
        bond_array_2 = atom_env.bond_array_2
        ctype = atom_env.ctype
        etypes = atom_env.etypes

        kern2 = np.zeros(3)
        for d in range(3):
            kern2[d] = \
                self_two_body_mc_jit(bond_array_2, ctype, etypes, d+1, sig, ls,
                                     r_cut, quadratic_cutoff)

        spcs, comp_r, comp_xyz = get_bonds(ctype, etypes, bond_array_2)
        return kern2, spcs, comp_r, comp_xyz

    def get_3body_comp(self, atom_env, sig, ls, r_cut):
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

        kern3_gp = np.zeros(3)
        for d in range(3):
            kern3_gp[d] = three_body_mc(atom_env, atom_env, d+1, d+1,
                                        self.hyps[-3:], self.cutoffs)

        spcs, comp_r, comp_xyz = \
            get_triplets(ctype, etypes, bond_array_3,
                         cross_bond_inds, cross_bond_dists, triplets)
        return kern3_gp, spcs, comp_r, comp_xyz

    def predict_multicomponent(self, atom_env, sig, ls, r_cut, get_comp,
                               spcs_list, mappings, mean_only):
        '''
        Add up results from `predict_component` to get the total contribution
        of all species
        '''
        f_spcs = 0
        vir_spcs = 0
        v_spcs = 0

        kern, spcs, comp_r, comp_xyz = get_comp(atom_env, sig, ls, r_cut)

        # predict for each species
        for i, spc in enumerate(spcs):
            lengths = np.array(comp_r[i])
            xyzs = np.array(comp_xyz[i])
            map_ind = spcs_list.index(spc)
            f, vir, v = self.predict_component(lengths, xyzs,
                                mappings[map_ind], mean_only)
            f_spcs += f
            vir_spcs += vir
            v_spcs += v

        return f_spcs, vir_spcs, kern, v_spcs

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
            v_0 = mapping.var(lengths)
            v_d = v_0 @ xyzs
            v = mapping.var.V @ v_d
        return f, vir, v


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


class Map2body:
    def __init__(self, grid_num, bounds, cutoffs, bond_struc, bodies='2',
                 svd_rank=0, mean_only=False, n_cpus=1, n_sample=100):
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
        self.n_cpus = n_cpus
        self.n_sample = n_sample

        self.build_map_container()

    def GenGrid(self, GP, processes=1):

        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''
        kernel_info = get_2bkernel(GP)

        if (self.n_cpus is None):
            processes = mp.cpu_count()
        else:
            processes = self.n_cpus

        # ------ construct grids ------
        nop = self.grid_num
        bond_lengths = np.linspace(self.l_bounds[0], self.u_bounds[0], nop)
        bond_means = np.zeros([nop])
        if not self.mean_only:
            bond_vars = np.zeros([nop, len(GP.alpha)])
        else:
            bond_vars = None
        env12 = AtomicEnvironment(self.bond_struc, 0, self.cutoffs)

        if processes == 1 :
            k12_v_all = self._GenGrid_inner(GP.name, 0, len(GP.training_data),
                                            bond_lengths, env12, kernel_info)
        else:
            with mp.Pool(processes=processes) as pool:
                size = len(GP.training_data)
                block_id, nbatch = partition_vector(self.n_sample, size, processes)

                k12_slice = []
                k12_v_all = np.zeros([len(bond_lengths), size*3])
                count = 0
                base = 0
                for ibatch in range(nbatch):
                    s, e = block_id[ibatch]
                    k12_slice.append(\
                            pool.apply_async(self._GenGrid_inner,
                                             args=(GP.name, s, e,
                                                   bond_lengths,
                                                   env12, kernel_info)))
                    count += 1
                    # when there are too many threads, collect some of
                    # the result to reduce memory footprint
                    if (count > processes*2):
                        for ibase in range(count):
                            s, e = block_id[ibase+base]
                            k12_v_all[:, s*3:e*3] = k12_slice[ibase].get()
                        del k12_slice
                        k12_slice = []
                        count = 0
                        base = ibatch+1
                if (count > 0):
                   for ibase in range(count):
                       s, e = block_id[ibase+base]
                       vec =  k12_slice[ibase].get()
                       k12_v_all[:, s*3:e*3] = k12_slice[ibase].get()
                   del k12_slice
                pool.close()
                pool.join()

        for b, r in enumerate(bond_lengths):
            k12_v = k12_v_all[b, :]
            bond_means[b] = np.matmul(k12_v, GP.alpha)
            if not self.mean_only:
                bond_vars[b, :] = solve_triangular(GP.l_mat, k12_v, lower=True)

        return bond_means, bond_vars


    def _GenGrid_inner(self, name, s, e, bond_lengths,
            env12, kernel_info):

        '''
        generate grid for each cos angle, used to parallelize grid generation
        '''

        kernel, ek, efk, cutoffs, hyps, hyps_mask = kernel_info
        size = e - s
        k12_v = np.zeros([len(bond_lengths), size*3])
        for b, r in enumerate(bond_lengths):
            env12.bond_array_2 = np.array([[r, 1, 0, 0]])
            k12_v[b, :] = \
                force_force_vector_unit(name, s, e, env12, kernel, hyps,
                                       cutoffs, hyps_mask, 1)
        return k12_v

    def build_map_container(self):
        self.mean = CubicSpline(self.l_bounds, self.u_bounds,
                                orders=[self.grid_num])

        if not self.mean_only:
            self.var = PCASplines(self.l_bounds, self.u_bounds,
                                  orders=[self.grid_num],
                                  svd_rank=self.svd_rank)

    def build_map(self, GP):
        '''
        build 1-d spline function for mean, 2-d for var
        '''
        assert (GP.multihyps is False), "multihyps is not supported in mgp"
        y_mean, y_var = self.GenGrid(GP)
        self.mean.set_values(y_mean)
        if not self.mean_only:
            self.var.set_values(y_var)

    def write(self, f, spc):
        '''
        Write LAMMPS coefficient file
        '''
        a = self.l_bounds[0]
        b = self.u_bounds[0]
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

    def __init__(self, grid_num, bounds, cutoffs, bond_struc, bodies='3',
            svd_rank=0, mean_only=False, load_grid=None, update=True,
            n_cpus=1, n_sample=100):
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
        self.n_cpus = n_cpus
        self.n_sample = n_sample

        self.build_map_container()


    def GenGrid(self, GP):
        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''

        if (self.n_cpus is None):
            processes = mp.cpu_count()
        else:
            processes = self.n_cpus

        if processes == 1:
            return self.GenGrid_serial(GP)

        # ------ get 3body kernel info ------
        kernel_info = get_3bkernel(GP)

        # ------ construct grids ------
        nop = self.grid_num[0]
        noa = self.grid_num[2]
        bond_lengths = np.linspace(self.l_bounds[0], self.u_bounds[0], nop)
        cos_angles = np.linspace(self.l_bounds[2], self.u_bounds[2], noa)

        bond_means = np.zeros([nop, nop, noa])
        if not self.mean_only:
            bond_vars = np.zeros([nop, nop, noa, len(GP.alpha)])
        else:
            bond_vars = None
        env12 = AtomicEnvironment(self.bond_struc, 0, self.cutoffs)

        with mp.Pool(processes=processes) as pool:
            if self.update:
                if 'kv3' in os.listdir():
                    os.rmdir('kv3')
                os.mkdir('kv3')

            size = len(GP.training_data)
            block_id, nbatch = partition_vector(self.n_sample, size, processes)

            k12_slice = []
            if (size>5000):
                print('parallel set up:', size, ns, n_sample, time.time())
            count = 0
            base = 0
            k12_v_all = np.zeros([len(bond_lengths), len(bond_lengths), len(cos_angles), size*3])
            for ibatch in range(nbatch):
                s, e = block_id[ibatch]
                k12_slice.append(\
                        pool.apply_async(self._GenGrid_inner_most,
                                         args=(GP.name, s, e,
                                               cos_angles, bond_lengths,
                                               env12, kernel_info)))
                if (size>5000):
                    print('send', ibatch, ns, s, e, time.time())
                count += 1
                if (count > processes*2):
                    for ibase in range(count):
                        s, e = block_id[ibase+base]
                        k12_v_all[:, :, :, s*3:e*3] = k12_slice[ibase].get()
                        if (size>5000):
                            print('get', ibase+base)
                    del k12_slice
                    k12_slice = []
                    count = 0
                    base = ibatch+1
            if (count > 0):
               for ibase in range(count):
                   s, e = block_id[ibase+base]
                   k12_v_all[:, :, :, s*3:e*3] = k12_slice[ibase].get()
               del k12_slice

            pool.close()
            pool.join()

        for a12, cos_angle in enumerate(cos_angles):
            for b1, r1 in enumerate(bond_lengths):
                for b2, r2 in enumerate(bond_lengths):
                    k12_v = k12_v_all[b1, b2, a12, :]
                    bond_means[b1, b2, a12] = np.matmul(k12_v, GP.alpha)
                    if not self.mean_only:
                        bond_vars[b1, b2, a12, :] = solve_triangular(GP.l_mat, k12_v, lower=True)


        # # ------ save mean and var to file -------
        np.save('grid3_mean', bond_means)
        np.save('grid3_var', bond_vars)

        return bond_means, bond_vars

    def GenGrid_serial(self, GP):
        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''
        
        # ------ get 3body kernel info ------
        kernel, ek, efk, cutoffs, hyps, hyps_mask = get_3bkernel(GP)

        # ------ construct grids ------
        nop = self.grid_num[0]
        noa = self.grid_num[2]
        bond_lengths = np.linspace(self.l_bounds[0], self.u_bounds[0], nop)
        cos_angles = np.linspace(self.l_bounds[2], self.u_bounds[2], noa)
        bond_means = np.zeros([nop, nop, noa])
        bond_vars = np.zeros([nop, nop, noa, len(GP.alpha)])
        env12 = AtomicEnvironment(self.bond_struc, 0, self.cutoffs)

        if self.update:
            if 'kv3' in os.listdir():
                os.rmdir('kv3')
            os.mkdir('kv3')

        size = len(GP.training_data)
        ds = [1, 2, 3]
        k_v = np.zeros(3)
        k12_v_all = np.zeros([len(bond_lengths), len(bond_lengths),
                              len(cos_angles), size*3])
        for b1, r1 in enumerate(bond_lengths):
            for b2, r2 in enumerate(bond_lengths):
                for a12, cos_angle12 in enumerate(cos_angles):

                    x2 = r2 * cos_angle12
                    y2 = r2 * np.sqrt(1-cos_angle12**2)
                    r12 = np.linalg.norm(np.array([x2-r1, y2, 0]))

                    env12.bond_array_3 = np.array([[r1, 1, 0, 0],
                                                   [r2, 0, 0, 0]])
                    env12.cross_bond_dists = np.array([[0, r12], [r12, 0]])

                    for isample, sample in enumerate(GP.training_data):
                        for d in ds:
                            k_v[d-1] = kernel(env12, sample, 1, d,
                                              hyps, cutoffs)

                        k12_v_all[b1, b2, a12, isample*3:isample*3+3] = k_v

        for b1, r1 in enumerate(bond_lengths):
            for b2, r2 in enumerate(bond_lengths):
                for a12, cos_angle in enumerate(cos_angles):
                    k12_v = k12_v_all[b1, b2, a12, :]
                    bond_means[b1, b2, a12] = np.matmul(k12_v, GP.alpha)
                    if not self.mean_only:
                        bond_vars[b1, b2, a12, :] = solve_triangular(GP.l_mat, k12_v, lower=True)

        # # ------ save mean and var to file -------
        np.save('grid3_mean', bond_means)
        np.save('grid3_var', bond_vars)

        return bond_means, bond_vars

    def _GenGrid_inner_most(self, name, s, e, cos_angles, bond_lengths, env12,
                            kernel_info):

        '''
        generate grid for each cos_angle, used to parallelize grid generation
        '''

        kernel, ek, efk, cutoffs, hyps, hyps_mask = kernel_info
        training_data = gp_algebra._global_training_data[name]
        # open saved k vector file, and write to new file
        size = (e-s)*3
        k12_v = np.zeros([len(bond_lengths), len(bond_lengths),
                          len(cos_angles), size])
        for a12, cos_angle12 in enumerate(cos_angles):
            for b1, r1 in enumerate(bond_lengths):
                for b2, r2 in enumerate(bond_lengths):

                    x2 = r2 * cos_angle12
                    y2 = r2 * np.sqrt(1-cos_angle12**2)
                    r12 = np.linalg.norm(np.array([x2-r1, y2, 0]))

                    env12.bond_array_3 = np.array([[r1, 1, 0, 0],
                                                   [r2, 0, 0, 0]])
                    env12.cross_bond_dists = np.array([[0, r12], [r12, 0]])

                    k12_v[b1, b2, a12, :] = \
                        force_force_vector_unit(name, s, e, env12, kernel, hyps,
                                               cutoffs, hyps_mask, 1)

        return k12_v

    def build_map_container(self):
       # create spline interpolation class object
        self.mean = CubicSpline(self.l_bounds, self.u_bounds,
                                orders=self.grid_num)

        if not self.mean_only:
            self.var = PCASplines(self.l_bounds, self.u_bounds,
                                  orders=self.grid_num,
                                  svd_rank=self.svd_rank)

    def build_map(self, GP):

        '''
        build 3-d spline function for mean,
        3-d for the low rank approximation of L^{-1}k*
        '''

        assert (GP.multihyps is False), "multihyps is not supported in mgp"

        # Load grid or generate grid values
        if not self.load_grid:
            y_mean, y_var = self.GenGrid(GP)
        else:
            y_mean = np.load(self.load_grid+'/grid3_mean.npy')
            y_var = np.load(self.load_grid+'/grid3_var.npy')

        self.mean.set_values(y_mean)
        if not self.mean_only:
            self.var.set_values(y_var)

    def write(self, f, spc):
        a = self.l_bounds
        b = self.u_bounds
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



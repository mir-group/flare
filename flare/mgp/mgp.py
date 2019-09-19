import time
import math
import numpy as np
from scipy.linalg import solve_triangular
import multiprocessing as mp
import subprocess
import os

from flare import gp, env, struc, kernels
from flare.gp import GaussianProcess
from flare.kernels import two_body, three_body, two_plus_three_body,\
    two_body_jit
from flare.cutoffs import quadratic_cutoff
from flare.mc_simple import two_body_mc, three_body_mc, two_plus_three_body_mc
import flare.mgp.utils as utils
from flare.mgp.utils import get_bonds, get_triplets, self_two_body_mc_jit, \
    self_three_body_mc_jit
from flare.mgp.splines_methods import PCASplines, SplinesInterpolation


class MappedGaussianProcess:
    def __init__(self, GP: GaussianProcess, grid_params: dict,
                 struc_params: dict, mean_only=False):

        '''
        :param: GP : gp model
        :param: struc_params : {'species': [0, 1],
                'cube_lat': cell, # should input the cell matrix
                'mass_dict': {'0': 27 * unit, '1': 16 * unit}}
        :param: grid_params : {'bounds_2': [[1.2], [3.5]], # [[lower_bound],
                                                            [upper]]
               'bounds_3': [[1.2, 1.2, 0], [3.5, 3.5, np.pi]],
                    #[[lower,lower,0],[upper,upper,np.pi]]
               'grid_num_2': 64,
               'grid_num_3': [16, 16, 16],
               'svd_rank_2': 64,
               'svd_rank_3': 16**3,
               'bodies': [2, 3],
               'update': True,
               'load_grid': None}
        :param: mean_only : if True: only build mapping for mean (force)

        '''
        self.GP = GP
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

        bond_struc, spcs = self.build_bond_struc(struc_params)
        self.spcs = spcs
        self.maps_2 = []
        self.maps_3 = []
        if 2 in self.bodies:
            for b_struc in bond_struc[0]:
                map_2 = Map2body(self.grid_num_2, self.bounds_2, self.GP,
                                 b_struc, self.bodies,
                                 grid_params['load_grid'],
                                 self.svd_rank_2, self.mean_only)
                self.maps_2.append(map_2)
        if 3 in self.bodies:
            for b_struc in bond_struc[1]:
                map_3 = Map3body(self.grid_num_3, self.bounds_3, self.GP,
                                 b_struc, self.bodies,
                                 grid_params['load_grid'],
                                 self.svd_rank_3,
                                 self.mean_only, self.update)

                self.maps_3.append(map_3)

    def build_bond_struc(self, struc_params):

        '''
        build a bond structure, used in grid generating
        '''

        cutoff = np.min(self.GP.cutoffs)
        cell = struc_params['cube_lat']
        mass_dict = struc_params['mass_dict']
        species_list = struc_params['species']
        N_spc = len(species_list)

        # ------------------- 2 body (2 atoms (1 bond) config) ---------------
        bodies = 2
        bond_struc_2 = []
        spc_2 = []
        for spc1_ind, spc1 in enumerate(species_list):
            for spc2 in species_list[spc1_ind:]:
                species = [spc1, spc2]
                spc_2.append(species)
                positions = [[(i+1)/(bodies+1)*cutoff, 0, 0]
                             for i in range(bodies)]
                spc_struc = \
                    struc.Structure(cell, species, positions, mass_dict)
                spc_struc.coded_species = np.array(species)
                bond_struc_2.append(spc_struc)

        # ------------------- 3 body (3 atoms (1 triplet) config) -------------
        bodies = 3
        bond_struc_3 = []
        spc_3 = []
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
#                    if spc1 != spc2:
#                        species = [spc2, spc3, spc1]
#                        spc_3.append(species)
#                        positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                    for i in range(bodies)]
#                        spc_struc = struc.Structure(cell, species, positions,
#                                                    mass_dict)
#                        spc_struc.coded_species = np.array(species)
#                        bond_struc_3.append(spc_struc)
#                    if spc2 != spc3:
#                        species = [spc3, spc1, spc2]
#                        spc_3.append(species)
#                        positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                    for i in range(bodies)]
#                        spc_struc = struc.Structure(cell, species, positions,
#                                                    mass_dict)
#                        spc_struc.coded_species = np.array(species)
#                        bond_struc_3.append(spc_struc)

        bond_struc = [bond_struc_2, bond_struc_3]
        spcs = [spc_2, spc_3]
        return bond_struc, spcs

    def predict(self, atom_env, mean_only=False):
        if self.mean_only:  # if not build mapping for var
            mean_only = True

        # ---------------- predict for two body -------------------
        f2 = kern2 = v2 = 0
        if 2 in self.bodies:
            sig2, ls2 = self.GP.hyps[:2]
            r_cut2 = self.GP.cutoffs[0]

            f2, kern2, v2 = \
                self.predict_multicomponent(atom_env, sig2, ls2, r_cut2,
                                            self.get_2body_comp, self.spcs[0],
                                            self.maps_2, mean_only)

        # ---------------- predict for three body -------------------
        f3 = kern3 = v3 = 0
        if 3 in self.bodies:
            sig3, ls3, _ = self.GP.hyps[-3:]
            r_cut3 = self.GP.cutoffs[1]

            f3, kern3, v3 = \
                self.predict_multicomponent(atom_env, sig3, ls3, r_cut3,
                                            self.get_3body_comp, self.spcs[1],
                                            self.maps_3, mean_only)

        f = f2 + f3
        v = kern2 + kern3 - np.sum((v2 + v3)**2, axis=0)
        return f, v

    def get_2body_comp(self, atom_env, sig, ls, r_cut):
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
                                        self.GP.hyps[-3:], self.GP.cutoffs)
        # print(kern3, kern3_gp)

        spcs, comp_r, comp_xyz = \
            get_triplets(ctype, etypes, bond_array_3,
                         cross_bond_inds, cross_bond_dists, triplets)
        return kern3_gp, spcs, comp_r, comp_xyz

    def predict_multicomponent(self, atom_env, sig, ls, r_cut, get_comp,
                               spcs_list, mappings, mean_only):
        f_spcs = 0
        v_spcs = 0

        kern, spcs, comp_r, comp_xyz = get_comp(atom_env, sig, ls, r_cut)

        # predict for each species
        for i, spc in enumerate(spcs):
            lengths = np.array(comp_r[i])
            xyzs = np.array(comp_xyz[i])
            map_ind = spcs_list.index(spc)
            f, v = self.predict_component(lengths, xyzs, mappings[map_ind],
                                          mean_only)
            f_spcs += f
            v_spcs += v

        return f_spcs, kern, v_spcs

    def predict_component(self, lengths, xyzs, mapping, mean_only):
        '''
        predict for an atom environment
        param: atom_env: ChemicalEnvironment
        return force on an atom with its variance
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

    def write_two_plus_three(self, lammps_name):
        # write header
        f = open(lammps_name, 'w')

        header_comment = '''# #2bodyarray #3bodyarray
        # elem1 elem2 a b order
        '''
        f.write(header_comment)

        twobodyarray = len(self.spcs[0])
        threebodyarray = len(self.spcs[1])
        lower_cut = self.bounds_2[0][0]
        two_cut = self.bounds_2[1][0]
        three_cut = self.bounds_3[1][0]
        grid_num_2 = self.grid_num_2
        grid_num_3 = self.grid_num_3[0]
        angle_lower = self.bounds_3[0][2]
        angle_upper = self.bounds_3[1][2]

        header = '\n{} {}\n'.format(twobodyarray, threebodyarray)
        f.write(header)

        a = lower_cut
        b = two_cut
        order = grid_num_2

        # write two body
        for ind, spc in enumerate(self.spcs[0]):
            coefs_2 = self.maps_2[ind].mean.model.__coeffs__

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

        # write three body
        a = [lower_cut, lower_cut, angle_lower]
        b = [three_cut, three_cut, angle_upper]
        order = [grid_num_3, grid_num_3, grid_num_3]

        for ind, spc in enumerate(self.spcs[1]):
            coefs_3 = self.maps_3[ind].mean.model.__coeffs__

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


class Map2body:
    def __init__(self, grid_num, bounds, GP, bond_struc, bodies='2',
                 load_prefix=None, svd_rank=0, mean_only=False):
        '''
        param grids: the 1st element is the number of grids for mean
        prediction, the 2nd is for var
        '''

        self.grid_num = grid_num
        self.l_bound, self.u_bound = bounds
        self.cutoffs = GP.cutoffs
        self.bodies = bodies
        self.svd_rank = svd_rank
        self.species = bond_struc.coded_species
        self.mean_only = mean_only

        y_mean, y_var = self.GenGrid(GP, bond_struc)

        self.build_map(y_mean, y_var)

    def GenGrid(self, GP, bond_struc, processes=mp.cpu_count()):

        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''

        # ------ change GP kernel to 2 body ------
        original_kernel = GP.kernel
        GP.kernel = two_body_mc
        original_cutoffs = np.copy(GP.cutoffs)
        GP.cutoffs = [GP.cutoffs[0]]
        original_hyps = np.copy(GP.hyps)
        GP.hyps = [GP.hyps[0], GP.hyps[1], GP.hyps[-1]]

        # ------ construct grids ------
        nop = self.grid_num
        bond_lengths = np.linspace(self.l_bound[0], self.u_bound[0], nop)
        bond_means = np.zeros([nop])
        bond_vars = np.zeros([nop, len(GP.alpha)])
        env12 = env.AtomicEnvironment(bond_struc, 0, self.cutoffs)

        pool_list = [(i, bond_lengths, GP, env12)
                     for i in range(nop)]
        pool = mp.Pool(processes=processes)
        A_list = pool.map(self._GenGrid_inner, pool_list)
        for p in range(nop):
            bond_means[p] = A_list[p][0]
            bond_vars[p, :] = A_list[p][1]
        pool.close()
        pool.join()

        # ------ change back original GP ------
        GP.cutoffs = original_cutoffs
        GP.hyps = original_hyps
        GP.kernel = original_kernel

        return bond_means, bond_vars

    def _GenGrid_inner(self, params):

        '''
        generate grid for each angle, used to parallelize grid generation
        '''
        b, bond_lengths, GP, env12 = params
        # nop = self.grid_num
        r = bond_lengths[b]
        env12.bond_array_2 = np.array([[r, 1, 0, 0]])

        k12_v = GP.get_kernel_vector(env12, 1)
        mean_diff = np.matmul(k12_v, GP.alpha)
        bond_means = mean_diff
        bond_vars = np.zeros(k12_v.shape)

        if not self.mean_only:
            v12_vec = solve_triangular(GP.l_mat, k12_v, lower=True)
            bond_vars = v12_vec

        return bond_means, bond_vars

    def build_map(self, y_mean, y_var):

        '''
        build 1-d spline function for mean, 2-d for var
        '''

        self.mean = \
            SplinesInterpolation(y_mean, u_bounds=np.array(self.u_bound),
                                 l_bounds=np.array(self.l_bound),
                                 orders=np.array([self.grid_num]))

        if not self.mean_only:
            self.var = \
                PCASplines(y_var, u_bounds=np.array(self.u_bound),
                           l_bounds=np.array(self.l_bound),
                           orders=np.array([self.grid_num]),
                           svd_rank=self.svd_rank)


class Map3body:

    def __init__(self, grid_num, bounds, GP, bond_struc, bodies='3',
                 load_grid=None, svd_rank=0, mean_only=False, update=True):
        '''
        param grids: the 1st element is the number of grids for mean
        prediction, the 2nd is for var
        '''

        self.grid_num = grid_num
        self.l_bound, self.u_bound = bounds
        self.cutoffs = GP.cutoffs
        self.bodies = bodies
        self.species = bond_struc.coded_species
        self.mean_only = mean_only

        if not load_grid:
            y_mean, y_var = self.GenGrid(GP, bond_struc, update)
        else:
            y_mean = np.load('grid3_mean.npy')
            y_var = np.load('grid3_var.npy')

        self.build_map(y_mean, y_var, svd_rank=svd_rank) 

    def GenGrid(self, GP, bond_struc, update, processes=mp.cpu_count()):

        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''
        # ------ change GP kernel to 3 body ------
        original_kernel = GP.kernel
        original_hyps = np.copy(GP.hyps)
        GP.kernel = three_body_mc
        GP.hyps = GP.hyps[-3:]

        # ------ construct grids ------
        nop = self.grid_num[0]
        noa = self.grid_num[2]
        bond_lengths = np.linspace(self.l_bound[0], self.u_bound[0], nop)
        angles = np.linspace(self.l_bound[2], self.u_bound[2], noa)
        bond_means = np.zeros([nop, nop, noa])
        bond_vars = np.zeros([nop, nop, noa, len(GP.alpha)])
        env12 = env.AtomicEnvironment(bond_struc, 0, self.cutoffs)

        pool_list = [(i, angles[i], bond_lengths, GP, env12, update)\
                     for i in range(noa)]
        pool = mp.Pool(processes=processes)

        if update:
            subprocess.run(['rm', '-r', 'kv3'])
            subprocess.run(['mkdir', 'kv3'])
       
        A_list = pool.map(self._GenGrid_inner, pool_list)
        for a12 in range(noa):
            bond_means[:, :, a12] = A_list[a12][0]
            bond_vars[:, :, a12, :] = A_list[a12][1]
        pool.close()
        pool.join()

        # ------ change back to original GP ------
        GP.hyps = original_hyps
        GP.kernel = original_kernel
      
        # ------ save mean and var to file -------
        np.save('grid3_mean', bond_means)
        np.save('grid3_var', bond_vars)

        return bond_means, bond_vars

    def _GenGrid_inner(self, params):

        '''
        generate grid for each angle, used to parallelize grid generation
        '''
        a12, angle12, bond_lengths, GP, env12, update = params
        nop = self.grid_num[0]
        angle12 = angle12
        bond_means = np.zeros([nop, nop])
        bond_vars = np.zeros([nop, nop, len(GP.alpha)])

        # open saved k vector file, and write to new file
        if update:
            kv_filename = 'kv3/'+str(a12)
            size = len(GP.training_data) * 3
            new_kv_file = np.zeros((nop**2+1, size))
            new_kv_file[0,0] = size
            if str(a12)+'.npy' in os.listdir('kv3'):
                old_kv_file = np.load(kv_filename+'.npy') 
                last_size = int(old_kv_file[0,0])
                new_kv_file[:, :last_size] = old_kv_file
            else:
                last_size = 0
            ds = [1, 2, 3]

        for b1, r1 in enumerate(bond_lengths):
            r1 = bond_lengths[b1]
            for b2, r2 in enumerate(bond_lengths):
                x2 = r2 * np.cos(angle12)
                y2 = r2 * np.sin(angle12)
                r12 = np.linalg.norm(np.array([x2-r1, y2, 0]))

                env12.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])
                env12.cross_bond_dists = np.array([[0, r12], [r12, 0]])

                # calculate kernel functions of those newly added training data
                if update:
                    k12_v = new_kv_file[1+b1*nop+b2, :]
                    for m_index in range(last_size, size):
                        x_2 = GP.training_data[int(math.floor(m_index / 3))]
                        d_2 = ds[m_index % 3]
                        k12_v[m_index] = GP.kernel(env12, x_2, 1, d_2,
                                               GP.hyps, GP.cutoffs)
                else:
                    k12_v = GP.get_kernel_vector(env12, 1)   

                new_kv_file[1+b1*nop+b2, :] = k12_v

                # calculate mean and var value for the mapping
                mean_diff = np.matmul(k12_v, GP.alpha)
                bond_means[b1, b2] = mean_diff

                if not self.mean_only:
                    v12_vec = solve_triangular(GP.l_mat, k12_v, lower=True)
                    bond_vars[b1, b2, :] = v12_vec

        # replace the old file with the new file
        if update:
            np.save(kv_filename, new_kv_file)

        return bond_means, bond_vars

    def build_map(self, y_mean, y_var, svd_rank):

        '''
        build 3-d spline function for mean,
        3-d for the low rank approximation of L^{-1}k*
        '''
        nop = self.grid_num[0]
        noa = self.grid_num[2]
        self.mean = \
            SplinesInterpolation(y_mean, u_bounds=self.u_bound,
                                 l_bounds=self.l_bound,
                                 orders=np.array([nop, nop, noa]))

        if not self.mean_only:
            self.var = \
                PCASplines(y_var, u_bounds=self.u_bound,
                           l_bounds=self.l_bound,
                           orders=np.array([nop, nop, noa]),
                           svd_rank=svd_rank)


import time
from math import exp
import numpy as np
from scipy.linalg import solve_triangular
import multiprocessing as mp
import sys
sys.path.append('../../flare/')

from flare import gp, env, struc, kernels
from flare.kernels import two_body, three_body, two_plus_three_body, two_body_jit
from flare.cutoffs import quadratic_cutoff
from flare.mc_simple import two_body_mc, three_body_mc, two_plus_three_body_mc

import flare.mff.utils as utils
from flare.mff.utils import get_bonds, get_triplets, self_two_body_mc_jit, self_three_body_mc_jit 
from flare.mff.splines_methods import PCASplines, SplinesInterpolation

class MappedForceField:
    
    def __init__(self, GP, grid_params, struc_params):
    
        '''
        param: struc_params = {'species': 'C', 'cube_lat': 2*1.763391008}
        param: grid_params = {'grid_num': list, 'bounds': list, 
                            'svd_rank': int>0, 'load_grid': None, 
                            'load_svd': None}
        '''
        self.GP = GP
        self.bodies = grid_params['bodies']
        self.grid_num_2 = grid_params['grid_num_2']
        self.bounds_2 = grid_params['bounds_2']
        self.grid_num_3 = grid_params['grid_num_3']
        self.bounds_3 = grid_params['bounds_3']
        self.svd_rank = grid_params['svd_rank']
        
        bond_struc, spcs = self.build_bond_struc(struc_params)
        self.spcs = spcs
        self.maps_2 = []
        self.maps_3 = []
        if 2 in self.bodies:
            for b_struc in bond_struc[0]:
                map_2 = Map2body(self.grid_num_2, self.bounds_2, self.GP, b_struc,
                         self.bodies, grid_params['load_grid'], self.svd_rank)
                self.maps_2.append(map_2)
        if 3 in self.bodies:
            for b_struc in bond_struc[1]:
                map_3 = Map3body(self.grid_num_3, self.bounds_3, self.GP, b_struc, 
                         self.bodies, grid_params['load_grid'],
                         grid_params['load_svd'], self.svd_rank)
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
                positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
                            for i in range(bodies)]
                spc_struc = struc.Structure(cell, species, positions, mass_dict)
                spc_struc.coded_species = np.array(species)
                bond_struc_2.append(spc_struc)

        # ------------------- 3 body (3 atoms (1 triplet) config) -------------
        bodies = 3
        bond_struc_3 = []
        spc_3 = []
        for spc1_ind in range(N_spc):
            spc1 = species_list[spc1_ind]
            for spc2_ind in range(spc1_ind, N_spc):
                spc2 = species_list[spc2_ind]
                for spc3_ind in range(spc2_ind, N_spc):
                    spc3 = species_list[spc3_ind]
                    species = [spc1, spc2, spc3]
                    spc_3.append(species)
                    positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
                                for i in range(bodies)]
                    spc_struc = struc.Structure(cell, species, positions, mass_dict)
                    spc_struc.coded_species = np.array(species)
                    bond_struc_3.append(spc_struc)
                    if spc1 != spc2:
                        species = [spc2, spc3, spc1]
                        spc_3.append(species)
                        positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
                                    for i in range(bodies)]
                        spc_struc = struc.Structure(cell, species, positions, mass_dict)
                        spc_struc.coded_species = np.array(species)
                        bond_struc_3.append(spc_struc)
                    if spc2 != spc3:
                        species = [spc3, spc1, spc2]
                        spc_3.append(species)
                        positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
                                    for i in range(bodies)]
                        spc_struc = struc.Structure(cell, species, positions, mass_dict)
                        spc_struc.coded_species = np.array(species)
                        bond_struc_3.append(spc_struc)
                     
        bond_struc = [bond_struc_2, bond_struc_3]
        spcs = [spc_2, spc_3]
        return bond_struc, spcs

    def predict(self, atom_env, mean_only=False):
        # ---------------- predict for two body -------------------
        f2 = kern2 = v2 = 0
        if 2 in self.bodies:
            sig2, ls2 = self.GP.hyps[:2]
            r_cut2 = self.GP.cutoffs[0]

            f2, kern2, v2 = self.predict_multicomponent(atom_env, sig2, ls2, r_cut2,
                    self.get_2body_comp, self.maps_2, mean_only)

        # ---------------- predict for three body -------------------
        f3 = kern3 = v3 = 0
        if 3 in self.bodies:
            sig3, ls3, noise = self.GP.hyps[-3:]
            r_cut3 = self.GP.cutoffs[1]

            f3, kern3, v3 = self.predict_multicomponent(atom_env, sig3, ls3, r_cut3,
                    self.get_3body_comp, self.maps_3, mean_only)

        f = f2 + f3
        v = kern2 + kern3 - np.sum((v2 + v3)**2, axis=0)
        return f, v

    def get_2body_comp(self, atom_env, sig, ls, r_cut):
        bond_array_2 = atom_env.bond_array_2
        ctype = atom_env.ctype
        etypes = atom_env.etypes

        kern2 = np.zeros(3)
        for d in range(3):
            kern2[d] = self_two_body_mc_jit(bond_array_2, ctype, etypes, 
                    d+1, sig, ls, r_cut, quadratic_cutoff)

        spcs, comp_r, comp_xyz = get_bonds(ctype, etypes, bond_array_2)
        return kern2, spcs, comp_r, comp_xyz

    def get_3body_comp(self, atom_env, sig, ls, r_cut):
        bond_array_3 = atom_env.bond_array_3
        cross_bond_inds = atom_env.cross_bond_inds
        cross_bond_dists = atom_env.cross_bond_dists
        triplets = atom_env.triplet_counts
        ctype = atom_env.ctype
        etypes = atom_env.etypes

        kern3 = np.zeros(3)
        for d in range(3):
            kern3[d] = self_three_body_mc_jit(bond_array_3, cross_bond_inds, 
                    cross_bond_dists, triplets, ctype, etypes, d+1, sig, ls,
                    r_cut, quadratic_cutoff)

        spcs, comp_r, comp_xyz = get_triplets(ctype, etypes, bond_array_3, 
                    cross_bond_inds, cross_bond_dists, triplets)
        return kern3, spcs, comp_r, comp_xyz

    def predict_multicomponent(self, atom_env, sig, ls, r_cut, get_comp,
            mappings, mean_only):
        f_spcs = 0
        v_spcs = 0

        kern, spcs, comp_r, comp_xyz = get_comp(atom_env, sig, ls, r_cut) 

        # predict for each species
        for i, spc in enumerate(spcs):
            lengths = np.array(comp_r[i])
            xyzs = np.array(comp_xyz[i])
            map_ind = self.spcs[0].index(spc)
            f, v = self.predict_component(lengths, xyzs, 
                    self.maps_2[map_ind], mean_only)
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
             
class Map2body:
   
    def __init__(self, grid_num, bounds, GP, bond_struc, bodies='2', 
            load_prefix=None, svd_rank=0): 
    
        '''
        param grids: the 1st element is the number of grids for mean prediction, 
                    the 2nd is for var
        '''       
        
        self.grid_num = grid_num
        self.l_bound, self.u_bound = bounds
        self.cutoffs = GP.cutoffs
        self.bodies = bodies
        self.svd_rank = svd_rank
        self.species = bond_struc.species
        
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
        
        pool_list = [(i, bond_lengths, GP, env12)\
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
        nop = self.grid_num
        r = bond_lengths[b]
        env12.bond_array_2 = np.array([[r, 1, 0, 0]])
        k12_v = GP.get_kernel_vector(env12, 1)   
        v12_vec = solve_triangular(GP.l_mat, k12_v, lower=True)
        mean_diff = np.matmul(k12_v, GP.alpha)
        bond_means = mean_diff
        bond_vars = v12_vec  
                      
        return bond_means, bond_vars

    def build_map(self, y_mean, y_var):
    
        '''
        build 1-d spline function for mean, 2-d for var
        '''
        
        self.mean = SplinesInterpolation(y_mean, 
                    u_bounds=np.array(self.u_bound), 
                    l_bounds=np.array(self.l_bound), 
                    orders=np.array([self.grid_num]))

        self.var = PCASplines(y_var, u_bounds=np.array(self.u_bound), 
                       l_bounds=np.array(self.l_bound), 
                       orders=np.array([self.grid_num]), 
                       svd_rank=self.svd_rank, load_svd=None)


class Map3body:
   
    def __init__(self, grid_num, bounds, GP, bond_struc, bodies='3', 
                load_grid=None, load_svd=None, svd_rank=0): 
    
        '''
        param grids: the 1st element is the number of grids for mean prediction, 
                    the 2nd is for var
        '''       
        
        self.grid_num = grid_num
        self.l_bound, self.u_bound = bounds
        self.cutoffs = GP.cutoffs
        self.bodies = bodies
        self.species = bond_struc.species
        
        if not load_grid:
            y_mean, y_var = self.GenGrid(GP, bond_struc)    
        else:
            y_mean, y_var = utils.merge(load_grid, noa, nop)
        self.build_map(y_mean, y_var, svd_rank=svd_rank, load_svd=load_svd) 

    def GenGrid(self, GP, bond_struc, processes=mp.cpu_count()):
    
        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''
        original_hyps = np.copy(GP.hyps)
        if self.bodies == '2+3':
            # ------ change GP kernel to 3 body ------
            GP.kernel = three_body_mc
            GP.hyps = [GP.hyps[2], GP.hyps[3], GP.hyps[-1]]

        # ------ construct grids ------
        nop = self.grid_num[0]
        noa = self.grid_num[2]
        bond_lengths = np.linspace(self.l_bound[0], self.u_bound[0], nop)
        angles = np.linspace(self.l_bound[2], self.u_bound[2], noa)
        bond_means = np.zeros([nop, nop, noa])
        bond_vars = np.zeros([nop, nop, noa, len(GP.alpha)])
        env12 = env.AtomicEnvironment(bond_struc, 0, self.cutoffs)
        
        pool_list = [(i, angles[i], bond_lengths, GP, env12)\
                     for i in range(noa)]
        pool = mp.Pool(processes=processes)
        A_list = pool.map(self._GenGrid_inner, pool_list)
        for a12 in range(noa):
            bond_means[:, :, a12] = A_list[a12][0]
            bond_vars[:, :, a12, :] = A_list[a12][1]
        pool.close()
        pool.join()

        # ------ change back to original GP ------
        if 2 in self.bodies and 3 in self.bodies:
            GP.hyps = original_hyps
            GP.kernel = two_plus_three_body
       
        return bond_means, bond_vars


    def _GenGrid_inner(self, params):
    
        '''
        generate grid for each angle, used to parallelize grid generation
        '''
        a12, angle12, bond_lengths, GP, env12 = params
        nop = self.grid_num[0]
        noa = self.grid_num[2]
        angle12 = angle12
        bond_means = np.zeros([nop, nop])
        bond_vars = np.zeros([nop, nop, len(GP.alpha)])
        
        for b1, r1 in enumerate(bond_lengths):
            r1 = bond_lengths[b1]
            for b2, r2 in enumerate(bond_lengths):
                x2 = r2 * np.cos(angle12)
                y2 = r2 * np.sin(angle12)
                r12 = np.linalg.norm(np.array([x2-r1, y2, 0]))

                env12.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])
                env12.cross_bond_dists = np.array([[0, r12], [r12, 0]])
                k12_v = GP.get_kernel_vector(env12, 1)   
                v12_vec = solve_triangular(GP.l_mat, k12_v, lower=True)
                mean_diff = np.matmul(k12_v, GP.alpha)
                bond_means[b1, b2] = mean_diff
                bond_vars[b1, b2, :] = v12_vec  
                      
        return bond_means, bond_vars


    def build_map(self, y_mean, y_var, svd_rank, load_svd):
    
        '''
        build 3-d spline function for mean, 
        3-d for the low rank approximation of L^{-1}k*
        '''
        nop = self.grid_num[0]
        noa = self.grid_num[2]
        self.mean = SplinesInterpolation(y_mean, u_bounds=self.u_bound, 
                    l_bounds=self.l_bound, orders=np.array([nop, nop, noa])) 

        self.var = PCASplines(y_var, u_bounds=self.u_bound, l_bounds=self.l_bound, 
                   orders=np.array([nop, nop, noa]), svd_rank=svd_rank, 
                   load_svd=load_svd)

    def build_selfkern(self, grid_kern):
        self.selfkern = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
        self.selfkern.fit(grid_kern)
       


import time
import math
from math import exp
import numpy as np
from numba import njit
from scipy.linalg import solve_triangular
import multiprocessing as mp
import sys
sys.path.append('../../flare/')
from memory_profiler import profile
import subprocess
import struct

import flare.gp as gp
import flare.env as env
from flare.kernels import two_body, three_body, two_plus_three_body, two_body_jit
import flare.struc as struc

import flare.mgp.utils as utils
from flare.mgp.splines_methods import PCASplines, SplinesInterpolation

class MappedGaussianProcess:
    
    def __init__(self, GP, grid_params, struc_params):
    
        '''
        param: struc_params = {'species': 'C', 'cube_lat': 2*1.763391008}
        param: grid_params = {'grid_num': list, 'bounds': list, 
                            'svd_rank': int>0, 'load_grid': None, 
                            'load_svd': None}
        '''
        self.GP = GP
        self.grid_params = grid_params
        self.struc_params = struc_params
        self.bodies = str(grid_params['bodies'])
        self.grid_num_2 = grid_params['grid_num_2']
        self.bounds_2 = grid_params['bounds_2']
        self.grid_num_3 = grid_params['grid_num_3']
        self.bounds_3 = grid_params['bounds_3']
        self.svd_rank_2 = grid_params['svd_rank_2']
        self.svd_rank_3 = grid_params['svd_rank_3']
        self.update = grid_params['update']
        
        bond_struc = self.build_bond_struc(struc_params)
        if len(GP.training_data) > 0:
            if self.bodies == '2':
                self.map = Map2body(self.grid_num_2, self.bounds_2, self.GP, bond_struc,  
                       self.bodies, grid_params['load_grid'], self.svd_rank_2)
            elif self.bodies == '3':
                self.map = Map3body(self.grid_num_3, self.bounds_3, self.GP, bond_struc, 
                       self.bodies, grid_params['load_grid'], 
                       grid_params['load_svd'], self.svd_rank_3, self.update)
            elif self.bodies == '2+3':
                self.map_2 = Map2body(self.grid_num_2, self.bounds_2, self.GP, bond_struc[0],
                         self.bodies, grid_params['load_grid'], self.svd_rank_2)
                self.map_3 = Map3body(self.grid_num_3, self.bounds_3, self.GP,
                         bond_struc[1], self.bodies, grid_params['load_grid'],
                         grid_params['load_svd'], self.svd_rank_3, self.update)

    def build_bond_struc(self, struc_params):
    
        '''
        build a bond structure, used in grid generating
        '''
    
        cutoff = np.min(self.GP.cutoffs)
        cell = struc_params['cube_lat']
        mass_dict = struc_params['mass_dict']
        bond_struc = []
        for bodies in [2, 3]:
            species = [struc_params['species'] for i in range(bodies)]
            positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
                        for i in range(bodies)]
            bond_struc.append(struc.Structure(cell, species, positions, mass_dict))
        if self.bodies == '2':
            return bond_struc[0]
        elif self.bodies == '3':
            return bond_struc[1]
        elif self.bodies == '2+3':
            return bond_struc

    def predict(self, atom_env, mean_only=False):
        if self.bodies == '2':
            f, v = self.map.predict(atom_env, self.GP, mean_only)
        elif self.bodies == '3':
            f, v = self.map.predict(atom_env, self.GP, mean_only)
        elif self.bodies == '2+3':
            f2, kern2, v2 = self.map_2.predict(atom_env, self.GP, mean_only)
            f3, kern3, v3 = self.map_3.predict(atom_env, self.GP, mean_only)
            f = f2 + f3
            v = kern2 + kern3 - np.sum((v2 + v3)**2, axis=0)
        return f, v
              
class Map2body:
   
    def __init__(self, grid_num, bounds, GP, bond_struc, bodies='2', load_prefix=None, svd_rank=0): 
    
        '''
        param grids: the 1st element is the number of grids for mean prediction, 
                    the 2nd is for var
        '''       
        
        self.grid_num = grid_num
        self.l_bound, self.u_bound = bounds
        self.cutoffs = GP.cutoffs
        self.bodies = bodies
        self.svd_rank = svd_rank
        
        if self.bodies == '2':
            y_mean, y_var = self.GenGrid(GP, bond_struc)
        elif self.bodies == '2+3':
            y_mean, y_var = self.GenGrid_svd(GP, bond_struc)

        self.build_map(y_mean, y_var)

    def GenGrid(self, GP, bond_struc, processes=mp.cpu_count()):
    
        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
        default implemented in a parallelized style
        '''
        processes = mp.cpu_count()
        nop = self.grid_num
        bond_lengths = np.linspace(self.l_bound, self.u_bound, nop)
        bond_means = np.zeros([nop])
        bond_vars = np.zeros([nop, nop])

        env1 = env.AtomicEnvironment(bond_struc, 0, self.cutoffs)
        env2 = env.AtomicEnvironment(bond_struc, 0, self.cutoffs)

        pool_list = [(i, bond_lengths[i], bond_lengths, GP, env1, env2) for i in range(nop)]
        pool = mp.Pool(processes=processes)
        A_list = pool.starmap(self._GenGrid_inner, pool_list)
        pool.close()
        pool.join()

        A_list.sort(key=lambda x: x[0])
        for b1 in range(nop):            
            bond_means[b1] = A_list[b1][1]
            bond_vars[b1, :] = A_list[b1][2]
        
        return bond_means, bond_vars


    def _GenGrid_inner(self, b1, r1, bond_lengths, GP, env1, env2):
    
        '''
        generate grid for each angle, used to parallelize grid generation
        '''
        
        
        nop = self.grid_num
        bond_vars = np.zeros(nop)
        
        bond1 = np.array([r1, 1.0, 0.0, 0.0])
        env1.bond_array_2 = np.array([bond1])
#        env1.cross_bond_dists = np.array([[0]])
        
        k1_v = GP.get_kernel_vector(env1, 1)
        v1_vec = solve_triangular(GP.l_mat, k1_v, lower=True)
        mean_diff = np.matmul(k1_v, GP.alpha)
        bond_means = mean_diff
        
        for b2, r2 in enumerate(bond_lengths):
            bond2 = np.array([r2, 1.0, 0.0, 0.0])
            env2.bond_array_2 = np.array([bond2])
#            env2.cross_bond_dists = np.array([[0]])
            
            k2_v = GP.get_kernel_vector(env2, 1)
            v2_vec = solve_triangular(GP.l_mat, k2_v, lower=True)
            self_kern = GP.kernel(env1, env2, 1, 1, GP.hyps, GP.cutoffs)
            var_diff = self_kern - np.matmul(v1_vec, v2_vec)
            bond_vars[b2] = var_diff   
                      
        return b1, bond_means, bond_vars

    #@profile
    def GenGrid_svd(self, GP, bond_struc, processes=mp.cpu_count()):
    
        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''

        # ------ change GP kernel to 2 body ------
        GP.kernel = two_body
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
        A_list = pool.map(self._GenGrid_svd_inner, pool_list)
        for p in range(nop):
            bond_means[p] = A_list[p][0]
            bond_vars[p, :] = A_list[p][1]
        pool.close()
        pool.join()

        # ------ change back original GP ------
        GP.cutoffs = original_cutoffs
        GP.hyps = original_hyps
        GP.kernel = two_plus_three_body
       
        return bond_means, bond_vars

    def _GenGrid_svd_inner(self, params):
    
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
        if self.bodies == '2':
            self.var = SplinesInterpolation(y_var, 
                        u_bounds=np.array([self.u_bound, self.u_bound]), 
                        l_bounds=np.array([self.l_bound, self.l_bound]), 
                        orders=np.array([self.grid_num, self.grid_num]))
        elif self.bodies == '2+3':
            self.var = PCASplines(y_var, u_bounds=np.array(self.u_bound), 
                       l_bounds=np.array(self.l_bound), 
                       orders=np.array([self.grid_num]), 
                       svd_rank=self.svd_rank, load_svd=None)


    def predict(self, atom_env, GP, mean_only):
    
        '''
        predict for an atom environment
        param: atom_env: ChemicalEnvironment
        return force on an atom with its variance
        '''
        
        bond_lengths = atom_env.bond_array_2[:,0]
        bond_dirs = atom_env.bond_array_2[:,1:]
        bond_num = len(bond_lengths)
       
        bond_lengths = np.expand_dims(bond_lengths, axis=1)
        mean_diffs = self.mean(bond_lengths)
        bond_forces = [mean_diffs*bond_dirs[:,i] for i in range(3)]
        atom_mean = np.sum(bond_forces, axis=1)
        
        atom_var = np.zeros(3)
        if not mean_only:
            if self.bodies == '2':
                ind_1, ind_2 = np.meshgrid(np.arange(bond_num), np.arange(bond_num))
                ind_1 = np.reshape(ind_1, (ind_1.shape[0]*ind_1.shape[1], 1)) 
                ind_2 = np.reshape(ind_2, (ind_2.shape[0]*ind_2.shape[1], 1)) 
                bond_1, bond_2 = (bond_lengths[ind_1], bond_lengths[ind_2])        
                bond_xyz1 = bond_dirs[ind_1,:] 
                bond_xyz2 = bond_dirs[ind_2,:] 
                bond_concat = np.concatenate([bond_1, bond_2], axis=1)  
                var_diffs = self.var(bond_concat)
                var_diffs = np.repeat(np.expand_dims(var_diffs, axis=1), 3, axis=1)
                atom_var = np.sum(var_diffs*bond_xyz1[:,0,:]*bond_xyz2[:,0,:], axis=0) 
                return atom_mean, atom_var    
            elif self.bodies == '2+3':
                sig_2, ls_2, sig_3, ls_3, noise = GP.hyps
                LambdaU = self.var(bond_lengths)
                VLambdaU = self.var.V @ LambdaU
                v = VLambdaU @ bond_dirs
                self_kern = np.zeros(3)
                for d in range(3):
                    self_kern[d] = self_two_body_jit(atom_env.bond_array_2, d+1, 
                           sig_2, ls_2, GP.cutoffs[0], quadratic_cutoff)
                return atom_mean, self_kern, v
        else:
            if self.bodies == '2':
                return atom_mean, atom_var
            elif self.bodies == '2+3':
                return atom_mean, 0, 0



class Map3body:
   
    def __init__(self, grid_num, bounds, GP, bond_struc, bodies='3', 
                load_grid=False, load_svd=None, svd_rank=0, update=True): 
    
        '''
        param grids: the 1st element is the number of grids for mean prediction, 
                    the 2nd is for var
        '''       
        
        self.grid_num = grid_num
        self.l_bound, self.u_bound = bounds
        self.cutoffs = GP.cutoffs
        self.bodies = bodies
        if not load_grid:
            y_mean, y_var = self.GenGrid(GP, bond_struc, update)
        else:
            y_mean = np.load('grid3_mean.npy')
            y_var = np.load('grid3_var.npy')
#            y_mean, y_var = utils.merge(load_grid, noa, nop)
        self.build_map(y_mean, y_var, svd_rank=svd_rank, load_svd=load_svd) 


    #@profile
    def GenGrid(self, GP, bond_struc, update, processes=1):#, processes=mp.cpu_count()):
    
        '''
        generate grid data of mean prediction and L^{-1}k* for each triplet
         implemented in a parallelized style
        '''
        # ------ change GP kernel to 3 body ------
        original_kernel = GP.kernel
        original_hyps = np.copy(GP.hyps)
        GP.kernel = three_body
        GP.hyps = [GP.hyps[-3], GP.hyps[-2], GP.hyps[-1]]

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
        if not update:
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
        noa = self.grid_num[2]
        angle12 = angle12
        bond_means = np.zeros([nop, nop])
        bond_vars = np.zeros([nop, nop, len(GP.alpha)])

        # open saved k vector file, and write to new file
        kv_filename = 'kv3/'+str(a12)
        size = len(GP.training_data) * 3
        new_kv_file = np.zeros((nop**2+1, size))
        new_kv_file[0,0] = size

        if update:
            old_kv_file = np.load(kv_filename+'.npy') 
            last_size = int(old_kv_file[0,0])
            new_kv_file[:, :last_size] = old_kv_file
            ds = [1, 2, 3]

        for b1, r1 in enumerate(bond_lengths):
            r1 = bond_lengths[b1]
            for b2, r2 in enumerate(bond_lengths):
                x2 = r2 * np.cos(angle12)
                y2 = r2 * np.sin(angle12)
                r12 = np.linalg.norm(np.array([x2-r1, y2, 0]))

                env12.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])
                env12.cross_bond_dists = np.array([[0, r12], [r12, 0]])

                if update:
                    # calculate kernel functions of those newly added training data
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
                v12_vec = solve_triangular(GP.l_mat, k12_v, lower=True)
                mean_diff = np.matmul(k12_v, GP.alpha)
                bond_means[b1, b2] = mean_diff
                bond_vars[b1, b2, :] = v12_vec  
                      
        # replace the old file with the new file
        np.save(kv_filename, new_kv_file)

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
       
    def predict(self, atom_env, GP, mean_only):

        '''
        predict for an atom environment
        param: atom_env: ChemicalEnvironment
        return force on an atom with its variance
        '''
        t0 = time.time()
        bond_array = atom_env.bond_array_3
        cross_bond_inds = atom_env.cross_bond_inds
        cross_bond_dists = atom_env.cross_bond_dists
        triplets = atom_env.triplet_counts
        tri_12, tri_21, xyz_1s, xyz_2s = get_triplets(bond_array, 
            cross_bond_inds, cross_bond_dists, triplets)
        tri_12 = np.array(tri_12)
        tri_21 = np.array(tri_21)
        xyz_1s = np.array(xyz_1s)
        xyz_2s = np.array(xyz_2s)
        #print('\nget triplets', time.time()-t0)       

        # predict mean
        t0 = time.time()
        f0_12 = self.mean(tri_12)
        f0_21 = self.mean(tri_21)
        f12 = np.diag(f0_12) @ xyz_1s
        f21 = np.diag(f0_21) @ xyz_2s
        mgp_f = np.sum(f12 + f21, axis=0)
        #print('mean', time.time()-t0)

        # predict var        
        mgp_v = np.zeros(3)
        if not mean_only:
            t0 = time.time()
            self_kern = np.zeros(3)
            if self.bodies == '3':
                sig, ls, noise = GP.hyps
            elif self.bodies == '2+3':
                sig2, ls2, sig, ls, noise = GP.hyps
            r_cut = GP.cutoffs[1]
            for d in range(3):
                self_kern[d] = self_three_body_jit(bond_array,
                       cross_bond_inds, 
                       cross_bond_dists,
                       triplets, 
                       d+1, sig, ls, r_cut, quadratic_cutoff)
         #   print('self kern', time.time()-t0, ',value:', self_kern)

            t0 = time.time()
            v0_12 = self.var(tri_12)
            v0_21 = self.var(tri_21)
            v12 = v0_12 @ xyz_1s
            v21 = v0_21 @ xyz_2s
            v = v12 + v21
            if self.bodies == '3':
                mgp_v = - np.sum(v ** 2, axis=0) + self_kern
                return mgp_f, mgp_v
            elif self.bodies == '2+3':
                v = self.var.V @ v
                return mgp_f, self_kern, v
          #  print('var', time.time()-t0, ',value:', mgp_v)        
        else: 
            if self.bodies == '3':
                return mgp_f, mgp_v
            elif self.bodies == '2+3':
                return mgp_f, 0, 0

@njit
def get_triplets(bond_array, cross_bond_inds, 
                 cross_bond_dists, triplets):
    num = np.sum(triplets)
    tris1 = np.zeros((num,3))
    tris2 = np.zeros((num,3))
    tri_dir1 = np.zeros((num,3))
    tri_dir2 = np.zeros((num,3))

    k = 0
    for m in range(bond_array.shape[0]):
        r1 = bond_array[m, 0]
        c1 = bond_array[m, 1:]

        for n in range(triplets[m]):
            ind1 = cross_bond_inds[m, m+n+1]
            r2 = bond_array[ind1, 0]
            c2 = bond_array[ind1, 1:]
            a12 = np.arccos(np.sum(c1*c2))

            tris1[k] = np.array((r1, r2, a12))
            tris2[k] = np.array((r2, r1, a12))
            tri_dir1[k] = c1
            tri_dir2[k] = c2
            k += 1
    return tris1, tris2, tri_dir1, tri_dir2 


@njit
def quadratic_cutoff(r_cut, ri, ci):
    rdiff = r_cut - ri
    fi = rdiff * rdiff
    fdi = 2 * rdiff * ci

    return fi, fdi

@njit
def self_two_body_jit(bond_array, d, sig, ls,
                      r_cut, cutoff_func):
    kern = 0

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig*sig

    for m in range(bond_array.shape[0]):
        ri = bond_array[m, 0]
        ci = bond_array[m, d]
        fi, fdi = cutoff_func(r_cut, ri, ci)

        for n in range(m, bond_array.shape[0]):
            rj = bond_array[n, 0]
            cj = bond_array[n, d]
            fj, fdj = cutoff_func(r_cut, rj, cj)
            r11 = ri - rj

            A = ci * cj
            B = r11 * ci
            C = r11 * cj
            D = r11 * r11

            if m == n:
                kern += force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2,
                                 ls3, sig2)
            else:
                kern += 2*force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2,
                                 ls3, sig2)
    return kern


@njit
def self_three_body_jit(bond_array, cross_bond_inds, 
                   cross_bond_dists, triplets,
                   d, sig, ls, r_cut, cutoff_func):
    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig*sig
    ls1 = 1 / (2*ls*ls)
    ls2 = 1 / (ls*ls)
    ls3 = ls2*ls2
    
    for m in range(bond_array.shape[0]):
        ri1 = bond_array[m, 0]
        ci1 = bond_array[m, d]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)

        for n in range(triplets[m]):
            ind1 = cross_bond_inds[m, m+n+1]
            ri2 = bond_array[ind1, 0]
            ci2 = bond_array[ind1, d]
            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)

            ri3 = cross_bond_dists[m, m+n+1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            fi = fi1*fi2*fi3
            fdi = fdi1*fi2*fi3+fi1*fdi2*fi3

            for p in range(m, bond_array.shape[0]):
                rj1 = bond_array[p, 0]
                cj1 = bond_array[p, d]
                fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)

                for q in range(triplets[p]):
                    ind2 = cross_bond_inds[p, p+1+q]
                    rj2 = bond_array[ind2, 0]
                    cj2 = bond_array[ind2, d]
                    fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)

                    rj3 = cross_bond_dists[p, p+1+q]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)

                    fj = fj1*fj2*fj3
                    fdj = fdj1*fj2*fj3+fj1*fdj2*fj3

                    tri_kern = triplet_kernel(ci1, ci2, cj1, cj2, ri1, ri2, ri3,
                                           rj1, rj2, rj3, fi, fj, fdi, fdj,
                                           ls1, ls2, ls3, sig2)
                    if p == m:
                        kern += tri_kern
                    else:
                        kern += 2 * tri_kern

    return kern


@njit
def triplet_kernel(ci1, ci2, cj1, cj2, ri1, ri2, ri3, rj1, rj2, rj3, fi, fj,
                   fdi, fdj, ls1, ls2, ls3, sig2):
    r11 = ri1-rj1
    r12 = ri1-rj2
    r13 = ri1-rj3
    r21 = ri2-rj1
    r22 = ri2-rj2
    r23 = ri2-rj3
    r31 = ri3-rj1
    r32 = ri3-rj2
    r33 = ri3-rj3

    # sum over all six permutations
    M1 = three_body_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M2 = three_body_helper_2(ci2, ci1, cj2, cj1, r21, r13, r32, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M3 = three_body_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M4 = three_body_helper_1(ci1, ci2, cj2, cj1, r12, r21, r33, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M5 = three_body_helper_2(ci2, ci1, cj1, cj2, r22, r13, r31, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M6 = three_body_helper_2(ci1, ci2, cj2, cj1, r11, r23, r32, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)

    return M1 + M2 + M3 + M4 + M5 + M6


@njit
def three_body_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33,
                        fi, fj, fdi, fdj,
                        ls1, ls2, ls3, sig2):
    A = ci1*cj1+ci2*cj2
    B = r11*ci1+r22*ci2
    C = r11*cj1+r22*cj2
    D = r11*r11+r22*r22+r33*r33

    M = force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2)

    return M


@njit
def three_body_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31,
                        fi, fj, fdi, fdj,
                        ls1, ls2, ls3, sig2):
    A = ci1*cj2
    B = r12*ci1+r23*ci2
    C = r12*cj2+r31*cj1
    D = r12*r12+r23*r23+r31*r31

    M = force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2)

    return M


@njit
def force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2):
    E = exp(-D * ls1)
    F = B * fi
    G = -C * fj
    I = fdi * fdj
    J = F * fdj
    K = G * fdi
    L = A * fi * fj + F * G * ls2
    M = sig2 * (I + (J + K + L) * ls2) * E

    return M


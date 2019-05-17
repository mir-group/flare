import time
from math import exp
import numpy as np
from numba import njit
from scipy.linalg import solve_triangular
import multiprocessing as mp
import sys
sys.path.append('../../flare/')

import flare.gp as gp
import flare.env as env
from flare.kernels import two_body, three_body, two_plus_three_body, two_body_jit
import flare.struc as struc
from flare.mc_simple import two_body_mc, three_body_mc, two_plus_three_body_mc

import flare.mff.utils as utils
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
        self.bodies = str(grid_params['bodies'])
        self.grid_num_2 = grid_params['grid_num_2']
        self.bounds_2 = grid_params['bounds_2']
        self.grid_num_3 = grid_params['grid_num_3']
        self.bounds_3 = grid_params['bounds_3']
        self.svd_rank = grid_params['svd_rank']
        
        bond_struc, spcs = self.build_bond_struc(struc_params)
        self.spcs = spcs
        self.maps_2 = []
        self.maps_3 = []
        if self.bodies in ['2', '2+3']:
            for b_struc in bond_struc[0]:
                map_2 = Map2body(self.grid_num_2, self.bounds_2, self.GP, b_struc,
                         self.bodies, grid_params['load_grid'], self.svd_rank)
                self.maps_2.append(map_2)
        if self.bodies in ['3', '2+3']:
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
        bodies = 3
        bond_struc_3 = []
        spc_3 = []
#        for spc1_ind in range(N_spc):
#            spc1 = species_list[spc1_ind]
#            for spc2_ind in range(spc1_ind, N_spc):
#                spc2 = species_list[spc2_ind]
#                for spc3_ind in range(spc2_ind, N_spc):
#                    spc3 = species_list[spc3_ind]
#                    species = [spc1, spc2, spc3]
#                    spc_3.append(species)
#                    positions = [[(i+1)/(bodies+1)*cutoff, 0, 0] \
#                                for i in range(bodies)]
#                    bond_struc_3.append(struc.Structure(cell, species, positions, mass_dict))
#                    if spc2_ind != spc3_ind: # for non-equivalent atoms, no permutation symmetry
#                        species_flip = [spc1, spc3, spc2]
#                        spc_3.append(species_flip)
#                        bond_struc_3.append(struc.Structure(cell, species_flip, positions, mass_dict))

        bond_struc = [bond_struc_2, bond_struc_3]
        spcs = [spc_2, spc_3]
        return bond_struc, spcs

    def predict(self, atom_env, mean_only=False):
        f2_spcs = 0
        kern2_spcs = 0
        v2_spcs = 0
        f3_spcs = 0
        kern3_spcs = 0
        v3_spcs = 0
        ctype = atom_env.ctype
        etypes = atom_env.etypes 
        if self.bodies in ['2', '2+3']:
            bond_array_2 = atom_env.bond_array_2
            spc_2, spc_bonds_2 = get_bonds(bond_array_2, etypes)
            for i, spc in enumerate(spc_2):
                ce_bonds = np.array(spc_bonds_2[i])
                ce_spc = [ctype, spc]
                ce_spc.sort()
                map_ind = self.spcs[0].index(ce_spc)
                f2, kern2, v2 = self.maps_2[map_ind].predict(ce_bonds, self.GP, mean_only)
                f2_spcs += f2
                kern2_spcs += kern2
                v2_spcs += v2

        elif self.bodies in ['3', '2+3']:
            bond_array_3 = atom_env.bond_array_3
            spc_3, tris, tris_dir = get_triplets(bond_array_3, cross_bond_inds, 
                cross_bond_dists, triplets, coded_species)
            for i, spc in enumerate(spc_3):
                tr1 = tris1[i]
                tr2 = tris2[i]
                tr_d1 = tri_dir1[i]
                tr_d2 = tri_dir2[i]
                ce_spc = [ctype]+spc
                ce_spc = ce_spc.sort()
                map_ind = self.spcs[1].index(ce_spc)
                f3, kern3, v3 = self.maps_3[map_ind].predict(tr1, tr2, tr_d1, tr_d2, self.GP, mean_only)
                f3_spcs += f3
                kern3_spcs += kern3
                v3_spcs += v3

        f = f2_spcs + f3_spcs
        v = kern2_spcs + kern3_spcs - np.sum((v2_spcs + v3_spcs)**2, axis=0)
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

    def predict(self, bond_array_2, GP, mean_only):
    
        '''
        predict for an atom environment
        param: atom_env: ChemicalEnvironment
        return force on an atom with its variance
        '''
        # ----------- prepare bonds for prediction ---------------
        bond_lengths = np.expand_dims(bond_array_2[:,0], axis=1)
        bond_dirs = bond_array_2[:,1:]
        bond_num = len(bond_lengths)
       
        # ----------- predict mean/force ------------------
        mean_diffs = self.mean(bond_lengths)
        bond_forces = [mean_diffs*bond_dirs[:,i] for i in range(3)]
        atom_mean = np.sum(bond_forces, axis=1)
        
        # ----------- predict variance --------------------
        self_kern = np.zeros(3)
        v = np.zeros(3)
        if not mean_only:
            sig_2 = GP.hyps[0]
            ls_2 = GP.hyps[1]
            LambdaU = self.var(bond_lengths)
            VLambdaU = self.var.V @ LambdaU
            v = VLambdaU @ bond_dirs
            for d in range(3):
                self_kern[d] = self_two_body_jit(bond_array_2, d+1, 
                       sig_2, ls_2, GP.cutoffs[0], quadratic_cutoff)
        return atom_mean, self_kern, v



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
        if self.bodies == '2+3':
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
        mff_f = np.sum(f12 + f21, axis=0)
        #print('mean', time.time()-t0)

        # predict var        
        mff_v = np.zeros(3)
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
                mff_v = - np.sum(v ** 2, axis=0) + self_kern
                return mff_f, mff_v
            elif self.bodies == '2+3':
                v = self.var.V @ v
                return mff_f, self_kern, v
          #  print('var', time.time()-t0, ',value:', mff_v)        
        else: 
            if self.bodies == '3':
                return mff_f, mff_v
            elif self.bodies == '2+3':
                return mff_f, 0, 0

@njit
def get_bonds(bond_array, coded_species): # get bonds list for specified species
    exist_species = []
    species_bonds = []
    for i in range(len(bond_array)):
        bond = bond_array[i]
        spec = coded_species[i]
        if spec in exist_species:
            ind = exist_species.index(spec)
            species_bonds[ind].append(bond)
        else:
            exist_species.append(spec)
            species_bonds.append([bond])
    return exist_species, species_bonds

@njit
def get_triplets(bond_array, cross_bond_inds, 
                 cross_bond_dists, triplets, coded_species):
    exist_species = []
    tris1 = []
    tris2 = []
    tri_dir1 = []
    tri_dir2 = []

    for m in range(bond_array.shape[0]):
        r1 = bond_array[m, 0]
        c1 = bond_array[m, 1:]
        spc1 = coded_species[m]

        for n in range(triplets[m]):
            ind1 = cross_bond_inds[m, m+n+1]
            r2 = bond_array[ind1, 0]
            c2 = bond_array[ind1, 1:]
            a12 = np.arccos(np.sum(c1*c2))
            spc2 = coded_species[ind1]
            
            spcs = [spc1, spc2]
            sort_ind = np.argsort(spcs.sort())
            spcs = spcs[sort_ind]
            if spec in exist_species:
                k = exist_species.index(spcs)
                tris1[k].append((r1, r2, a12))
                tris2[k].append((r2, r1, a12))
                tri_dir1[k].append(c1)
                tri_dir2[k].append(c2)
            else:
                exist_species.append(spcs)
                tris1.append([(r1, r2, a12)])
                tris2.append([(r2, r1, a12)])
                tri_dir1.append([c1])
                tri_dir2.append([c2])

    return exist_species, tris1, tris2, tri_dir1, tri_dir2 


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


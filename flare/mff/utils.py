import numpy as np
from numba import njit
import io
import sys
sys.path.append('../flare')
import os

import flare.gp as gp
import flare.env as env
import flare.struc as struc
import flare.kernels as kernels
import flare.modules.qe_parsers as qe_parsers
import flare.modules.analyze_gp as analyze_gp
from flare.env import AtomicEnvironment
from flare.kernels import triplet_kernel, three_body_helper_1, three_body_helper_2, force_helper
from flare.cutoffs import quadratic_cutoff

import time
import random
import logging
import multiprocessing as mp
import concurrent.futures
import cProfile
       
def save_GP(GP, prefix):
    np.save(prefix+'alpha', GP.alpha)
    np.save(prefix+'hyps', GP.hyps)
    np.save(prefix+'l_mat', GP.l_mat)
    
def load_GP(GP, prefix):
    GP.alpha = np.load(prefix+'alpha.npy')
    GP.hyps = np.load(prefix+'hyps.npy')
    GP.l_mat = np.load(prefix+'l_mat.npy')
    l_mat_inv = np.linalg.inv(GP.l_mat)
    GP.ky_mat_inv = l_mat_inv.T @ l_mat_inv
    
def save_grid(bond_lens, bond_ens_diff, bond_vars_diff, prefix):
    np.save(prefix+'-bond_lens', bond_lens)
    np.save(prefix+'-bond_ens_diff', bond_ens_diff)
    np.save(prefix+'-bond_vars_diff', bond_vars_diff) 
  
def load_grid(prefix):
    bond_lens = np.load(prefix+'bond_lens.npy')
    bond_ens_diff = np.load(prefix+'bond_ens_diff.npy')
    bond_vars_diff = np.load(prefix+'bond_vars_diff.npy')  
    return bond_lens, bond_ens_diff, bond_vars_diff

def merge(prefix, a_num, g_num):
    grid_means = np.zeros((g_num, g_num, a_num))
    grid_vars = np.zeros((g_num, g_num, a_num, g_num, g_num, a_num))
    for a12 in range(a_num):
        grid_means[:,:,a12] = np.load(prefix+str((a12, 0))+'-bond_means.npy')
        for a34 in range(a_num):
            grid_vars[:,:,a12,:,:,a34] = np.load(prefix+str((a12, a34))+'-bond_vars.npy')
    return grid_means, grid_vars
    
def svd_grid(matr, rank=55, prefix=None):
    if not prefix:
        u, s, vh = np.linalg.svd(matr, full_matrices=False)
#        np.save('../params/SVD_U', u)
#        np.save('../params/SVD_S', s)
    else:
        u = np.load(prefix+'SVD_U.npy')
        s = np.load(prefix+'SVD_S.npy')
    return u[:,:rank], s[:rank], vh[:rank, :]

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




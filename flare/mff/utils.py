import numpy as np
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



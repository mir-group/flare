import numpy as np
import math
from math import exp
from math import factorial
from itertools import combinations
from itertools import permutations
from numba import njit
from struc import Structure
from env import ChemicalEnvironment
import gp
import struc
import env
import time


def two_body_smooth(env1, env2, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]
    d = hyps[2]

    return two_body_smooth_jit(env1.bond_array, env2.bond_array,
                               d1, d2, sig, ls, r_cut, d)


def two_body_smooth_grad(env1, env2, d1, d2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]
    d = hyps[2]

    kernel, ls_derv, sig_derv, d_derv = \
        two_body_smooth_grad_jit(env1.bond_array, env2.bond_array,
                                 d1, d2, sig, ls, r_cut, d)
    kernel_grad = np.array([sig_derv, ls_derv, d_derv])
    return kernel, kernel_grad


def two_body_smooth_en(env1, env2, hyps, r_cut):
    sig = hyps[0]
    ls = hyps[1]
    d = hyps[2]

    kernel = two_body_smooth_en_jit(env1.bond_array, env2.bond_array, sig, ls,
                                    r_cut, d)

    return kernel


@njit
def two_body_smooth_grad_jit(bond_array_1, bond_array_2, d1, d2, sig, ls,
                             r_cut, d):

    kern = 0
    sig_derv = 0
    ls_derv = 0
    d_derv = 0

    for m in range(bond_array_1.shape[0]):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]

        # check if r1 is in the smoothed region
        if r1 > r_cut - d:
            fcut1 = (1/2) * (np.cos(np.pi * (r1-r_cut+d) / d) + 1)
            fderv1 = (np.pi/(2*d)) * np.sin(np.pi*(r_cut-r1) / d) * coord1
            dterm11 = (np.pi*(r1-r_cut)/(2*d*d))*np.sin(np.pi*(r_cut-r1) / d)
            dterm12 = (np.pi*coord1/(2*d**3)) * \
                (np.pi*(r1-r_cut) *
                 np.cos(np.pi * (r1-r_cut) / d) +
                 d*np.sin(np.pi*(r1-r_cut) / d))
        else:
            fcut1 = 1
            fderv1 = 0
            dterm11 = 0
            dterm12 = 0

        for n in range(bond_array_2.shape[0]):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]

            # check if r2 is in the smoothed region
            if r2 > r_cut - d:
                fcut2 = (1/2) * (np.cos(np.pi * (r2-r_cut+d) / d) + 1)
                fderv2 = (np.pi/(2*d)) * np.sin(np.pi*(r_cut-r2) / d) * coord2
                dterm21 = (np.pi*(r2-r_cut)/(2*d*d)) * \
                    np.sin(np.pi*(r_cut-r2) / d)
                dterm22 = (np.pi*coord2/(2*d**3)) * \
                    (np.pi*(r2-r_cut) *
                     np.cos(np.pi * (r2-r_cut) / d) +
                     d*np.sin(np.pi*(r2-r_cut) / d))
            else:
                fcut2 = 1
                fderv2 = 0
                dterm21 = 0
                dterm22 = 0

            rdiff = r1 - r2

            A_cp = coord1 * coord2
            B_cp_1 = rdiff * coord1
            B_cp_2 = -rdiff * coord2
            B_cp = rdiff * coord1 * rdiff * coord2
            C_cp = rdiff * rdiff

            # kernel terms
            kern0 = sig*sig*exp(-C_cp / (2*ls*ls))
            kern1 = kern0 * B_cp_1 / (ls*ls)
            kern2 = kern0 * B_cp_2 / (ls*ls)
            kern3 = kern0 * (A_cp*ls*ls-B_cp) / (ls**4)

            # ls derv terms
            ls0 = kern0*rdiff*rdiff/(ls*ls*ls)
            ls1 = kern0*coord1*(-2*ls*ls+rdiff*rdiff)*rdiff/(ls**5)
            ls2 = -kern0*coord2*(-2*ls*ls+rdiff*rdiff)*rdiff/(ls**5)
            ls3 = kern0*coord1*coord2 * \
                (-2*ls**4+5*ls*ls*rdiff*rdiff-rdiff**4)/(ls**7)

            kern_val = kern3*fcut1*fcut2 + kern2*fderv1*fcut2 +\
                kern1*fderv2*fcut1 + kern0*fderv1*fderv2

            kern += kern_val
            ls_derv += ls3*fcut1*fcut2 + ls2*fderv1*fcut2 +\
                ls1*fderv2*fcut1 + ls0*fderv1*fderv2
            sig_derv += 2*kern_val/sig
            d_derv += kern3*(dterm11*fcut2+fcut1*dterm21) + \
                kern2*(dterm12*fcut2+fderv1*dterm21) + \
                kern1*(dterm11*fderv2+fcut1*dterm22) + \
                kern0*(dterm12*fderv2+fderv1*dterm22)

    return kern, ls_derv, sig_derv, d_derv


@njit
def two_body_smooth_jit(bond_array_1, bond_array_2, d1, d2, sig, ls,
                        r_cut, d):
    kern = 0

    for m in range(bond_array_1.shape[0]):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]

        # check if r1 is in the smoothed region
        if r1 > r_cut - d:
            fcut1 = (1/2) * (np.cos(np.pi * (r1-r_cut+d) / d) + 1)
            fderv1 = (np.pi/(2*d)) * np.sin(np.pi*(r_cut-r1) / d) * coord1
        else:
            fcut1 = 1
            fderv1 = 0

        for n in range(bond_array_2.shape[0]):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]

            # check if r2 is in the smoothed region
            if r2 > r_cut - d:
                fcut2 = (1/2) * (np.cos(np.pi * (r2-r_cut+d) / d) + 1)
                fderv2 = (np.pi/(2*d)) * np.sin(np.pi*(r_cut-r2) / d) * coord2
            else:
                fcut2 = 1
                fderv2 = 0

            rdiff = r1 - r2

            A_cp = coord1 * coord2
            B_cp_1 = rdiff * coord1
            B_cp_2 = -rdiff * coord2
            B_cp = rdiff * coord1 * rdiff * coord2
            C_cp = rdiff * rdiff

            kern0 = sig*sig*exp(-C_cp / (2*ls*ls))
            kern1 = kern0 * B_cp_1 / (ls*ls)
            kern2 = kern0 * B_cp_2 / (ls*ls)
            kern3 = kern0 * (A_cp*ls*ls-B_cp) / (ls**4)

            kern += kern3*fcut1*fcut2 + kern2*fderv1*fcut2 +\
                kern1*fderv2*fcut1 + kern0*fderv1*fderv2
    return kern


@njit
def two_body_smooth_en_jit(bond_array_1, bond_array_2, sig, ls,
                           r_cut, d):
    kern = 0

    for m in range(bond_array_1.shape[0]):
        r1 = bond_array_1[m, 0]

        # check if r1 is in the smoothed region
        if r1 > r_cut - d:
            fcut1 = (1/2) * (np.cos(np.pi * (r1-r_cut+d) / d) + 1)
        else:
            fcut1 = 1

        for n in range(bond_array_2.shape[0]):
            r2 = bond_array_2[n, 0]

            # check if r2 is in the smoothed region
            if r2 > r_cut - d:
                fcut2 = (1/2) * (np.cos(np.pi * (r2-r_cut+d) / d) + 1)
            else:
                fcut2 = 1

            rdiff = r1 - r2
            C_cp = rdiff * rdiff
            kern0 = sig*sig*exp(-C_cp / (2*ls*ls))
            kern += kern0*fcut1*fcut2

    return kern


if __name__ == '__main__':
    # create env 1
    delt = 1e-5
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5]))

    positions_1 = [np.array([0, 0, 0]),
                   np.array([0.1, 0.2, 0.3]),
                   np.array([0.3, 0.2, 0.1])]
    positions_2 = [np.array([delt, 0, 0]),
                   np.array([0.1, 0.2, 0.3]),
                   np.array([0.3, 0.2, 0.1])]
    positions_3 = [np.array([-delt, 0, 0]),
                   np.array([0.1, 0.2, 0.3]),
                   np.array([0.3, 0.2, 0.1])]

    species_1 = ['A', 'B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
    test_structure_2 = struc.Structure(cell, species_1, positions_2, cutoff)
    test_structure_3 = struc.Structure(cell, species_1, positions_3, cutoff)

    env1_1 = env.ChemicalEnvironment(test_structure_1, atom_1)
    env1_2 = env.ChemicalEnvironment(test_structure_2, atom_1)
    env1_3 = env.ChemicalEnvironment(test_structure_3, atom_1)

    # create env 2
    positions_1 = [np.array([0, 0, 0]),
                   np.array([0.25, 0.3, 0.4]),
                   np.array([0.4, 0.3, 0.25])]
    positions_2 = [np.array([0, delt, 0]),
                   np.array([0.25, 0.3, 0.4]),
                   np.array([0.4, 0.3, 0.25])]
    positions_3 = [np.array([0, -delt, 0]),
                   np.array([0.25, 0.3, 0.4]),
                   np.array([0.4, 0.3, 0.25])]

    species_2 = ['A', 'A', 'B']
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1, cutoff)
    test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
    test_structure_3 = struc.Structure(cell, species_2, positions_3, cutoff)

    env2_1 = env.ChemicalEnvironment(test_structure_1, atom_2)
    env2_2 = env.ChemicalEnvironment(test_structure_2, atom_2)
    env2_3 = env.ChemicalEnvironment(test_structure_3, atom_2)

    sig = 1
    ls = 1
    d = 0.1
    d1 = 1
    d2 = 2

    hyps = np.array([sig, ls, d])

    # check force kernel
    calc1 = two_body_smooth_en(env1_2, env2_2, hyps, cutoff)
    calc2 = two_body_smooth_en(env1_3, env2_3, hyps, cutoff)
    calc3 = two_body_smooth_en(env1_2, env2_3, hyps, cutoff)
    calc4 = two_body_smooth_en(env1_3, env2_2, hyps, cutoff)

    kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delt**2)
    kern_analytical = two_body_smooth(env1_1, env2_1, d1, d2, hyps, cutoff)

    tol = 1e-4
    assert(np.isclose(kern_finite_diff, kern_analytical, tol))

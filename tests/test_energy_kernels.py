import pytest
import numpy as np
import sys
from random import random
from copy import deepcopy
from numpy.random import rand
sys.path.append('../otf_engine')
import env
import gp
import struc
import kernels
from kernels import n_body_sc_grad, n_body_mc_grad
from energy_conserving_kernels import three_body_cons_quad, \
    three_body_cons_quad_en


def test_three_body_cons():
    # create env 1
    delt = 1e-4
    cell = np.eye(3)
    cutoff = 1

    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][0] = delt

    positions_3 = deepcopy(positions_1)
    positions_3[0][0] = -delt

    species_1 = ['A', 'B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
    test_structure_2 = struc.Structure(cell, species_1, positions_2, cutoff)
    test_structure_3 = struc.Structure(cell, species_1, positions_3, cutoff)

    env1_1 = env.ChemicalEnvironment(test_structure_1, atom_1)
    env1_2 = env.ChemicalEnvironment(test_structure_2, atom_1)
    env1_3 = env.ChemicalEnvironment(test_structure_3, atom_1)

    # create env 2
    positions_1 = [np.array([0., 0., 0.]),
                   np.array([random(), random(), random()]),
                   np.array([random(), random(), random()])]
    positions_2 = deepcopy(positions_1)
    positions_2[0][1] = delt
    positions_3 = deepcopy(positions_1)
    positions_3[0][1] = -delt

    species_2 = ['A', 'A', 'B']
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_1, cutoff)
    test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
    test_structure_3 = struc.Structure(cell, species_2, positions_3, cutoff)

    env2_1 = env.ChemicalEnvironment(test_structure_1, atom_2)
    env2_2 = env.ChemicalEnvironment(test_structure_2, atom_2)
    env2_3 = env.ChemicalEnvironment(test_structure_3, atom_2)

    sig = 1
    ls = 0.1
    d1 = 1
    d2 = 2
    bodies = None

    hyps = np.array([sig, ls])

    # check force kernel
    calc1 = three_body_cons_quad_en(env1_2, env2_2, bodies, hyps, cutoff)
    calc2 = three_body_cons_quad_en(env1_3, env2_3, bodies, hyps, cutoff)
    calc3 = three_body_cons_quad_en(env1_2, env2_3, bodies, hyps, cutoff)
    calc4 = three_body_cons_quad_en(env1_3, env2_2, bodies, hyps, cutoff)

    kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delt**2)
    kern_analytical = three_body_cons_quad(env1_1, env2_1, bodies,
                                           d1, d2, hyps, cutoff)

    tol = 1e-4
    assert(np.isclose(kern_finite_diff, kern_analytical, tol))

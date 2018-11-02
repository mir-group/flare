"""
Chemical environment test suite

Jon V
"""

import pytest
import numpy as np
import sys
import time
sys.path.append('../otf_engine')
import env
import struc


# -----------------------------------------------------------------------------
#                              functions
# -----------------------------------------------------------------------------

# test jit and python two body kernels
def kernel_performance(env1, env2, d1, d2, sig, ls, kernel, its):
    # warm up jit
    time0 = time.time()
    kern_val = kernel(env1, env2, d1, d2, sig, ls)
    time1 = time.time()
    warm_up_time = time1 - time0

    # test run time performance
    time2 = time.time()
    for n in range(its):
        kernel(env1, env2, d1, d2, sig, ls)
    time3 = time.time()
    run_time = (time3 - time2) / its

    return kern_val, run_time, warm_up_time


def get_jit_speedup(env1, env2, d1, d2, sig, ls, jit_kern, py_kern,
                    its):

    kern_val_jit, run_time_jit, warm_up_time_jit = \
        kernel_performance(env1, env2, d1, d2, sig, ls, jit_kern, its)

    kern_val_py, run_time_py, warm_up_time_py = \
        kernel_performance(env1, env2, d1, d2, sig, ls, py_kern, its)

    speed_up = run_time_py / run_time_jit

    return speed_up, kern_val_jit, kern_val_py, warm_up_time_jit,\
        warm_up_time_py


def get_random_structure(cell, unique_species, cutoff, noa):
    positions = []
    forces = []
    species = []
    for n in range(noa):
        positions.append(np.random.uniform(-1, 1, 3))
        forces.append(np.random.uniform(-1, 1, 3))
        species.append(unique_species[np.random.randint(0, 2)])

    test_structure = struc.Structure(cell, species, positions, cutoff)

    return test_structure, forces


# -----------------------------------------------------------------------------
#                                fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope='module')
def test_env():
    # create test structure
    positions = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
    species = ['B', 'A']
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001
    test_structure = struc.Structure(cell, species, positions, cutoff)

    # create environment
    atom = 0
    toy_env = env.ChemicalEnvironment(test_structure, atom)

    yield toy_env

    del toy_env


# set up two test environments
@pytest.fixture(scope='module')
def env1():
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    positions_1 = [np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])]
    species_1 = ['B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
    env1 = env.ChemicalEnvironment(test_structure_1, atom_1)

    yield env1

    del env1


@pytest.fixture(scope='module')
def env2():
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    positions_2 = [np.array([0, 0, 0]), np.array([0.25, 0.3, 0.4])]
    species_2 = ['B', 'A']
    atom_2 = 0
    test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
    env2 = env.ChemicalEnvironment(test_structure_2, atom_2)

    yield env2

    del env2

# -----------------------------------------------------------------------------
#                          test env methods
# -----------------------------------------------------------------------------


def test_species_to_bond(test_env):
    is_bl_right = test_env.structure.bond_list ==\
        [['B', 'B'], ['B', 'A'], ['A', 'A']]
    assert(is_bl_right)


def test_is_bond():
    assert(env.ChemicalEnvironment.is_bond('A', 'B', ['A', 'B']))
    assert(env.ChemicalEnvironment.is_bond('B', 'A', ['A', 'B']))
    assert(not env.ChemicalEnvironment.is_bond('C', 'A', ['A', 'B']))


def test_is_triplet():
    assert(env.ChemicalEnvironment.is_triplet('A', 'B', 'C', ['A', 'B', 'C']))
    assert(env.ChemicalEnvironment.is_triplet('A', 'B', 'B', ['A', 'B', 'B']))
    assert(not env.ChemicalEnvironment.is_triplet('C', 'B', 'B',
                                                  ['A', 'B', 'B']))


def test_species_to_index(test_env):
    struc = test_env.structure
    assert(test_env.species_to_index(struc, 'B', 'B') == 0)
    assert(test_env.species_to_index(struc, 'B', 'A') == 1)
    assert(test_env.species_to_index(struc, 'A', 'A') == 2)


def test_triplet_to_index(test_env):
    assert(test_env.structure.triplet_list.index(['A', 'B', 'B']) ==
           test_env.triplet_to_index('A', 'B', 'B'))


def test_get_local_atom_images(test_env):
    vec = np.array([0.5, 0.5, 0.5])
    vecs, dists = test_env.get_local_atom_images(test_env.structure, vec)
    assert(len(dists) == 8)
    assert(len(vecs) == 8)


def test_get_atoms_within_cutoff(test_env):
    atom = 0
    bond_array, bond_positions, _, _ =\
        test_env.get_atoms_within_cutoff(test_env.structure, atom)

    assert(bond_array.shape[0] == 8)
    assert(bond_array[0, 1] == bond_positions[0, 0] / bond_array[0, 0])


def test_get_cross_bonds(test_env):
    nat = len(test_env.etyps)
    mrand = np.random.randint(0, nat)
    nrand = np.random.randint(0, nat)
    pos1 = test_env.bond_positions[mrand]
    pos2 = test_env.bond_positions[nrand]
    assert(test_env.cross_bond_dists[mrand, nrand] ==
           np.linalg.norm(pos1-pos2))
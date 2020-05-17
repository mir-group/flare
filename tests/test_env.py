import pytest
import numpy as np
from numpy import allclose
from flare.struc import Structure
from flare.env import AtomicEnvironment
from .fake_gp import generate_mb_envs

cutoff_list=[np.ones(2), np.ones(3)*0.8]

def test_species_count():
    cell = np.eye(3)
    species = [1, 2, 3]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    struc_test = Structure(cell, species, positions)
    env_test = AtomicEnvironment(structure=struc_test,
                                 atom=0,
                                 cutoffs=np.array([1, 1]))
    assert (len(struc_test.positions) == len(struc_test.coded_species))
    assert (len(env_test.bond_array_2) == len(env_test.etypes))
    assert (isinstance(env_test.etypes[0], np.int8))

def test_env_methods():
    cell = np.eye(3)
    species = [1, 2, 3]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    struc_test = Structure(cell, species, positions)
    env_test = AtomicEnvironment(struc_test, 0, np.array([1, 1, 0.9, 0.9]))
    assert str(env_test) == 'Atomic Env. of Type 1 surrounded by 12 atoms' \
                            ' of Types [2, 3]'

    the_dict = env_test.as_dict()
    assert isinstance(the_dict, dict)
    for key in ['positions', 'cell', 'atom', 'cutoffs', 'species']:
        assert key in the_dict.keys()

    remade_env = AtomicEnvironment.from_dict(the_dict)
    assert isinstance(remade_env, AtomicEnvironment)

    assert np.array_equal(remade_env.bond_array_2, env_test.bond_array_2)
    assert np.array_equal(remade_env.bond_array_3, env_test.bond_array_3)
    assert np.array_equal(remade_env.m2b_array, env_test.m2b_array)


def test_mb():
    delta = 1e-4
    tol = 1e-4
    cell = 1e7 * np.eye(3)
    cutoffs = np.ones(4)*1.2

    np.random.seed(10)
    atom = 0
    d1 = 1
    env_test = generate_mb_envs(cutoffs, cell, delta, d1=d1, kern_type='mc')
    env_0 = env_test[0][atom]
    env_p = env_test[1][atom]
    env_m = env_test[2][atom]
    ctype = env_0.ctype

    # test m2b
    mb_grads_analytic = env_0.m2b_neigh_grads[:, d1-1]

    s_p = np.where(env_p.m2b_unique_species==ctype)[0][0]
    p_neigh_array = env_p.m2b_neigh_array[:, s_p]

    s_m = np.where(env_m.m2b_unique_species==ctype)[0][0]
    m_neigh_array = env_m.m2b_neigh_array[:, s_m]

    mb_grads_finitediff = (p_neigh_array - m_neigh_array) / (2 * delta)
    assert(allclose(mb_grads_analytic, mb_grads_finitediff))

    # test m3b
    mb_grads_analytic = env_0.m3b_grads[:, :, d1-1]
    mb_neigh_grads_analytic = env_0.m3b_neigh_grads[:, :, d1-1]

    s_p = np.where(env_p.m3b_unique_species==ctype)[0][0]
    p_array = env_p.m3b_array
    p_neigh_array = env_p.m3b_neigh_array[:, s_p, :]

    s_m = np.where(env_m.m3b_unique_species==ctype)[0][0]
    m_array = env_m.m3b_array
    m_neigh_array = env_m.m3b_neigh_array[:, s_m, :]

    mb_grads_finitediff = (p_array - m_array) / (2 * delta)
    assert(allclose(mb_grads_analytic, mb_grads_finitediff))
    print(mb_grads_analytic, mb_grads_finitediff)

    for n in range(p_neigh_array.shape[0]):
        mb_neigh_grads_finitediff = (p_neigh_array[n] - m_neigh_array[n]) / (2 * delta)
#        if env_p.etypes[n] == ctype:
#            mb_neigh_grads_finitediff /= 2
        assert(allclose(mb_neigh_grads_analytic[n], mb_neigh_grads_finitediff))
        print(mb_neigh_grads_analytic[n], mb_neigh_grads_finitediff)

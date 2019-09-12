import pytest
import numpy as np
from flare.struc import Structure
from flare.env import AtomicEnvironment


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
    env_test = AtomicEnvironment(struc_test, 0, np.array([1, 1]))
    assert str(env_test) == 'Atomic Env. of Type 1 surrounded by 12 atoms' \
                            ' of Types [2, 3]'

    the_dict = env_test.as_dict()
    assert isinstance(the_dict, dict)
    for key in ['positions', 'cell', 'atom', 'cutoff_2', 'cutoff_3',
                'species']:
        assert key in the_dict.keys()

    remade_env = AtomicEnvironment.from_dict(the_dict)
    assert isinstance(remade_env, AtomicEnvironment)

    assert np.array_equal(remade_env.bond_array_2, env_test.bond_array_2)
    assert np.array_equal(remade_env.bond_array_3, env_test.bond_array_3)

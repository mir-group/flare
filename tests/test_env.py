import pytest
import numpy as np
from flare.struc import Structure
from flare.env import AtomicEnvironment

cutoff_list=[np.ones(2), np.ones(3)*0.8]

@pytest.mark.parametrize('cutoff', cutoff_list)
def test_species_count(cutoff):
    cell = np.eye(3)
    species = [1, 2, 3]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    struc_test = Structure(cell, species, positions)
    env_test = AtomicEnvironment(structure=struc_test, atom=0,
                                 cutoffs=np.array([1, 1]))
    assert (len(struc_test.positions) == len(struc_test.coded_species))
    assert (len(env_test.bond_array_2) == len(env_test.etypes))
    assert (isinstance(env_test.etypes[0], np.int8))


@pytest.mark.parametrize('cutoff', cutoff_list)
def test_env_methods(cutoff):
    cell = np.eye(3)
    species = [1, 2, 3]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    struc_test = Structure(cell, species, positions)
    env_test = AtomicEnvironment(struc_test, 0, np.array([1, 1]))
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
    assert np.array_equal(remade_env.bond_array_mb, env_test.bond_array_mb)


def test_auto_sweep():
    """Test that the number of neighbors inside the local environment is
        correctly computed."""

    # Make an arbitrary non-cubic structure.
    cell = np.array([[1.3, 0.5, 0.8],
                     [-1.2, 1, 0.73],
                     [-0.8, 0.1, 0.9]])
    positions = np.array([[1.2, 0.7, 2.3],
                          [3.1, 2.5, 8.9],
                          [-1.8, -5.8, 3.0],
                          [0.2, 1.1, 2.1],
                          [3.2, 1.1, 3.3]])
    species = np.array([1, 2, 3, 4, 5])
    arbitrary_structure = Structure(cell, species, positions)

    # Construct an environment.
    cutoffs = np.array([4., 3.])
    arbitrary_environment = \
        AtomicEnvironment(arbitrary_structure, 0, cutoffs)

    # Count the neighbors.
    n_neighbors_1 = len(arbitrary_environment.etypes)

    # Reduce the sweep value, and check that neighbors are missing.
    sweep_val = arbitrary_environment.sweep_val
    arbitrary_environment.sweep_array = \
        np.arange(-sweep_val + 1, sweep_val, 1)
    arbitrary_environment.compute_env()
    n_neighbors_2 = len(arbitrary_environment.etypes)
    assert(n_neighbors_1 > n_neighbors_2)

    # Increase the sweep value, and check that the count is the same.
    arbitrary_environment.sweep_array = \
        np.arange(-sweep_val - 1, sweep_val + 2, 1)
    arbitrary_environment.compute_env()
    n_neighbors_3 = len(arbitrary_environment.etypes)
    assert(n_neighbors_1 == n_neighbors_3)

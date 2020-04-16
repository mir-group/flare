import pytest
import numpy as np
from flare.struc import Structure
from flare.env import AtomicEnvironment

cutoff_list=[np.array([1]), np.array([1, 0.8]), np.array([1, 0.8, 0.4])]
mask_list = [True, False]

@pytest.mark.parametrize('cutoff', cutoff_list)
@pytest.mark.parametrize('mask', mask_list)
def test_species_count(cutoff, mask):

    cell = np.eye(3)
    species = [1, 2, 3, 1]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5],
                          [0.1, 0.1, 0.1], [1, 1, 1]])
    struc_test = Structure(cell, species, positions)

    if (mask is True):
        mask = generate_mask(cutoff)
    else:
        mask = None

    env_test = AtomicEnvironment(structure=struc_test,
                                 atom=0,
                                 cutoffs=np.array([1, 1]),
                                 cutoffs_mask=mask)
    assert (len(struc_test.positions) == len(struc_test.coded_species))
    assert (len(env_test.bond_array_2) == len(env_test.etypes))
    assert (isinstance(env_test.etypes[0], np.int8))


@pytest.mark.parametrize('cutoff', cutoff_list)
@pytest.mark.parametrize('mask', mask_list)
def test_env_methods(cutoff, mask):
    cell = np.eye(3)
    species = [1, 2, 3, 1]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5],
                          [0.1, 0.1, 0.1], [1, 1, 1]])
    struc_test = Structure(cell, species, positions)

    if (mask is True):
        mask = generate_mask(cutoff)
    else:
        mask = None

    env_test = AtomicEnvironment(struc_test, 0, np.array([1, 1]), mask)

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

@pytest.mark.parametrize('cutoff', cutoff_list)
def test_env_methods(cutoff):
    cell = np.eye(3)
    species = [1, 2, 3, 1]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5],
                          [0.1, 0.1, 0.1], [1, 1, 1]])
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

def generate_mask(cutoff):
    ncutoff = len(cutoff)
    mask = {'nspec': 2, 'spec_mask':np.zeros(118, dtype=int)}
    mask['spec_mask'][2] = 1
    if (ncutoff == 1):
        mask['cutoff_2b'] = np.ones(2)*cutoff[0]
        mask['nbond'] = 2
        mask['bond_mask'] = np.ones(4, dtype=int)
        mask['bond_mask'][0] = 1
    elif (ncutoff == 2):
        mask['cutoff_3b'] = np.ones(2)*cutoff[1]
        mask['ntriplet'] = 2
        mask['triplet_mask'] = np.ones(4, dtype=int)
        mask['triplet_mask'][0] = 1
    elif (ncutoff == 3):
        mask['cutoff_mb'] = np.ones(2)*cutoff[2]
        mask['nmb'] = 2
        mask['mb_mask'] = np.ones(4, dtype=int)
        mask['mb_mask'][0] = 1
    return mask

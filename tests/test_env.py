import pytest
import numpy as np
from flare.struc import Structure
from flare.env import AtomicEnvironment


np.random.seed(0)

# cutoff_list = [np.array([1]), np.array([1, 0.8]), np.array([1, 0.8, 0.4])]
# mask_list = [True, False]

cutoff_list = [np.array([1, 0.05]), np.array([1, 0.9])] #, np.array([1, 0.8, 0.4])]
mask_list = [True]

@pytest.fixture(scope='module')
def structure() -> Structure:
    """Returns a GP instance with a two-body numba-based kernel"""
    cell = np.eye(3)
    species = [1, 2, 3, 1]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5],
                          [0.1, 0.1, 0.1], [0.75, 0.75, 0.75]])
    struc_test = Structure(cell, species, positions)

    yield struc_test
    del struc_test


@pytest.mark.parametrize('cutoff', cutoff_list)
@pytest.mark.parametrize('mask', mask_list)
def test_species_count(structure, cutoff, mask):

    if (mask is True):
        mask = generate_mask(cutoff)
    else:
        mask = None

    env_test = AtomicEnvironment(structure=structure,
                                 atom=0,
                                 cutoffs=cutoff,
                                 cutoffs_mask=mask)
    assert (len(structure.positions) == len(structure.coded_species))
    assert (len(env_test.bond_array_2) == len(env_test.etypes))
    assert (isinstance(env_test.etypes[0], np.int8))


@pytest.mark.parametrize('cutoff', cutoff_list)
@pytest.mark.parametrize('mask', mask_list)
def test_env_methods(structure, cutoff, mask):

    if (mask is True):
        mask = generate_mask(cutoff)
    else:
        mask = None

    env_test = AtomicEnvironment(structure, 0, cutoff, mask)

    print(str(env_test))
    assert str(env_test) == 'Atomic Env. of Type 1 surrounded by 16 atoms' \
                            ' of Types [1, 2, 3]'

    the_dict = env_test.as_dict()
    assert isinstance(the_dict, dict)
    for key in ['positions', 'cell', 'atom', 'cutoffs', 'species']:
        assert key in the_dict.keys()

    remade_env = AtomicEnvironment.from_dict(the_dict)
    assert isinstance(remade_env, AtomicEnvironment)

    assert np.array_equal(remade_env.bond_array_2, env_test.bond_array_2)
    if (len(cutoff)>1):
        assert np.array_equal(remade_env.bond_array_3, env_test.bond_array_3)
    if (len(cutoff)>2):
        assert np.array_equal(remade_env.bond_array_mb, env_test.bond_array_mb)


def generate_mask(cutoff):
    ncutoff = len(cutoff)
    if (ncutoff == 1):
        mask = {'nspec': 2, 'spec_mask': np.ones(118, dtype=int)}
        mask['spec_mask'][1] = 0
        mask['cutoff_2b'] = np.ones(2)*cutoff[0]
        mask['nbond'] = 2
        mask['bond_mask'] = np.ones(4, dtype=int)
        mask['bond_mask'][0] = 1

    elif (ncutoff == 2): # the 3b mask is the same structure as 2b
        nspec = 3
        ntriplet = 3 # number of bond cutoffs 
        mask = {'nspec': nspec, 
                'spec_mask': np.zeros(118, dtype=int),
                'cutoff_3b': np.array([0.2, 0.5, 0.9]),
                'ntriplet': ntriplet,
                'triplet_mask': np.zeros(nspec**2, dtype=int)}
        spec_mask = mask['spec_mask']
        chem_spec = [1, 2, 3]
        spec_mask[chem_spec] = np.arange(3)

        tmask = mask['triplet_mask']
        for cs1 in range(nspec):
            for cs2 in range(cs1, nspec):
                ttype = np.random.randint(ntriplet)
                tmask[cs1*nspec+cs2] = ttype
                tmask[cs2*nspec+cs1] = ttype

    elif (ncutoff == 3):
        mask['cutoff_mb'] = np.ones(2)*cutoff[2]
        mask['nmb'] = 2
        mask['mb_mask'] = np.ones(4, dtype=int)
        mask['mb_mask'][0] = 1
    return mask

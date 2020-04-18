import pytest
import numpy as np
from flare.struc import Structure
from flare.env import AtomicEnvironment


np.random.seed(0)

cutoff_mask_list = [(True, np.array([1]), [10]),
                    (False, np.array([1]), [16]),
                    (False, np.array([1, 0.05]), [16, 0]),
                    (False, np.array([1, 0.8]), [16, 1]),
                    (False, np.array([1, 0.9]), [16, 21]),
                    (True, np.array([1, 0.8]), [16, 9]),
                    (True, np.array([1, 0.05, 0.4]), [16, 0]),
                    (False, np.array([1, 0.05, 0.4]), [16, 0])]


@pytest.fixture(scope='module')
def structure() -> Structure:
    """
    Returns a GP instance with a two-body numba-based kernel
    """

    # list of all bonds and triplets can be found in test_files/test_env_list
    cell = np.eye(3)
    species = [1, 2, 3, 1]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5],
                          [0.1, 0.1, 0.1], [0.75, 0.75, 0.75]])
    struc_test = Structure(cell, species, positions)

    yield struc_test
    del struc_test


@pytest.mark.parametrize('mask, cutoff, result', cutoff_mask_list)
def test_2bspecies_count(structure, mask, cutoff, result):

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
    assert (len(env_test.bond_array_2) == result[0])

    if (len(cutoff) > 1):
        assert (np.sum(env_test.triplet_counts) == result[1])


@pytest.mark.parametrize('mask, cutoff, result', cutoff_mask_list)
def test_env_methods(structure, mask, cutoff, result):

    if (mask is True):
        mask = generate_mask(cutoff)
    else:
        mask = None

    env_test = AtomicEnvironment(structure,
                                 atom=0,
                                 cutoffs=cutoff,
                                 cutoffs_mask=mask)

    assert str(env_test) == f'Atomic Env. of Type 1 surrounded by {result[0]} atoms' \
                            ' of Types [1, 2, 3]'

    the_dict = env_test.as_dict()
    assert isinstance(the_dict, dict)
    for key in ['positions', 'cell', 'atom', 'cutoffs', 'species']:
        assert key in the_dict.keys()

    remade_env = AtomicEnvironment.from_dict(the_dict)
    assert isinstance(remade_env, AtomicEnvironment)

    assert np.array_equal(remade_env.bond_array_2, env_test.bond_array_2)
    if (len(cutoff) > 1):
        assert np.array_equal(remade_env.bond_array_3, env_test.bond_array_3)
    if (len(cutoff) > 2):
        assert np.array_equal(remade_env.bond_array_mb, env_test.bond_array_mb)


def generate_mask(cutoff):
    ncutoff = len(cutoff)
    if (ncutoff == 1):
        # (1, 1) uses 0.5 cutoff,  (1, 2) (1, 3) (2, 3) use 0.9 cutoff
        mask = {'nspec': 2, 'spec_mask': np.ones(118, dtype=int)}
        mask['spec_mask'][1] = 0
        mask['cutoff_2b'] = np.array([0.5, 0.9])
        mask['nbond'] = 2
        mask['bond_mask'] = np.ones(4, dtype=int)
        mask['bond_mask'][0] = 0

    elif (ncutoff == 2):
        # the 3b mask is the same structure as 2b
        nspec = 3
        spec_mask = np.zeros(118, dtype=int)
        chem_spec = [1, 2, 3]
        spec_mask[chem_spec] = np.arange(3)

        # from type 1 to 4 is
        # (1, 1) (1, 2) (1, 3) (2, 3) (*, *)
        # correspond to cutoff 0.5, 0.9, 0.8, 0.9, 0.05
        ncut3b = 5
        tmask = np.ones(nspec**2, dtype=int)*(ncut3b-1)
        count = 0
        for i, j in [(1, 1), (1, 2), (1, 3), (2, 3)]:
            cs1 = spec_mask[i]
            cs2 = spec_mask[j]
            tmask[cs1*nspec+cs2] = count
            tmask[cs2*nspec+cs1] = count
            count += 1

        mask = {'nspec': nspec,
                'spec_mask': spec_mask,
                'cutoff_3b': np.array([0.5, 0.9, 0.8, 0.9, 0.05]),
                'ncut3b': ncut3b,
                'cut3b_mask': tmask}

    elif (ncutoff == 3):
        # (1, 1) uses 0.5 cutoff,  (1, 2) (1, 3) (2, 3) use 0.9 cutoff
        mask = {'nspec': 2, 'spec_mask': np.ones(118, dtype=int)}
        mask['spec_mask'][1] = 0
        mask['cutoff_mb'] = np.array([0.5, 0.9])
        mask['nmb'] = 2
        mask['mb_mask'] = np.ones(4, dtype=int)
        mask['mb_mask'][0] = 0
    return mask

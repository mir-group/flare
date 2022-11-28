import pytest
import numpy as np

from json import dumps
from copy import deepcopy
from os import remove
from flare.atoms import FLARE_Atoms
from flare.descriptors.env import AtomicEnvironment
from flare.utils import NumpyEncoder

np.random.seed(0)

cutoff_mask_list = [  # (True, np.array([1]), [10]),
    (False, {"twobody": 1}, [16]),
    (False, {"twobody": 1, "threebody": 0.05}, [16, 0]),
    (False, {"twobody": 1, "threebody": 0.8}, [16, 1]),
    (False, {"twobody": 1, "threebody": 0.9}, [16, 21]),
    (True, {"twobody": 1, "threebody": 0.8}, [16, 9]),
    (True, {"twobody": 1, "threebody": 0.05, "manybody": 0.4}, [16, 0]),
    (False, {"twobody": 1, "threebody": 0.05, "manybody": 0.4}, [16, 0]),
]


@pytest.fixture(scope="module")
def structure() -> FLARE_Atoms:
    """
    Returns a GP instance with a two-body numba-based kernel
    """

    # list of all twobodys and threebodys can be found in test_files/test_env_list
    cell = np.eye(3)
    species = [1, 2, 3, 1]
    positions = np.array(
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.75, 0.75, 0.75]]
    )
    struc_test = FLARE_Atoms(symbols=species, positions=positions, cell=cell)

    yield struc_test
    del struc_test


@pytest.mark.parametrize("mask, cutoff, result", cutoff_mask_list)
def test_2bspecies_count(structure, mask, cutoff, result):
    if mask is True:
        mask = generate_mask(cutoff)
    else:
        mask = None

    env_test = AtomicEnvironment(
        structure=structure, atom=0, cutoffs=cutoff, cutoffs_mask=mask
    )
    assert len(structure.positions) == len(structure.numbers)
    print(env_test.__dict__)

    assert len(env_test.bond_array_2) == len(env_test.etypes)
    assert isinstance(env_test.etypes[0], np.int8)
    assert len(env_test.bond_array_2) == result[0]

    if len(cutoff) > 1:
        assert np.sum(env_test.triplet_counts) == result[1]


@pytest.mark.parametrize("mask, cutoff, result", cutoff_mask_list)
def test_env_methods(structure, mask, cutoff, result):
    if mask is True:
        mask = generate_mask(cutoff)
    else:
        mask = None

    structure = deepcopy(structure)
    structure.forces = np.random.random(size=(len(structure), 3))

    env_test = AtomicEnvironment(structure, atom=0, cutoffs=cutoff, cutoffs_mask=mask)

    assert np.array_equal(structure.forces[0], env_test.force)

    assert (
        str(env_test)
        == f"Atomic Env. of Type 1 surrounded by {result[0]} atoms of Types [1, 2, 3]"
    )

    the_dict = env_test.as_dict()
    assert isinstance(the_dict, dict)
    the_str = env_test.as_str()
    assert dumps(the_dict, cls=NumpyEncoder) == the_str
    for key in ["positions", "cell", "atom", "cutoffs", "species"]:
        assert key in the_dict.keys()

    # This saves a few seconds, the masked envs take longer to read/write
    if not mask:
        with open("test_environment.json", "w") as f:
            f.write(env_test.as_str())
        remade_env = AtomicEnvironment.from_file("test_environment.json")
    else:
        remade_env = AtomicEnvironment.from_dict(the_dict)

    assert isinstance(remade_env, AtomicEnvironment)

    assert np.array_equal(remade_env.bond_array_2, env_test.bond_array_2)
    if len(cutoff) > 1:
        assert np.array_equal(remade_env.bond_array_3, env_test.bond_array_3)
    if len(cutoff) > 2:
        assert np.array_equal(remade_env.q_array, env_test.q_array)

    if not mask:
        remove("test_environment.json")


# Only run on the first four tests, namely, the ones which have
# False as a parameter.
@pytest.mark.parametrize("mask, cutoff, result", cutoff_mask_list[:4])
def test_backwards_compatibility(structure, mask, cutoff, result):
    """
    This test can be deleted if backwards compatibility is dropped for the
    sake of code cleanup. (This test executes in about 5 milliseconds). Tests a
    particular branch of code within the Environment's as_dict() method for
    older pickled environments without a cutoffs mask.
    :return:
    """
    if mask is True:
        mask = generate_mask(cutoff)
    else:
        mask = None

    env_test = deepcopy(
        AtomicEnvironment(structure, atom=0, cutoffs=cutoff, cutoffs_mask=mask)
    )
    pre_test_dict = env_test.as_dict()

    delattr(env_test, "cutoffs_mask")

    test_dict = env_test.as_dict()

    assert pre_test_dict["cutoffs_mask"] == test_dict["cutoffs_mask"]

    new_env = AtomicEnvironment.from_dict(test_dict)

    assert isinstance(new_env, AtomicEnvironment)

    assert str(new_env) == str(env_test)


def generate_mask(cutoff):
    ncutoff = len(cutoff)
    if ncutoff == 1:
        # (1, 1) uses 0.5 cutoff,  (1, 2) (1, 3) (2, 3) use 0.9 cutoff
        mask = {"nspecie": 2, "species_mask": np.ones(118, dtype=int)}
        mask["species_mask"][1] = 0
        mask["twobody_cutoff_list"] = np.array([0.5, 0.9])
        mask["ntwobody"] = 2
        mask["twobody_mask"] = np.ones(4, dtype=int)
        mask["twobody_mask"][0] = 0

    elif ncutoff == 2:
        # the 3b mask is the same structure as 2b
        nspecie = 3
        species_mask = np.zeros(118, dtype=int)
        chem_spec = [1, 2, 3]
        species_mask[chem_spec] = np.arange(3)

        # from type 1 to 4 is
        # (1, 1) (1, 2) (1, 3) (2, 3) (*, *)
        # correspond to cutoff 0.5, 0.9, 0.8, 0.9, 0.05
        ncut3b = 5
        tmask = np.ones(nspecie**2, dtype=int) * (ncut3b - 1)
        count = 0
        for i, j in [(1, 1), (1, 2), (1, 3), (2, 3)]:
            cs1 = species_mask[i]
            cs2 = species_mask[j]
            tmask[cs1 * nspecie + cs2] = count
            tmask[cs2 * nspecie + cs1] = count
            count += 1

        mask = {
            "nspecie": nspecie,
            "species_mask": species_mask,
            "threebody_cutoff_list": np.array([0.5, 0.9, 0.8, 0.9, 0.05]),
            "ncut3b": ncut3b,
            "cut3b_mask": tmask,
        }

    elif ncutoff == 3:
        # (1, 1) uses 0.5 cutoff,  (1, 2) (1, 3) (2, 3) use 0.9 cutoff
        mask = {"nspecie": 2, "species_mask": np.ones(118, dtype=int)}
        mask["species_mask"][1] = 0
        mask["manybody_cutoff_list"] = np.array([0.5, 0.9])
        mask["nmanybody"] = 2
        mask["manybody_mask"] = np.ones(4, dtype=int)
        mask["manybody_mask"][0] = 0
    mask["cutoffs"] = cutoff
    return mask


def test_auto_sweep():
    """Test that the number of neighbors inside the local environment is
    correctly computed."""

    # Make an arbitrary non-cubic structure.
    cell = np.array([[1.3, 0.5, 0.8], [-1.2, 1, 0.73], [-0.8, 0.1, 0.9]])
    positions = np.array(
        [
            [1.2, 0.7, 2.3],
            [3.1, 2.5, 8.9],
            [-1.8, -5.8, 3.0],
            [0.2, 1.1, 2.1],
            [3.2, 1.1, 3.3],
        ]
    )
    species = np.array([1, 2, 3, 4, 5])
    arbitrary_structure = FLARE_Atoms(symbols=species, positions=positions, cell=cell)

    # Construct an environment.
    cutoffs = np.array([4.0, 3.0])
    arbitrary_environment = AtomicEnvironment(arbitrary_structure, 0, cutoffs)

    # Count the neighbors.
    n_neighbors_1 = len(arbitrary_environment.etypes)

    # Reduce the sweep value, and check that neighbors are missing.
    sweep_val = arbitrary_environment.sweep_val
    arbitrary_environment.sweep_array = np.arange(-sweep_val + 1, sweep_val, 1)
    arbitrary_environment.compute_env()
    n_neighbors_2 = len(arbitrary_environment.etypes)
    assert n_neighbors_1 > n_neighbors_2

    # Increase the sweep value, and check that the count is the same.
    arbitrary_environment.sweep_array = np.arange(-sweep_val - 1, sweep_val + 2, 1)
    arbitrary_environment.compute_env()
    n_neighbors_3 = len(arbitrary_environment.etypes)
    assert n_neighbors_1 == n_neighbors_3

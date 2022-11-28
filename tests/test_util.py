import numpy as np
import pytest

from pytest import raises

from flare.atoms import FLARE_Atoms
from flare.learners.utils import (
    is_std_in_bound_per_species,
    is_force_in_bound_per_species,
    subset_of_frame_by_element,
)

from tests.test_gp import get_random_structure


def test_std_in_bound_per_species():
    test_structure, _ = get_random_structure(np.eye(3), ["H", "O"], 3)
    test_structure.symbols = ["H", "H", "O"]
    test_structure.stds = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    # Test that 'test mode' works
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=0, abs_std_tolerance=0, noise=0, structure=test_structure
    )
    assert result is True and target_atoms == [-1]
    # Test that high abs tolerance works
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=0, abs_std_tolerance=4, noise=0, structure=test_structure
    )
    assert result is True and target_atoms == [-1]
    # Test that low abs tolerance works
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=0, abs_std_tolerance=2.9, noise=0, structure=test_structure
    )
    assert result is False and target_atoms == [2]
    # Test that high rel tolerance works
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1, abs_std_tolerance=0, noise=4, structure=test_structure
    )
    assert result is True and target_atoms == [-1]
    # Test that low rel tolerance works
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1, abs_std_tolerance=0, noise=2.9, structure=test_structure
    )
    assert result is False and target_atoms == [2]
    # Test that both work
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1, abs_std_tolerance=0.1, noise=2.9, structure=test_structure
    )
    assert result is False and target_atoms == [2, 1, 0]
    # Test that the max atoms added works
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1,
        abs_std_tolerance=0.1,
        noise=2.9,
        structure=test_structure,
        max_atoms_added=2,
    )
    assert result is False and target_atoms == [2, 1]
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1,
        abs_std_tolerance=0.1,
        noise=2.9,
        structure=test_structure,
        max_atoms_added=1,
    )
    assert result is False and target_atoms == [2]

    # Test that max by species works
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1,
        abs_std_tolerance=0.1,
        noise=2.9,
        structure=test_structure,
        max_atoms_added=1,
        max_by_species={"H": 1, "O": 0},
    )
    assert result is False and target_atoms == [1]

    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1,
        abs_std_tolerance=0.1,
        noise=2.9,
        structure=test_structure,
        max_atoms_added=1,
        max_by_species={"H": 0, "O": 1},
    )
    assert result is False and target_atoms == [2]

    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1,
        abs_std_tolerance=0.1,
        noise=2.9,
        structure=test_structure,
        max_atoms_added=1,
        max_by_species={"H": 2, "O": 0},
    )
    assert result is False and target_atoms == [1]

    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1,
        abs_std_tolerance=0.1,
        noise=2.9,
        structure=test_structure,
        max_atoms_added=3,
        max_by_species={"H": 2, "O": 0},
    )
    assert result is False and target_atoms == [1, 0]

    # Ensure NANS can be handled
    test_structure.stds = np.array([[np.nan, np.nan, np.nan], [2, 0, 0], [3, 0, 0]])
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1,
        abs_std_tolerance=0.1,
        noise=2.9,
        structure=test_structure,
        max_atoms_added=3,
        max_by_species={"H": 2, "O": 0},
    )
    assert result is False and target_atoms == [1]

    test_structure.stds = np.array(
        [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [3, 0, 0]]
    )
    result, target_atoms = is_std_in_bound_per_species(
        rel_std_tolerance=1,
        abs_std_tolerance=0.1,
        noise=2.9,
        structure=test_structure,
        max_atoms_added=3,
        max_by_species={"H": 2, "O": 0},
    )
    assert result is True and target_atoms == [-1]


def test_force_in_bound_per_species():
    test_structure, _ = get_random_structure(np.eye(3), ["H", "O"], 3)
    test_structure.symbols = ["H", "H", "O"]
    test_structure.forces = np.array([[0, 0, 0], [2, 0, 0], [3, 0, 0]])
    true_forces = np.array([[0.001, 0, 0], [1, 0, 0], [5, 0, 0]])

    # Test that 'test mode' works

    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=0,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
    )

    assert result is True and target_atoms == [-1]

    # Test that high abs tolerance works
    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=10,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
    )

    assert result is True and target_atoms == [-1]

    # Test that low abs tolerance works
    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=1.5,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
    )
    assert result is False and target_atoms == [2]

    # Test that the max atoms added works
    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=0.01,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
        max_atoms_added=2,
    )
    assert result is False and target_atoms == [2, 1]

    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=0.00001,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
        max_atoms_added=1,
    )
    assert result is False and target_atoms == [2]

    # Test that max by species works
    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=0.00001,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
        max_atoms_added=1,
        max_by_species={"H": 1, "O": 0},
    )
    assert result is False and target_atoms == [1]

    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=0.00001,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
        max_atoms_added=1,
        max_by_species={"H": 0, "O": 1},
    )
    assert result is False and target_atoms == [2]

    # Test max force error feature
    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=0.9,
        max_force_error=0.00000000001,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
    )
    assert result is True and target_atoms == [-1]

    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=0.9,
        max_force_error=1.5,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
    )
    assert result is False and target_atoms == [1]

    # Ensure that nans can be handled
    test_structure.forces = np.array([[np.nan, np.nan, np.nan], [2, 0, 0], [3, 0, 0]])
    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=1,
        max_force_error=1.5,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
    )
    assert result is False and target_atoms == [1]

    result, target_atoms = is_force_in_bound_per_species(
        abs_force_tolerance=0.9,
        max_force_error=500,
        predicted_forces=test_structure.forces,
        label_forces=true_forces,
        structure=test_structure,
    )
    assert result is False and set(target_atoms) == set([1, 2])


def test_subset_of_frame_by_element():
    spec_list = ["H", "H", "O", "O", "O", "C"]
    test_struc_1 = FLARE_Atoms(
        cell=np.eye(3), symbols=spec_list, positions=np.zeros(shape=(len(spec_list), 3))
    )

    assert np.array_equal(
        subset_of_frame_by_element(test_struc_1, {}), list(range(len(test_struc_1)))
    )

    assert np.array_equal(
        subset_of_frame_by_element(test_struc_1, {"H": 2, "O": 3}),
        list(range(len(test_struc_1))),
    )
    assert np.array_equal(
        subset_of_frame_by_element(test_struc_1, {"H": 2, "O": 15}),
        list(range(len(test_struc_1))),
    )

    assert set(subset_of_frame_by_element(test_struc_1, {"H": 1, "O": 1})).issubset(
        range(len(spec_list))
    )
    assert len(subset_of_frame_by_element(test_struc_1, {"H": 1, "O": 1, "C": 1})) == 3

    assert subset_of_frame_by_element(test_struc_1, {"H": 0, "O": 0, "C": 0}) == []

    assert subset_of_frame_by_element(test_struc_1, {"H": 0, "O": 0, "C": 1}) == [5]

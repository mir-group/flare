import numpy as np
import pytest
import torch
from flare.tensor.neighbors import (
    get_neighbors_ase,
    get_neighbors_brute_force,
    get_neighbors_minimum_image,
    mic_safe,
    get_supercell_containing_rcut,
)
from flare.tensor.b2 import get_edge_dist

torch.manual_seed(1)


@pytest.fixture(scope="module")
def all_neighbors():
    structure_list = {
        "cells": [],
        "positions": [],
    }
    n_random_structures = 100
    for n in range(n_random_structures):
        n_atoms = 100
        cell = 5 * torch.randn(3, 3, dtype=torch.float64)
        positions = torch.randn(n_atoms, 3, dtype=torch.float64) * 1000
        structure_list["cells"].append(cell)
        structure_list["positions"].append(positions)

    yield structure_list
    del structure_list


def test_cubic_neighbors():
    """
    Verify that a single atom in a unit cube with full periodic boundary
    conditions has six nearest neighbors with the expected lattice shifts.
    """
    pbc = (True,) * 3
    cell = np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0]])
    cutoff = 1.001

    first_index, second_index, shifts = get_neighbors_ase(positions, cell, cutoff, pbc)

    assert np.all(first_index == 0)
    assert np.all(second_index == 0)
    assert len(shifts) == 6

    expected_shifts = {
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    }
    actual_shifts = {tuple(s) for s in shifts}
    assert actual_shifts == expected_shifts


def test_neighbor_list(all_neighbors):
    cells = all_neighbors["cells"]
    positions = all_neighbors["positions"]
    cutoff = 1.0
    for cell, pos in zip(cells, positions):
        repetitions = get_supercell_containing_rcut(cell, cutoff)
        if torch.prod(repetitions) > 100:
            continue

        first_index, second_index, shift_vec = get_neighbors_brute_force(
            cell, pos, cutoff
        )

        _, edge_dist = get_edge_dist(
            pos, cell, first_index, second_index, shift_vec
        )
        for dist in edge_dist:
            if dist > cutoff:
                print(dist)
        assert torch.all(edge_dist < cutoff), "Found edge distances >= cutoff"

        first_index_mi, second_index_mi, shift_vec_mi = get_neighbors_minimum_image(
            cell, pos, cutoff
        )

        _, edge_dist_mi = get_edge_dist(
            pos, cell, first_index_mi, second_index_mi, shift_vec_mi
        )
        assert torch.all(edge_dist_mi < cutoff), "Found edge distances >= cutoff"

        mic_check = mic_safe(cell, cutoff)
        if mic_check:
            assert len(first_index) == len(
                first_index_mi
            ), "MIC neighbors don't match brute force"

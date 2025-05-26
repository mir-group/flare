import numpy as np
from flare.tensor.neighbors import get_neighbor_list


def test_cubic_neighbors():
    """
    Verify that a single atom in a unit cube with full periodic boundary
    conditions has six nearest neighbors with the expected lattice shifts.
    """
    pbc = (True,) * 3
    cell = np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0]])
    cutoff = 1.001

    first_index, second_index, shifts = get_neighbor_list(positions, cell, cutoff, pbc)

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

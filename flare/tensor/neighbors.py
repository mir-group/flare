import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list
from ase.geometry import complete_cell
import time


def get_neighbor_list(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    pbc: bool = True,
):
    """
    Compute the neighbor list for a set of atoms using periodic boundary conditions.

    Parameters:
        positions (np.ndarray): Array of shape (N_atoms, 3) with Cartesian coordinates.
        cell (np.ndarray): Array of shape (3, 3) representing the unit cell vectors.
        cutoff (float): Distance cutoff for neighbor searching.
        pbc (bool or tuple): Periodic boundary conditions. Can be a bool or a tuple of 3 bools.

    Returns:
        first_index (np.ndarray): Indices of source atoms in each neighbor pair.
        second_index (np.ndarray): Indices of target atoms in each neighbor pair.
        shifts (np.ndarray): Lattice shift vectors for each neighbor interaction.
    """
    pbc = (pbc,) * 3
    first_index, second_index, shifts = primitive_neighbor_list(
        "ijS",
        pbc,
        cell,
        positions,
        cutoff=cutoff,
        self_interaction=True,
        use_scaled_positions=False,
    )

    # remove self interactions
    bad_edge = first_index == second_index
    bad_edge &= np.all(shifts == 0, axis=1)
    keep_edge = ~bad_edge
    first_index = first_index[keep_edge]
    second_index = second_index[keep_edge]
    shifts = shifts[keep_edge]

    return first_index, second_index, shifts
